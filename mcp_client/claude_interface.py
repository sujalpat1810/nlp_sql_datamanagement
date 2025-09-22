"""
Claude API Interface for Natural Language Processing.

This module handles communication with Anthropic's Claude API for 
natural language understanding and SQL generation.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

from pydantic import BaseModel, ValidationError

# Import config for mock responses
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.mock_responses import get_mock_response_for_query, get_mock_explanation_for_query, simulate_api_delay
    from config.settings import get_settings
except ImportError:
    # Fallback if config not available
    get_mock_response_for_query = None
    get_mock_explanation_for_query = None
    simulate_api_delay = None
    get_settings = None

logger = logging.getLogger(__name__)


class NLQueryRequest(BaseModel):
    """Request model for natural language query processing."""
    query: str
    context: Optional[Dict[str, Any]] = None
    schema_info: Optional[Dict[str, Any]] = None
    previous_queries: Optional[List[str]] = None
    max_tokens: Optional[int] = 1000


class SQLQueryResponse(BaseModel):
    """Response model for generated SQL queries."""
    sql_query: str
    explanation: str
    confidence_score: float
    query_type: str  # SELECT, INSERT, UPDATE, DELETE, etc.
    estimated_complexity: str  # simple, moderate, complex
    warnings: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    parameters: Optional[List[Any]] = None
    estimated_rows: Optional[int] = None
    execution_time_estimate: Optional[float] = None
    tables_used: Optional[List[str]] = None


class ClaudeInterface:
    """Interface for communicating with Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229", mock_mode: bool = False):
        """
        Initialize Claude interface.
        
        Args:
            api_key: Anthropic API key (optional in mock mode)
            model: Claude model to use
            mock_mode: Whether to use mock responses instead of real API
        """
        self.mock_mode = mock_mode or not api_key or api_key == "test_api_key" or os.getenv("MOCK_CLAUDE_RESPONSES", "false").lower() == "true"
        
        if not self.mock_mode and AsyncAnthropic and api_key:
            self.client = AsyncAnthropic(api_key=api_key)
        else:
            self.client = None
            self.mock_mode = True
            logger.info("Claude interface initialized in mock mode")
            
        self.model = model
        self.system_prompt = self._build_system_prompt()
        
        # Load settings if available
        self.settings = get_settings() if get_settings else None
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for NLP to SQL conversion."""
        return """You are an expert SQL assistant that converts natural language queries into SQL statements.

Your role:
1. Convert natural language questions into correct SQL queries
2. Provide clear explanations of what the query does
3. Identify potential issues or optimizations
4. Suggest improvements when appropriate

Guidelines:
- Always generate syntactically correct SQL
- Use appropriate table and column names from the provided schema
- Consider query performance and optimization
- Provide confidence scores based on query clarity
- Flag potentially dangerous operations (DELETE, UPDATE without WHERE)
- Suggest indexes or schema improvements when relevant

Response Format:
Respond with a JSON object containing:
- sql_query: The generated SQL query
- explanation: Clear explanation of what the query does
- confidence_score: Float between 0.0 and 1.0
- query_type: Type of SQL operation (SELECT, INSERT, UPDATE, DELETE, etc.)
- estimated_complexity: "simple", "moderate", or "complex"
- warnings: Array of any warnings about the query
- suggestions: Array of optimization suggestions

Always prioritize safety and correctness over complexity."""
    
    async def process_natural_language_query(
        self, 
        request: NLQueryRequest
    ) -> SQLQueryResponse:
        """
        Process a natural language query and generate SQL.
        
        Args:
            request: Natural language query request
            
        Returns:
            SQL query response with metadata
        """
        try:
            # Check if using mock mode
            if self.mock_mode:
                return await self._get_mock_response(request)
            
            # Build context for Claude
            context_info = self._build_context(request)
            
            # Create the user prompt
            user_prompt = self._build_user_prompt(request, context_info)
            
            # Send request to Claude
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens or 1000,
                system=self.system_prompt,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }]
            )
            
            # Parse Claude's response
            response_text = response.content[0].text
            
            # Try to parse as JSON
            try:
                # Clean response text - remove any control characters
                import re
                cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', response_text)
                
                # Try to extract JSON if it's embedded in the response
                json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(0)
                
                response_data = json.loads(cleaned_text)
                sql_response = SQLQueryResponse(**response_data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Failed to parse structured response: {e}")
                # Fallback to extracting SQL from text response
                sql_response = self._extract_sql_fallback(response_text, request)
            
            # Validate and enhance the response
            sql_response = await self._validate_and_enhance_response(sql_response, request)
            
            logger.info(f"Generated SQL query with confidence {sql_response.confidence_score}")
            return sql_response
            
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            # Return a safe fallback response
            return SQLQueryResponse(
                sql_query="-- Error generating query",
                explanation=f"Failed to process query: {str(e)}",
                confidence_score=0.0,
                query_type="ERROR",
                estimated_complexity="unknown",
                warnings=[f"Query processing failed: {str(e)}"]
            )
    
    def _build_context(self, request: NLQueryRequest) -> Dict[str, Any]:
        """Build context information for the query."""
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "has_schema": bool(request.schema_info),
            "has_previous_queries": bool(request.previous_queries)
        }
        
        if request.context:
            context.update(request.context)
        
        return context
    
    def _build_user_prompt(self, request: NLQueryRequest, context: Dict[str, Any]) -> str:
        """Build the user prompt for Claude."""
        prompt_parts = [
            f"Natural Language Query: {request.query}",
            ""
        ]
        
        # Add schema information if available
        if request.schema_info:
            prompt_parts.extend([
                "Database Schema:",
                json.dumps(request.schema_info, indent=2),
                ""
            ])
        
        # Add previous queries for context
        if request.previous_queries:
            prompt_parts.extend([
                "Previous Queries (for context):",
                *[f"- {query}" for query in request.previous_queries[-3:]],  # Last 3 queries
                ""
            ])
        
        # Add additional context
        if request.context:
            prompt_parts.extend([
                "Additional Context:",
                json.dumps(request.context, indent=2),
                ""
            ])
        
        prompt_parts.append("Please convert this natural language query to SQL and respond with the JSON format specified in the system prompt.")
        
        return "\n".join(prompt_parts)
    
    def _extract_sql_fallback(self, response_text: str, request: NLQueryRequest) -> SQLQueryResponse:
        """Fallback method to extract SQL from unstructured response."""
        # Try to find SQL query in the response
        sql_query = self._extract_sql_from_text(response_text)
        
        # Determine query type
        query_type = self._determine_query_type(sql_query)
        
        return SQLQueryResponse(
            sql_query=sql_query,
            explanation=f"Generated from natural language query: {request.query}",
            confidence_score=0.7,  # Lower confidence for fallback
            query_type=query_type,
            estimated_complexity="moderate",
            warnings=["Response was not in expected JSON format"]
        )
    
    def _extract_sql_from_text(self, text: str) -> str:
        """Extract SQL query from text response."""
        import re
        
        # Look for SQL keywords and common patterns
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',  # Code blocks
            r'```\s*(SELECT.*?);?\s*```',  # SQL in code blocks
            r'(SELECT\s+.*?)(?:\n\n|\Z)',  # SELECT statements
            r'(INSERT\s+.*?)(?:\n\n|\Z)',  # INSERT statements
            r'(UPDATE\s+.*?)(?:\n\n|\Z)',  # UPDATE statements
            r'(DELETE\s+.*?)(?:\n\n|\Z)',  # DELETE statements
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(1).strip()
                if sql:
                    return sql
        
        # If no SQL found, return a placeholder
        return "-- Could not extract SQL query from response"
    
    def _determine_query_type(self, sql_query: str) -> str:
        """Determine the type of SQL query."""
        sql_upper = sql_query.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            return 'SELECT'
        elif sql_upper.startswith('INSERT'):
            return 'INSERT'
        elif sql_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif sql_upper.startswith('DELETE'):
            return 'DELETE'
        elif sql_upper.startswith('CREATE'):
            return 'CREATE'
        elif sql_upper.startswith('DROP'):
            return 'DROP'
        elif sql_upper.startswith('ALTER'):
            return 'ALTER'
        else:
            return 'UNKNOWN'
    
    async def _validate_and_enhance_response(
        self, 
        response: SQLQueryResponse, 
        request: NLQueryRequest
    ) -> SQLQueryResponse:
        """Validate and enhance the SQL response."""
        # Check for dangerous operations
        if response.query_type in ['DELETE', 'UPDATE']:
            if 'WHERE' not in response.sql_query.upper():
                if not response.warnings:
                    response.warnings = []
                response.warnings.append("DELETE/UPDATE query without WHERE clause - this will affect all rows!")
        
        # Check for DROP operations
        if response.query_type == 'DROP':
            if not response.warnings:
                response.warnings = []
            response.warnings.append("DROP operation detected - this will permanently delete database objects!")
        
        # Enhance explanations
        if not response.explanation or len(response.explanation) < 20:
            response.explanation = f"This {response.query_type} query was generated from: '{request.query}'"
        
        # Adjust confidence based on query complexity
        if 'JOIN' in response.sql_query.upper() and response.estimated_complexity == 'simple':
            response.estimated_complexity = 'moderate'
        
        if any(keyword in response.sql_query.upper() for keyword in ['SUBQUERY', 'CTE', 'WINDOW']):
            response.estimated_complexity = 'complex'
        
        return response
    
    async def explain_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Get an explanation of an existing SQL query.
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Explanation and analysis of the query
        """
        try:
            prompt = f"""Explain this SQL query in detail:

{sql_query}

Please provide:
1. A clear explanation of what this query does
2. Performance considerations
3. Potential improvements
4. Any warnings or issues

Respond in JSON format with fields: explanation, performance_notes, improvements, warnings."""

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system="You are an expert SQL analyst. Analyze and explain SQL queries clearly and thoroughly.",
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            response_text = response.content[0].text
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return {
                    "explanation": response_text,
                    "performance_notes": "Could not parse detailed analysis",
                    "improvements": [],
                    "warnings": []
                }
                
        except Exception as e:
            logger.error(f"Error explaining SQL query: {e}")
            return {
                "explanation": f"Error analyzing query: {str(e)}",
                "performance_notes": "Analysis failed",
                "improvements": [],
                "warnings": ["Failed to analyze query"]
            }
    
    async def suggest_optimizations(
        self, 
        sql_query: str, 
        schema_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest optimizations for a SQL query.
        
        Args:
            sql_query: SQL query to optimize
            schema_info: Database schema information
            
        Returns:
            List of optimization suggestions
        """
        try:
            schema_context = ""
            if schema_info:
                schema_context = f"\n\nDatabase Schema:\n{json.dumps(schema_info, indent=2)}"
            
            prompt = f"""Analyze this SQL query for optimization opportunities:

{sql_query}{schema_context}

Provide specific optimization suggestions with:
1. The optimization type (index, query rewrite, etc.)
2. Detailed explanation
3. Expected impact (low/medium/high)
4. Implementation notes

Respond in JSON format as an array of objects with fields: type, explanation, impact, implementation."""

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system="You are a database performance expert. Provide actionable optimization suggestions.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            response_text = response.content[0].text
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback to simple suggestions
                return [{
                    "type": "analysis",
                    "explanation": response_text,
                    "impact": "unknown",
                    "implementation": "See explanation above"
                }]
                
        except Exception as e:
            logger.error(f"Error suggesting optimizations: {e}")
            return [{
                "type": "error",
                "explanation": f"Failed to analyze query: {str(e)}",
                "impact": "unknown", 
                "implementation": "Fix the error first"
            }]
    
    async def validate_query_safety(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query for safety concerns.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Safety analysis results
        """
        safety_issues = []
        risk_level = "low"
        
        query_upper = sql_query.upper()
        
        # Check for dangerous operations
        if 'DELETE' in query_upper and 'WHERE' not in query_upper:
            safety_issues.append("DELETE without WHERE clause will remove all rows")
            risk_level = "critical"
        
        if 'UPDATE' in query_upper and 'WHERE' not in query_upper:
            safety_issues.append("UPDATE without WHERE clause will modify all rows")
            risk_level = "critical"
        
        if 'DROP' in query_upper:
            safety_issues.append("DROP statement will permanently delete database objects")
            if risk_level != "critical":
                risk_level = "high"
        
        if 'TRUNCATE' in query_upper:
            safety_issues.append("TRUNCATE will remove all data from the table")
            if risk_level not in ["critical", "high"]:
                risk_level = "high"
        
        # Check for potentially expensive operations
        if all(keyword in query_upper for keyword in ['SELECT', '*']) and 'LIMIT' not in query_upper:
            safety_issues.append("SELECT * without LIMIT may return large amounts of data")
            if risk_level == "low":
                risk_level = "medium"
        
        return {
            "is_safe": len(safety_issues) == 0 or risk_level == "low",
            "risk_level": risk_level,
            "issues": safety_issues,
            "recommendations": self._get_safety_recommendations(safety_issues)
        }
    
    def _get_safety_recommendations(self, issues: List[str]) -> List[str]:
        """Get safety recommendations based on identified issues."""
        recommendations = []
        
        for issue in issues:
            if "DELETE" in issue and "WHERE" in issue:
                recommendations.append("Add a WHERE clause to limit which rows are deleted")
            elif "UPDATE" in issue and "WHERE" in issue:
                recommendations.append("Add a WHERE clause to limit which rows are updated")
            elif "DROP" in issue:
                recommendations.append("Ensure you have backups before running DROP statements")
            elif "SELECT *" in issue:
                recommendations.append("Use specific column names and add LIMIT clause")
        
        if not recommendations:
            recommendations.append("Query appears safe to execute")
        
        return recommendations
    
    async def _get_mock_response(self, request: NLQueryRequest) -> SQLQueryResponse:
        """
        Generate mock response for testing without real API calls.
        
        Args:
            request: Natural language query request
            
        Returns:
            Mock SQL query response
        """
        logger.info(f"Generating mock response for query: {request.query}")
        
        # Simulate API delay if function is available
        if simulate_api_delay:
            delay = simulate_api_delay()
            await asyncio.sleep(delay)
        
        # Get mock response based on query content
        if get_mock_response_for_query:
            mock_data = get_mock_response_for_query(request.query)
        else:
            # Fallback mock response - analyze query to generate appropriate SQL
            query_lower = request.query.lower()
            
            if "highest salary" in query_lower or "max salary" in query_lower:
                mock_data = {
                    "sql": "SELECT * FROM employees ORDER BY Salary DESC LIMIT 1",
                    "confidence": 0.85,
                    "explanation": "This query finds the employee with the highest salary by sorting all employees by salary in descending order and taking the first result.",
                    "parameters": [],
                    "query_type": "SELECT_FILTERED",
                    "tables_involved": ["employees"],
                    "estimated_rows": 1
                }
            elif "all employees" in query_lower:
                mock_data = {
                    "sql": "SELECT * FROM employees",
                    "confidence": 0.90,
                    "explanation": "This query retrieves all records from the employees table.",
                    "parameters": [],
                    "query_type": "SELECT_ALL",
                    "tables_involved": ["employees"],
                    "estimated_rows": 1000
                }
            elif "engineering" in query_lower or "department" in query_lower:
                mock_data = {
                    "sql": "SELECT * FROM employees WHERE Team = 'Engineering'",
                    "confidence": 0.80,
                    "explanation": "This query filters employees by the Engineering department/team.",
                    "parameters": [],
                    "query_type": "SELECT_FILTERED",
                    "tables_involved": ["employees"],
                    "estimated_rows": 200
                }
            else:
                # Generic fallback
                mock_data = {
                    "sql": "SELECT * FROM employees LIMIT 10",
                    "confidence": 0.70,
                    "explanation": f"Mock response for query: {request.query}",
                    "parameters": [],
                    "query_type": "SELECT_FILTERED",
                    "tables_involved": ["employees"],
                    "estimated_rows": 10
                }
        
        # Convert to SQLQueryResponse format
        response_data = {
            "sql_query": mock_data.get("sql", "SELECT 1 as mock_result"),
            "explanation": mock_data.get("explanation", "Mock response generated for testing"),
            "confidence_score": mock_data.get("confidence", 0.80),
            "query_type": mock_data.get("query_type", "SELECT"),
            "estimated_complexity": "simple",
            "warnings": [],
            "suggestions": ["This is a mock response for testing"],
            "parameters": mock_data.get("parameters", []),
            "estimated_rows": mock_data.get("estimated_rows", 1),
            "execution_time_estimate": 0.05,
            "tables_used": mock_data.get("tables_involved", ["mock_table"])
        }
        
        return SQLQueryResponse(**response_data)