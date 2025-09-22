"""
Main NLP to SQL MCP Client.

This module provides the primary client interface that orchestrates
communication between Claude API and the MCP server for natural language
to SQL query conversion and execution.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import httpx
from pydantic import BaseModel, ValidationError

from .claude_interface import ClaudeInterface, NLQueryRequest, SQLQueryResponse
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class QueryResult(BaseModel):
    """Result of a complete query processing pipeline."""
    session_id: str
    natural_language_query: str
    generated_sql: str
    sql_explanation: str
    execution_results: Optional[Dict[str, Any]] = None
    execution_success: bool = True
    execution_error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    confidence_score: float
    query_type: str
    warnings: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None


class NLPSQLClient:
    """Main client for NLP to SQL processing."""
    
    def __init__(
        self,
        claude_api_key: Optional[str] = None,
        mcp_server_url: str = "http://localhost:8000",
        session_storage_dir: Optional[str] = None,
        claude_model: str = "claude-3-sonnet-20240229",
        mock_mode: bool = False
    ):
        """
        Initialize the NLP SQL Client.
        
        Args:
            claude_api_key: Anthropic Claude API key (optional in mock mode)
            mcp_server_url: URL of the MCP server
            session_storage_dir: Directory for session persistence
            claude_model: Claude model to use
            mock_mode: Whether to use mock responses for testing
        """
        self.mcp_server_url = mcp_server_url.rstrip('/')
        self.mock_mode = mock_mode or not claude_api_key or claude_api_key in ["test", "mock", "test_api_key"]
        
        # Initialize components
        self.claude = ClaudeInterface(claude_api_key, claude_model, mock_mode=self.mock_mode)
        self.session_manager = SessionManager(session_storage_dir)
        
        if self.mock_mode:
            logger.info("NLP SQL Client initialized in mock mode")
        
        # HTTP client for MCP server communication
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Cache for database schema
        self._schema_cache: Dict[str, Any] = {}
        self._schema_cache_timestamp: Optional[float] = None
        self._schema_cache_ttl = 300  # 5 minutes
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close client connections."""
        await self.http_client.aclose()
    
    async def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new user session.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        session_id = await self.session_manager.create_session(user_id)
        
        # Load database schema for new session
        try:
            schema_info = await self._get_database_schema()
            await self.session_manager.update_session_schema(session_id, schema_info)
        except Exception as e:
            logger.warning(f"Failed to load database schema for new session: {e}")
        
        return session_id
    
    async def process_natural_language_query(
        self,
        session_id: str,
        natural_language_query: str,
        execute: bool = True,
        explain_only: bool = False
    ) -> QueryResult:
        """
        Process a natural language query through the complete pipeline.
        
        Args:
            session_id: Session identifier
            natural_language_query: Natural language query from user
            execute: Whether to execute the generated SQL
            explain_only: Only generate SQL explanation without execution
            
        Returns:
            Complete query processing result
        """
        start_time = time.time()
        
        try:
            # Get session context
            session = await self.session_manager.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Prepare request for Claude
            request = await self._prepare_claude_request(session_id, natural_language_query)
            
            # Generate SQL using Claude
            sql_response = await self.claude.process_natural_language_query(request)
            
            # Initialize result
            result = QueryResult(
                session_id=session_id,
                natural_language_query=natural_language_query,
                generated_sql=sql_response.sql_query,
                sql_explanation=sql_response.explanation,
                confidence_score=sql_response.confidence_score,
                query_type=sql_response.query_type,
                warnings=sql_response.warnings,
                suggestions=sql_response.suggestions
            )
            
            execution_time_ms = None
            
            # Execute SQL if requested and not explain-only mode
            if execute and not explain_only and sql_response.sql_query != "-- Error generating query":
                try:
                    execution_start = time.time()
                    execution_result = await self._execute_sql_query(sql_response.sql_query)
                    execution_time_ms = (time.time() - execution_start) * 1000
                    
                    result.execution_results = execution_result
                    result.execution_success = True
                    result.execution_time_ms = execution_time_ms
                    
                except Exception as e:
                    result.execution_success = False
                    result.execution_error = str(e)
                    logger.error(f"SQL execution failed: {e}")
            
            # Add to session history
            await self.session_manager.add_query_to_history(
                session_id=session_id,
                natural_language_query=natural_language_query,
                generated_sql=sql_response.sql_query,
                execution_result=result.execution_results,
                success=result.execution_success,
                error_message=result.execution_error,
                execution_time_ms=execution_time_ms
            )
            
            total_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Processed query in {total_time_ms:.2f}ms (confidence: {sql_response.confidence_score})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            
            # Create error result
            error_result = QueryResult(
                session_id=session_id,
                natural_language_query=natural_language_query,
                generated_sql="-- Error processing query",
                sql_explanation=f"Failed to process query: {str(e)}",
                execution_success=False,
                execution_error=str(e),
                confidence_score=0.0,
                query_type="ERROR",
                warnings=[f"Processing failed: {str(e)}"]
            )
            
            # Still add to history for debugging
            await self.session_manager.add_query_to_history(
                session_id=session_id,
                natural_language_query=natural_language_query,
                generated_sql="-- Error processing query",
                execution_result=None,
                success=False,
                error_message=str(e)
            )
            
            return error_result
    
    async def explain_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Get detailed explanation of a SQL query.
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Detailed explanation and analysis
        """
        return await self.claude.explain_sql_query(sql_query)
    
    async def get_optimization_suggestions(
        self, 
        sql_query: str, 
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions for a SQL query.
        
        Args:
            sql_query: SQL query to optimize
            session_id: Optional session ID for schema context
            
        Returns:
            List of optimization suggestions
        """
        schema_info = None
        if session_id:
            session = await self.session_manager.get_session(session_id)
            if session:
                schema_info = session.database_schema
        
        return await self.claude.suggest_optimizations(sql_query, schema_info)
    
    async def validate_sql_safety(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query for safety concerns.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Safety validation results
        """
        return await self.claude.validate_query_safety(sql_query)
    
    async def get_session_history(
        self, 
        session_id: str, 
        limit: int = 20,
        include_failed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get query history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum entries to return
            include_failed: Whether to include failed queries
            
        Returns:
            List of query history entries
        """
        history = await self.session_manager.get_query_history(session_id, limit, include_failed)
        
        # Convert to dictionaries for JSON serialization
        return [
            {
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "natural_language_query": entry.natural_language_query,
                "generated_sql": entry.generated_sql,
                "execution_result": entry.execution_result,
                "success": entry.success,
                "error_message": entry.error_message,
                "execution_time_ms": entry.execution_time_ms
            }
            for entry in history
        ]
    
    async def get_session_statistics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session statistics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session statistics
        """
        return await self.session_manager.get_session_statistics(session_id)
    
    async def get_database_tables(self) -> List[Dict[str, Any]]:
        """
        Get list of available database tables.
        
        Returns:
            List of table information
        """
        try:
            response = await self.http_client.post(
                f"{self.mcp_server_url}/tools/get_table_list",
                json={}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get database tables: {e}")
            return []
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table schema information
        """
        try:
            response = await self.http_client.post(
                f"{self.mcp_server_url}/tools/get_table_schema",
                json={"table_name": table_name}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get table schema for {table_name}: {e}")
            return {}
    
    async def analyze_data_quality(self, table_name: str) -> Dict[str, Any]:
        """
        Perform data quality analysis on a table.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Data quality analysis results
        """
        try:
            response = await self.http_client.post(
                f"{self.mcp_server_url}/tools/analyze_data_quality",
                json={"table_name": table_name}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to analyze data quality for {table_name}: {e}")
            return {}
    
    async def load_data_from_file(
        self, 
        file_path: str, 
        table_name: str,
        file_format: str = "csv"
    ) -> Dict[str, Any]:
        """
        Load data from a file into the database.
        
        Args:
            file_path: Path to the data file
            table_name: Target table name
            file_format: File format ("csv" or "json")
            
        Returns:
            Data loading results
        """
        try:
            tool_name = f"load_{file_format}_data"
            response = await self.http_client.post(
                f"{self.mcp_server_url}/tools/{tool_name}",
                json={
                    "file_path": file_path,
                    "table_name": table_name
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            return {"error": str(e)}
    
    # Private methods
    
    async def _prepare_claude_request(
        self, 
        session_id: str, 
        natural_language_query: str
    ) -> NLQueryRequest:
        """Prepare request for Claude API."""
        # Get session context
        session = await self.session_manager.get_session(session_id)
        
        # Get recent query context
        recent_queries = await self.session_manager.get_recent_successful_queries(session_id, 3)
        
        # Prepare request
        request = NLQueryRequest(
            query=natural_language_query,
            schema_info=session.database_schema if session else None,
            previous_queries=recent_queries,
            context={
                "session_id": session_id,
                "active_tables": list(session.active_tables) if session else [],
                "preferences": session.preferences if session else {}
            }
        )
        
        return request
    
    async def _get_database_schema(self) -> Dict[str, Any]:
        """Get complete database schema with caching."""
        current_time = time.time()
        
        # Check cache
        if (self._schema_cache and self._schema_cache_timestamp and 
            current_time - self._schema_cache_timestamp < self._schema_cache_ttl):
            return self._schema_cache
        
        try:
            # Get list of tables
            tables_response = await self.http_client.post(
                f"{self.mcp_server_url}/tools/get_table_list",
                json={}
            )
            tables_response.raise_for_status()
            tables_data = tables_response.json()
            
            schema_info = {
                "tables": {},
                "timestamp": current_time
            }
            
            # Get schema for each table
            if isinstance(tables_data, list):
                for table_info in tables_data:
                    table_name = table_info.get("name") if isinstance(table_info, dict) else str(table_info)
                    
                    try:
                        table_schema = await self.get_table_schema(table_name)
                        schema_info["tables"][table_name] = table_schema
                    except Exception as e:
                        logger.warning(f"Failed to get schema for table {table_name}: {e}")
            
            # Update cache
            self._schema_cache = schema_info
            self._schema_cache_timestamp = current_time
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Failed to get database schema: {e}")
            return {}
    
    async def _execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query via MCP server."""
        # Determine query type to call appropriate endpoint
        query_upper = sql_query.upper().strip()
        
        if query_upper.startswith('SELECT'):
            tool_name = "execute_select_query"
        elif query_upper.startswith('INSERT'):
            tool_name = "execute_insert_query"
        elif query_upper.startswith('UPDATE'):
            tool_name = "execute_update_query"
        elif query_upper.startswith('DELETE'):
            tool_name = "execute_delete_query"
        else:
            tool_name = "execute_custom_sql"
        
        response = await self.http_client.post(
            f"{self.mcp_server_url}/tools/{tool_name}",
            json={"query": sql_query}
        )
        response.raise_for_status()
        
        return response.json()
    
    async def cleanup_expired_sessions(self, expiry_hours: int = 24) -> int:
        """Clean up expired sessions."""
        return await self.session_manager.cleanup_expired_sessions(expiry_hours)


# Convenience functions for common operations

async def create_client(
    claude_api_key: Optional[str] = None,
    mcp_server_url: str = "http://localhost:8000", 
    session_storage_dir: Optional[str] = None
) -> NLPSQLClient:
    """
    Create an NLP SQL Client with environment variable defaults.
    
    Args:
        claude_api_key: Claude API key (defaults to CLAUDE_API_KEY env var)
        mcp_server_url: MCP server URL
        session_storage_dir: Session storage directory
        
    Returns:
        Configured NLP SQL Client
    """
    if not claude_api_key:
        claude_api_key = os.getenv("CLAUDE_API_KEY")
        if not claude_api_key:
            raise ValueError("Claude API key must be provided or set in CLAUDE_API_KEY environment variable")
    
    return NLPSQLClient(
        claude_api_key=claude_api_key,
        mcp_server_url=mcp_server_url,
        session_storage_dir=session_storage_dir
    )


async def quick_query(
    query: str, 
    claude_api_key: Optional[str] = None,
    execute: bool = True
) -> QueryResult:
    """
    Quick one-off natural language query processing.
    
    Args:
        query: Natural language query
        claude_api_key: Claude API key
        execute: Whether to execute the generated SQL
        
    Returns:
        Query processing result
    """
    async with await create_client(claude_api_key) as client:
        session_id = await client.create_session()
        return await client.process_natural_language_query(
            session_id=session_id,
            natural_language_query=query,
            execute=execute
        )