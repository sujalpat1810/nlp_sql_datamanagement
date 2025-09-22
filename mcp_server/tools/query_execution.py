"""
Query Execution Tools for MCP Server.

Provides tools for executing SQL queries with validation, parameterization,
and comprehensive logging for security and performance monitoring.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
import re
import sqlparse
from sqlparse import sql, tokens

from mcp.types import CallToolResult, TextContent

from database.connection import get_database_manager

logger = logging.getLogger(__name__)


class QueryExecutionTools:
    """Tools for SQL query execution and management."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.dangerous_keywords = {
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 
            'TRUNCATE', 'REPLACE', 'MERGE'
        }
    
    async def execute_select_query(
        self,
        sql_query: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        session_id: str = "default"
    ) -> CallToolResult:
        """
        Execute a SELECT query with optional parameters and result limiting.
        
        Args:
            sql_query: SQL SELECT query to execute
            parameters: Optional query parameters for parameterized queries
            limit: Optional limit on number of rows returned
            session_id: Session identifier for logging
            
        Returns:
            Query results with metadata
        """
        try:
            # Validate query is SELECT only
            if not self._is_select_query(sql_query):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: Only SELECT queries are allowed with this tool"
                    )]
                )
            
            # Apply limit if specified
            if limit:
                sql_query = self._apply_limit(sql_query, limit)
            
            # Execute query with timing
            start_time = time.time()
            results = await self.db_manager.execute_query(
                sql_query, 
                parameters or {}
            )
            execution_time = int((time.time() - start_time) * 1000)
            
            # Log the query execution
            await self._log_query_execution(
                session_id=session_id,
                sql_query=sql_query,
                query_type="SELECT",
                parameters=parameters,
                execution_time=execution_time,
                rows_affected=len(results),
                success=True
            )
            
            # Format results
            response_data = {
                "success": True,
                "query": sql_query,
                "execution_time_ms": execution_time,
                "row_count": len(results),
                "results": results[:100] if len(results) > 100 else results,  # Limit display
                "truncated": len(results) > 100,
                "parameters": parameters
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error executing SELECT query: {e}")
            
            # Log failed execution
            await self._log_query_execution(
                session_id=session_id,
                sql_query=sql_query,
                query_type="SELECT",
                parameters=parameters,
                execution_time=0,
                rows_affected=0,
                success=False,
                error_message=str(e)
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error executing SELECT query: {str(e)}"
                )]
            )
    
    async def execute_insert_query(
        self,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        batch_size: int = 1000,
        session_id: str = "default"
    ) -> CallToolResult:
        """
        Execute INSERT operations with batch processing.
        
        Args:
            table_name: Target table name
            data: Data to insert (single record or list of records)
            batch_size: Number of records per batch
            session_id: Session identifier for logging
            
        Returns:
            Insert operation results
        """
        try:
            # Normalize data to list format
            if isinstance(data, dict):
                records = [data]
            else:
                records = data
            
            if not records:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: No data provided for insertion"
                    )]
                )
            
            # Validate table exists
            table_info = await self.db_manager.get_table_info(table_name)
            if not table_info.get('exists', False):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: Table '{table_name}' does not exist"
                    )]
                )
            
            # Build INSERT query
            columns = list(records[0].keys())
            placeholders = ", ".join([f":{col}" for col in columns])
            insert_query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
            """
            
            # Execute batch insert
            start_time = time.time()
            total_inserted = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                for record in batch:
                    rows_affected = await self.db_manager.execute_non_query(
                        insert_query, record
                    )
                    total_inserted += rows_affected
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Log the operation
            await self._log_query_execution(
                session_id=session_id,
                sql_query=insert_query,
                query_type="INSERT",
                parameters={"batch_size": batch_size, "total_records": len(records)},
                execution_time=execution_time,
                rows_affected=total_inserted,
                success=True
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": f"Successfully inserted {total_inserted} rows into '{table_name}'",
                        "table_name": table_name,
                        "rows_inserted": total_inserted,
                        "execution_time_ms": execution_time,
                        "batch_size": batch_size
                    }, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error executing INSERT query: {e}")
            
            await self._log_query_execution(
                session_id=session_id,
                sql_query=f"INSERT INTO {table_name}",
                query_type="INSERT",
                parameters={"error": str(e)},
                execution_time=0,
                rows_affected=0,
                success=False,
                error_message=str(e)
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error executing INSERT query: {str(e)}"
                )]
            )
    
    async def execute_update_query(
        self,
        table_name: str,
        set_clause: Dict[str, Any],
        where_clause: Optional[Dict[str, Any]] = None,
        session_id: str = "default"
    ) -> CallToolResult:
        """
        Execute UPDATE operations with WHERE clause validation.
        
        Args:
            table_name: Target table name
            set_clause: Columns and values to update
            where_clause: Optional WHERE conditions
            session_id: Session identifier for logging
            
        Returns:
            Update operation results
        """
        try:
            # Validate inputs
            if not set_clause:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: SET clause cannot be empty"
                    )]
                )
            
            # Build UPDATE query
            set_parts = [f"{col} = :{col}" for col in set_clause.keys()]
            update_query = f"UPDATE {table_name} SET {', '.join(set_parts)}"
            
            query_params = dict(set_clause)
            
            if where_clause:
                where_parts = [f"{col} = :where_{col}" for col in where_clause.keys()]
                update_query += f" WHERE {' AND '.join(where_parts)}"
                
                # Add where parameters with prefix to avoid conflicts
                for col, val in where_clause.items():
                    query_params[f"where_{col}"] = val
            else:
                # Require confirmation for updates without WHERE clause
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Warning: UPDATE without WHERE clause would affect all rows. Please provide WHERE conditions."
                    )]
                )
            
            # Execute update
            start_time = time.time()
            rows_affected = await self.db_manager.execute_non_query(
                update_query, query_params
            )
            execution_time = int((time.time() - start_time) * 1000)
            
            # Log the operation
            await self._log_query_execution(
                session_id=session_id,
                sql_query=update_query,
                query_type="UPDATE",
                parameters=query_params,
                execution_time=execution_time,
                rows_affected=rows_affected,
                success=True
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": f"Successfully updated {rows_affected} rows in '{table_name}'",
                        "table_name": table_name,
                        "rows_affected": rows_affected,
                        "execution_time_ms": execution_time,
                        "set_clause": set_clause,
                        "where_clause": where_clause
                    }, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error executing UPDATE query: {e}")
            
            await self._log_query_execution(
                session_id=session_id,
                sql_query=f"UPDATE {table_name}",
                query_type="UPDATE",
                parameters={"error": str(e)},
                execution_time=0,
                rows_affected=0,
                success=False,
                error_message=str(e)
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error executing UPDATE query: {str(e)}"
                )]
            )
    
    async def execute_delete_query(
        self,
        table_name: str,
        where_clause: Dict[str, Any],
        session_id: str = "default"
    ) -> CallToolResult:
        """
        Execute DELETE operations with mandatory WHERE clause.
        
        Args:
            table_name: Target table name
            where_clause: WHERE conditions (required for safety)
            session_id: Session identifier for logging
            
        Returns:
            Delete operation results
        """
        try:
            # Require WHERE clause for safety
            if not where_clause:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: DELETE operations require WHERE clause for safety"
                    )]
                )
            
            # Build DELETE query
            where_parts = [f"{col} = :{col}" for col in where_clause.keys()]
            delete_query = f"DELETE FROM {table_name} WHERE {' AND '.join(where_parts)}"
            
            # Execute delete
            start_time = time.time()
            rows_affected = await self.db_manager.execute_non_query(
                delete_query, where_clause
            )
            execution_time = int((time.time() - start_time) * 1000)
            
            # Log the operation
            await self._log_query_execution(
                session_id=session_id,
                sql_query=delete_query,
                query_type="DELETE",
                parameters=where_clause,
                execution_time=execution_time,
                rows_affected=rows_affected,
                success=True
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": f"Successfully deleted {rows_affected} rows from '{table_name}'",
                        "table_name": table_name,
                        "rows_affected": rows_affected,
                        "execution_time_ms": execution_time,
                        "where_clause": where_clause
                    }, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error executing DELETE query: {e}")
            
            await self._log_query_execution(
                session_id=session_id,
                sql_query=f"DELETE FROM {table_name}",
                query_type="DELETE",
                parameters={"error": str(e)},
                execution_time=0,
                rows_affected=0,
                success=False,
                error_message=str(e)
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error executing DELETE query: {str(e)}"
                )]
            )
    
    async def execute_custom_sql(
        self,
        sql_query: str,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
        admin_mode: bool = False
    ) -> CallToolResult:
        """
        Execute custom SQL with safety validations.
        
        Args:
            sql_query: SQL query to execute
            parameters: Optional query parameters
            session_id: Session identifier for logging
            admin_mode: Whether to allow dangerous operations
            
        Returns:
            Query execution results
        """
        try:
            # Parse and validate query
            parsed = sqlparse.parse(sql_query)[0]
            query_type = self._get_query_type(parsed)
            
            # Safety check for dangerous operations
            if not admin_mode and query_type in self.dangerous_keywords:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: {query_type} operations require admin mode for safety"
                    )]
                )
            
            # Execute query
            start_time = time.time()
            
            if query_type == "SELECT":
                results = await self.db_manager.execute_query(sql_query, parameters or {})
                execution_time = int((time.time() - start_time) * 1000)
                rows_affected = len(results)
                
                response_data = {
                    "success": True,
                    "query_type": query_type,
                    "execution_time_ms": execution_time,
                    "row_count": rows_affected,
                    "results": results[:100] if len(results) > 100 else results,
                    "truncated": len(results) > 100
                }
            else:
                rows_affected = await self.db_manager.execute_non_query(sql_query, parameters or {})
                execution_time = int((time.time() - start_time) * 1000)
                
                response_data = {
                    "success": True,
                    "query_type": query_type,
                    "execution_time_ms": execution_time,
                    "rows_affected": rows_affected
                }
            
            # Log the operation
            await self._log_query_execution(
                session_id=session_id,
                sql_query=sql_query,
                query_type=query_type,
                parameters=parameters,
                execution_time=execution_time,
                rows_affected=rows_affected,
                success=True
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error executing custom SQL: {e}")
            
            await self._log_query_execution(
                session_id=session_id,
                sql_query=sql_query,
                query_type="UNKNOWN",
                parameters=parameters,
                execution_time=0,
                rows_affected=0,
                success=False,
                error_message=str(e)
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error executing custom SQL: {str(e)}"
                )]
            )
    
    async def validate_sql_syntax(self, sql_query: str) -> CallToolResult:
        """
        Validate SQL syntax and provide analysis.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Validation results and analysis
        """
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql_query)
            
            if not parsed:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps({
                            "valid": False,
                            "error": "Unable to parse SQL query"
                        }, indent=2)
                    )]
                )
            
            statement = parsed[0]
            query_type = self._get_query_type(statement)
            
            # Extract components
            analysis = {
                "valid": True,
                "query_type": query_type,
                "formatted_query": sqlparse.format(
                    sql_query, 
                    reindent=True, 
                    keyword_case='upper'
                ),
                "tables": self._extract_tables(statement),
                "columns": self._extract_columns(statement),
                "has_where_clause": self._has_where_clause(statement),
                "has_joins": self._has_joins(statement),
                "complexity_score": self._calculate_complexity(statement)
            }
            
            # Safety warnings
            warnings = []
            if query_type in self.dangerous_keywords:
                warnings.append(f"{query_type} operation - use with caution")
            
            if query_type in ['UPDATE', 'DELETE'] and not analysis['has_where_clause']:
                warnings.append("Missing WHERE clause - will affect all rows")
            
            analysis["warnings"] = warnings
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(analysis, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error validating SQL: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "valid": False,
                        "error": str(e)
                    }, indent=2)
                )]
            )
    
    def _is_select_query(self, sql_query: str) -> bool:
        """Check if query is a SELECT statement."""
        try:
            parsed = sqlparse.parse(sql_query)[0]
            return self._get_query_type(parsed) == "SELECT"
        except:
            return False
    
    def _get_query_type(self, statement) -> str:
        """Extract query type from parsed SQL."""
        for token in statement.tokens:
            if token.ttype is tokens.Keyword.DML:
                return token.value.upper()
        return "UNKNOWN"
    
    def _apply_limit(self, sql_query: str, limit: int) -> str:
        """Add LIMIT clause to query if not present."""
        if re.search(r'\bLIMIT\b', sql_query, re.IGNORECASE):
            return sql_query
        return f"{sql_query.rstrip(';')} LIMIT {limit}"
    
    def _extract_tables(self, statement) -> List[str]:
        """Extract table names from SQL statement."""
        tables = []
        # Simplified table extraction - could be enhanced
        for token in statement.flatten():
            if token.ttype is None and isinstance(token.value, str):
                if any(keyword in str(statement).upper() for keyword in ['FROM', 'JOIN', 'UPDATE', 'INTO']):
                    tables.append(token.value)
        return list(set(tables))
    
    def _extract_columns(self, statement) -> List[str]:
        """Extract column names from SQL statement."""
        # Simplified column extraction - could be enhanced
        return []
    
    def _has_where_clause(self, statement) -> bool:
        """Check if statement has WHERE clause."""
        return 'WHERE' in str(statement).upper()
    
    def _has_joins(self, statement) -> bool:
        """Check if statement has JOIN clauses."""
        sql_text = str(statement).upper()
        return any(join_type in sql_text for join_type in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN'])
    
    def _calculate_complexity(self, statement) -> int:
        """Calculate query complexity score."""
        sql_text = str(statement).upper()
        complexity = 0
        
        # Base complexity
        complexity += 1
        
        # Add for joins
        complexity += sql_text.count('JOIN')
        
        # Add for subqueries
        complexity += sql_text.count('SELECT') - 1
        
        # Add for aggregations
        aggregations = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY']
        complexity += sum(1 for agg in aggregations if agg in sql_text)
        
        return complexity
    
    async def _log_query_execution(
        self,
        session_id: str,
        sql_query: str,
        query_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        execution_time: int = 0,
        rows_affected: int = 0,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log query execution to database."""
        try:
            log_query = """
            INSERT INTO query_logs (
                session_id, sql_query, query_type, parameters,
                execution_time_ms, rows_affected, success, error_message, created_at
            )
            VALUES (
                :session_id, :sql_query, :query_type, :parameters,
                :execution_time_ms, :rows_affected, :success, :error_message, datetime('now')
            )
            """
            
            await self.db_manager.execute_non_query(log_query, {
                'session_id': session_id,
                'sql_query': sql_query[:1000],  # Truncate long queries
                'query_type': query_type,
                'parameters': json.dumps(parameters) if parameters else None,
                'execution_time_ms': execution_time,
                'rows_affected': rows_affected,
                'success': success,
                'error_message': error_message
            })
        except Exception as e:
            logger.error(f"Failed to log query execution: {e}")