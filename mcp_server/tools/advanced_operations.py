"""
Advanced Operations Tools for MCP Server.

Provides advanced database operations including joins, aggregations, 
view management, indexing, and complex query operations.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import sqlparse
from sqlparse import sql, tokens
from mcp.types import CallToolResult, TextContent

from database.connection import get_database_manager

logger = logging.getLogger(__name__)


class AdvancedOperationsTools:
    """Tools for advanced database operations and query management."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    async def create_join_query(
        self,
        primary_table: str,
        join_table: str,
        join_condition: str,
        join_type: str = "INNER",
        select_columns: Optional[List[str]] = None,
        where_conditions: Optional[str] = None,
        limit: Optional[int] = None
    ) -> CallToolResult:
        """
        Create and execute a JOIN query between two tables.
        
        Args:
            primary_table: Main table for the join
            join_table: Table to join with
            join_condition: JOIN condition (e.g., "primary_table.id = join_table.primary_id")
            join_type: Type of join (INNER, LEFT, RIGHT, FULL OUTER)
            select_columns: Columns to select (None for all)
            where_conditions: WHERE clause conditions
            limit: Number of rows to return
            
        Returns:
            Join query results
        """
        try:
            # Validate join type
            valid_join_types = ['INNER', 'LEFT', 'RIGHT', 'FULL OUTER', 'CROSS']
            if join_type.upper() not in valid_join_types:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Invalid join type '{join_type}'. Valid types: {', '.join(valid_join_types)}"
                    )]
                )
            
            # Validate tables exist
            primary_exists = await self.db_manager.get_table_info(primary_table)
            join_exists = await self.db_manager.get_table_info(join_table)
            
            if not primary_exists.get('exists', False):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Primary table '{primary_table}' does not exist"
                    )]
                )
            
            if not join_exists.get('exists', False):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Join table '{join_table}' does not exist"
                    )]
                )
            
            # Build SELECT clause
            if select_columns:
                select_clause = ", ".join(select_columns)
            else:
                select_clause = f"{primary_table}.*, {join_table}.*"
            
            # Build the JOIN query
            query_parts = [
                f"SELECT {select_clause}",
                f"FROM {primary_table}",
                f"{join_type} JOIN {join_table}",
                f"ON {join_condition}"
            ]
            
            if where_conditions:
                query_parts.append(f"WHERE {where_conditions}")
            
            if limit:
                query_parts.append(f"LIMIT {limit}")
            
            query = " ".join(query_parts)
            
            # Log the query
            logger.info(f"Executing JOIN query: {query}")
            
            # Validate query syntax
            try:
                parsed = sqlparse.parse(query)[0]
                if not parsed.tokens:
                    raise ValueError("Empty query")
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Invalid query syntax: {str(e)}"
                    )]
                )
            
            # Execute query
            results = await self.db_manager.execute_query(query)
            
            response = {
                "query": query,
                "primary_table": primary_table,
                "join_table": join_table,
                "join_type": join_type,
                "results_count": len(results) if results else 0,
                "results": results[:100] if results else []  # Limit output size
            }
            
            if results and len(results) > 100:
                response["note"] = f"Showing first 100 results out of {len(results)} total"
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error creating join query: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error creating join query: {str(e)}"
                )]
            )
    
    async def create_aggregation_query(
        self,
        table_name: str,
        group_by_columns: List[str],
        aggregations: Dict[str, str],
        having_conditions: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> CallToolResult:
        """
        Create and execute an aggregation query with GROUP BY.
        
        Args:
            table_name: Table to aggregate
            group_by_columns: Columns to group by
            aggregations: Dict of {column: function} for aggregations (e.g., {"price": "AVG", "quantity": "SUM"})
            having_conditions: HAVING clause conditions
            order_by: ORDER BY clause
            limit: Number of rows to return
            
        Returns:
            Aggregation query results
        """
        try:
            # Validate table exists
            table_info = await self.db_manager.get_table_info(table_name)
            if not table_info.get('exists', False):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Table '{table_name}' does not exist"
                    )]
                )
            
            # Validate aggregation functions
            valid_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COUNT_DISTINCT']
            
            select_parts = []
            
            # Add GROUP BY columns to SELECT
            for col in group_by_columns:
                select_parts.append(col)
            
            # Add aggregations to SELECT
            for column, func in aggregations.items():
                func_upper = func.upper()
                if func_upper not in valid_functions:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Invalid aggregation function '{func}'. Valid functions: {', '.join(valid_functions)}"
                        )]
                    )
                
                if func_upper == 'COUNT_DISTINCT':
                    select_parts.append(f"COUNT(DISTINCT {column}) as {column}_{func_upper.lower()}")
                else:
                    select_parts.append(f"{func_upper}({column}) as {column}_{func_upper.lower()}")
            
            # Build the query
            query_parts = [
                f"SELECT {', '.join(select_parts)}",
                f"FROM {table_name}",
                f"GROUP BY {', '.join(group_by_columns)}"
            ]
            
            if having_conditions:
                query_parts.append(f"HAVING {having_conditions}")
            
            if order_by:
                query_parts.append(f"ORDER BY {order_by}")
            
            if limit:
                query_parts.append(f"LIMIT {limit}")
            
            query = " ".join(query_parts)
            
            # Log the query
            logger.info(f"Executing aggregation query: {query}")
            
            # Execute query
            results = await self.db_manager.execute_query(query)
            
            response = {
                "query": query,
                "table_name": table_name,
                "group_by_columns": group_by_columns,
                "aggregations": aggregations,
                "results_count": len(results) if results else 0,
                "results": results
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error creating aggregation query: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error creating aggregation query: {str(e)}"
                )]
            )
    
    async def create_database_view(
        self,
        view_name: str,
        query: str,
        replace_if_exists: bool = False
    ) -> CallToolResult:
        """
        Create a database view from a SELECT query.
        
        Args:
            view_name: Name for the new view
            query: SELECT query to create the view from
            replace_if_exists: Whether to replace existing view
            
        Returns:
            View creation results
        """
        try:
            # Validate view name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', view_name):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Invalid view name '{view_name}'. Use only letters, numbers, and underscores."
                    )]
                )
            
            # Validate query is a SELECT statement
            parsed = sqlparse.parse(query.strip())[0]
            if not parsed.tokens or parsed.get_type() != 'SELECT':
                first_token = parsed.tokens[0] if parsed.tokens else None
                if not first_token or first_token.ttype is tokens.Keyword and first_token.value.upper() != 'SELECT':
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Query must be a SELECT statement"
                        )]
                    )
            
            # Check if view already exists
            view_exists_query = f"""
            SELECT name FROM sqlite_master 
            WHERE type='view' AND name='{view_name}'
            """
            
            existing_view = await self.db_manager.execute_query(view_exists_query)
            
            if existing_view and not replace_if_exists:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"View '{view_name}' already exists. Use replace_if_exists=True to replace it."
                    )]
                )
            
            # Drop existing view if replacing
            if existing_view and replace_if_exists:
                drop_query = f"DROP VIEW IF EXISTS {view_name}"
                await self.db_manager.execute_query(drop_query)
                logger.info(f"Dropped existing view: {view_name}")
            
            # Create the view
            create_view_query = f"CREATE VIEW {view_name} AS {query}"
            
            await self.db_manager.execute_query(create_view_query)
            
            # Verify view was created
            verify_query = f"SELECT * FROM {view_name} LIMIT 5"
            sample_data = await self.db_manager.execute_query(verify_query)
            
            response = {
                "view_name": view_name,
                "status": "created" if not existing_view else "replaced",
                "query": query,
                "create_view_sql": create_view_query,
                "sample_data": sample_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Created view '{view_name}' successfully")
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error creating view: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error creating view: {str(e)}"
                )]
            )
    
    async def create_database_index(
        self,
        table_name: str,
        columns: List[str],
        index_name: Optional[str] = None,
        unique: bool = False,
        partial_condition: Optional[str] = None
    ) -> CallToolResult:
        """
        Create a database index on specified columns.
        
        Args:
            table_name: Table to create index on
            columns: Columns to include in the index
            index_name: Name for the index (auto-generated if not provided)
            unique: Whether to create a unique index
            partial_condition: WHERE condition for partial index
            
        Returns:
            Index creation results
        """
        try:
            # Validate table exists
            table_info = await self.db_manager.get_table_info(table_name)
            if not table_info.get('exists', False):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Table '{table_name}' does not exist"
                    )]
                )
            
            # Generate index name if not provided
            if not index_name:
                index_name = f"idx_{table_name}_{'_'.join(columns)}"
            
            # Validate index name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', index_name):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Invalid index name '{index_name}'. Use only letters, numbers, and underscores."
                    )]
                )
            
            # Check if index already exists
            existing_index_query = f"""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name='{index_name}'
            """
            
            existing_index = await self.db_manager.execute_query(existing_index_query)
            
            if existing_index:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Index '{index_name}' already exists"
                    )]
                )
            
            # Build CREATE INDEX statement
            create_parts = ["CREATE"]
            
            if unique:
                create_parts.append("UNIQUE")
            
            create_parts.extend([
                "INDEX",
                index_name,
                "ON",
                f"{table_name} ({', '.join(columns)})"
            ])
            
            if partial_condition:
                create_parts.extend(["WHERE", partial_condition])
            
            create_index_query = " ".join(create_parts)
            
            # Execute the CREATE INDEX statement
            await self.db_manager.execute_query(create_index_query)
            
            # Get index information
            index_info_query = f"""
            SELECT name, sql FROM sqlite_master 
            WHERE type='index' AND name='{index_name}'
            """
            
            index_info = await self.db_manager.execute_query(index_info_query)
            
            response = {
                "index_name": index_name,
                "table_name": table_name,
                "columns": columns,
                "unique": unique,
                "partial_condition": partial_condition,
                "create_sql": create_index_query,
                "status": "created",
                "index_info": index_info[0] if index_info else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Created index '{index_name}' on table '{table_name}'")
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error creating index: {str(e)}"
                )]
            )
    
    async def create_subquery(
        self,
        outer_table: str,
        subquery: str,
        subquery_alias: str,
        join_condition: Optional[str] = None,
        select_columns: Optional[List[str]] = None,
        where_conditions: Optional[str] = None
    ) -> CallToolResult:
        """
        Create a query with a subquery (either in FROM clause or as JOIN).
        
        Args:
            outer_table: Main table for the outer query
            subquery: The subquery SQL
            subquery_alias: Alias for the subquery
            join_condition: Condition to join with subquery (if None, subquery goes in FROM)
            select_columns: Columns to select
            where_conditions: WHERE clause conditions
            
        Returns:
            Subquery results
        """
        try:
            # Validate subquery syntax
            try:
                parsed = sqlparse.parse(subquery.strip())[0]
                if not parsed.tokens:
                    raise ValueError("Empty subquery")
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Invalid subquery syntax: {str(e)}"
                    )]
                )
            
            # Build SELECT clause
            if select_columns:
                select_clause = ", ".join(select_columns)
            else:
                select_clause = "*"
            
            # Build the main query
            if join_condition:
                # Subquery as JOIN
                query_parts = [
                    f"SELECT {select_clause}",
                    f"FROM {outer_table}",
                    f"JOIN ({subquery}) AS {subquery_alias}",
                    f"ON {join_condition}"
                ]
            else:
                # Subquery in FROM clause
                query_parts = [
                    f"SELECT {select_clause}",
                    f"FROM ({subquery}) AS {subquery_alias}"
                ]
            
            if where_conditions:
                query_parts.append(f"WHERE {where_conditions}")
            
            final_query = " ".join(query_parts)
            
            # Log the query
            logger.info(f"Executing subquery: {final_query}")
            
            # Execute query
            results = await self.db_manager.execute_query(final_query)
            
            response = {
                "outer_table": outer_table,
                "subquery": subquery,
                "subquery_alias": subquery_alias,
                "final_query": final_query,
                "results_count": len(results) if results else 0,
                "results": results[:100] if results else []
            }
            
            if results and len(results) > 100:
                response["note"] = f"Showing first 100 results out of {len(results)} total"
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error executing subquery: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error executing subquery: {str(e)}"
                )]
            )
    
    async def execute_union_query(
        self,
        queries: List[str],
        union_type: str = "UNION",
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> CallToolResult:
        """
        Execute UNION query combining multiple SELECT statements.
        
        Args:
            queries: List of SELECT queries to union
            union_type: Type of union (UNION, UNION ALL)
            order_by: ORDER BY clause for the combined result
            limit: Number of rows to return
            
        Returns:
            Union query results
        """
        try:
            if len(queries) < 2:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="At least 2 queries are required for UNION"
                    )]
                )
            
            # Validate union type
            if union_type.upper() not in ['UNION', 'UNION ALL']:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Invalid union type. Use 'UNION' or 'UNION ALL'"
                    )]
                )
            
            # Validate each query is a SELECT statement
            for i, query in enumerate(queries):
                try:
                    parsed = sqlparse.parse(query.strip())[0]
                    if not parsed.tokens:
                        raise ValueError(f"Empty query at index {i}")
                    
                    # Check if it's a SELECT statement
                    first_token = parsed.tokens[0]
                    if (first_token.ttype is not tokens.Keyword or 
                        first_token.value.upper() != 'SELECT'):
                        raise ValueError(f"Query {i} is not a SELECT statement")
                        
                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Invalid query at index {i}: {str(e)}"
                        )]
                    )
            
            # Build the UNION query
            union_query = f" {union_type} ".join(f"({query})" for query in queries)
            
            if order_by:
                union_query += f" ORDER BY {order_by}"
            
            if limit:
                union_query += f" LIMIT {limit}"
            
            # Log the query
            logger.info(f"Executing UNION query: {union_query}")
            
            # Execute query
            results = await self.db_manager.execute_query(union_query)
            
            response = {
                "union_type": union_type,
                "queries_count": len(queries),
                "individual_queries": queries,
                "union_query": union_query,
                "results_count": len(results) if results else 0,
                "results": results[:100] if results else []
            }
            
            if results and len(results) > 100:
                response["note"] = f"Showing first 100 results out of {len(results)} total"
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error executing union query: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error executing union query: {str(e)}"
                )]
            )
    
    async def create_common_table_expression(
        self,
        cte_definitions: List[Dict[str, str]],
        main_query: str
    ) -> CallToolResult:
        """
        Execute a query with Common Table Expressions (CTEs).
        
        Args:
            cte_definitions: List of {"name": "cte_name", "query": "SELECT ..."} 
            main_query: Main query that uses the CTEs
            
        Returns:
            CTE query results
        """
        try:
            if not cte_definitions:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="At least one CTE definition is required"
                    )]
                )
            
            # Validate CTE definitions
            cte_parts = []
            for i, cte in enumerate(cte_definitions):
                if 'name' not in cte or 'query' not in cte:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"CTE {i} must have 'name' and 'query' fields"
                        )]
                    )
                
                # Validate CTE name
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', cte['name']):
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Invalid CTE name '{cte['name']}'. Use only letters, numbers, and underscores."
                        )]
                    )
                
                # Validate CTE query
                try:
                    parsed = sqlparse.parse(cte['query'].strip())[0]
                    if not parsed.tokens:
                        raise ValueError(f"Empty CTE query for '{cte['name']}'")
                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Invalid CTE query for '{cte['name']}': {str(e)}"
                        )]
                    )
                
                cte_parts.append(f"{cte['name']} AS ({cte['query']})")
            
            # Build the full CTE query
            with_clause = "WITH " + ", ".join(cte_parts)
            full_query = f"{with_clause} {main_query}"
            
            # Log the query
            logger.info(f"Executing CTE query: {full_query}")
            
            # Execute query
            results = await self.db_manager.execute_query(full_query)
            
            response = {
                "cte_count": len(cte_definitions),
                "cte_definitions": cte_definitions,
                "main_query": main_query,
                "full_query": full_query,
                "results_count": len(results) if results else 0,
                "results": results[:100] if results else []
            }
            
            if results and len(results) > 100:
                response["note"] = f"Showing first 100 results out of {len(results)} total"
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error executing CTE query: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error executing CTE query: {str(e)}"
                )]
            )
    
    async def manage_database_views(
        self,
        action: str,
        view_name: Optional[str] = None
    ) -> CallToolResult:
        """
        Manage database views (list, drop, or get info).
        
        Args:
            action: Action to perform ('list', 'drop', 'info')
            view_name: Name of view (required for 'drop' and 'info' actions)
            
        Returns:
            View management results
        """
        try:
            if action == 'list':
                # List all views
                query = """
                SELECT name, sql 
                FROM sqlite_master 
                WHERE type='view'
                ORDER BY name
                """
                
                views = await self.db_manager.execute_query(query)
                
                response = {
                    "action": "list",
                    "views_count": len(views) if views else 0,
                    "views": views
                }
                
            elif action == 'drop':
                if not view_name:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="view_name is required for 'drop' action"
                        )]
                    )
                
                # Check if view exists
                check_query = f"""
                SELECT name FROM sqlite_master 
                WHERE type='view' AND name='{view_name}'
                """
                
                existing = await self.db_manager.execute_query(check_query)
                
                if not existing:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"View '{view_name}' does not exist"
                        )]
                    )
                
                # Drop the view
                drop_query = f"DROP VIEW {view_name}"
                await self.db_manager.execute_query(drop_query)
                
                response = {
                    "action": "drop",
                    "view_name": view_name,
                    "status": "dropped",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"Dropped view: {view_name}")
                
            elif action == 'info':
                if not view_name:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="view_name is required for 'info' action"
                        )]
                    )
                
                # Get view information
                info_query = f"""
                SELECT name, sql 
                FROM sqlite_master 
                WHERE type='view' AND name='{view_name}'
                """
                
                view_info = await self.db_manager.execute_query(info_query)
                
                if not view_info:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"View '{view_name}' does not exist"
                        )]
                    )
                
                # Get sample data from view
                sample_query = f"SELECT * FROM {view_name} LIMIT 5"
                sample_data = await self.db_manager.execute_query(sample_query)
                
                response = {
                    "action": "info",
                    "view_name": view_name,
                    "definition": view_info[0],
                    "sample_data": sample_data
                }
                
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Invalid action '{action}'. Valid actions: 'list', 'drop', 'info'"
                    )]
                )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error managing views: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error managing views: {str(e)}"
                )]
            )
    
    async def manage_database_indexes(
        self,
        action: str,
        table_name: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> CallToolResult:
        """
        Manage database indexes (list, drop, or get info).
        
        Args:
            action: Action to perform ('list', 'drop', 'info')
            table_name: Name of table (for 'list' action to filter by table)
            index_name: Name of index (required for 'drop' and 'info' actions)
            
        Returns:
            Index management results
        """
        try:
            if action == 'list':
                # List indexes
                if table_name:
                    query = f"""
                    SELECT name, sql, tbl_name 
                    FROM sqlite_master 
                    WHERE type='index' AND tbl_name='{table_name}'
                    AND name NOT LIKE 'sqlite_autoindex_%'
                    ORDER BY name
                    """
                else:
                    query = """
                    SELECT name, sql, tbl_name 
                    FROM sqlite_master 
                    WHERE type='index' 
                    AND name NOT LIKE 'sqlite_autoindex_%'
                    ORDER BY tbl_name, name
                    """
                
                indexes = await self.db_manager.execute_query(query)
                
                response = {
                    "action": "list",
                    "table_filter": table_name,
                    "indexes_count": len(indexes) if indexes else 0,
                    "indexes": indexes
                }
                
            elif action == 'drop':
                if not index_name:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="index_name is required for 'drop' action"
                        )]
                    )
                
                # Check if index exists
                check_query = f"""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name='{index_name}'
                """
                
                existing = await self.db_manager.execute_query(check_query)
                
                if not existing:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Index '{index_name}' does not exist"
                        )]
                    )
                
                # Drop the index
                drop_query = f"DROP INDEX {index_name}"
                await self.db_manager.execute_query(drop_query)
                
                response = {
                    "action": "drop",
                    "index_name": index_name,
                    "status": "dropped",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"Dropped index: {index_name}")
                
            elif action == 'info':
                if not index_name:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="index_name is required for 'info' action"
                        )]
                    )
                
                # Get index information
                info_query = f"""
                SELECT name, sql, tbl_name 
                FROM sqlite_master 
                WHERE type='index' AND name='{index_name}'
                """
                
                index_info = await self.db_manager.execute_query(info_query)
                
                if not index_info:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Index '{index_name}' does not exist"
                        )]
                    )
                
                response = {
                    "action": "info",
                    "index_name": index_name,
                    "index_info": index_info[0]
                }
                
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Invalid action '{action}'. Valid actions: 'list', 'drop', 'info'"
                    )]
                )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error managing indexes: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error managing indexes: {str(e)}"
                )]
            )
    
    async def explain_query_plan(self, query: str) -> CallToolResult:
        """
        Get the execution plan for a SQL query.
        
        Args:
            query: SQL query to explain
            
        Returns:
            Query execution plan
        """
        try:
            # Validate query syntax
            try:
                parsed = sqlparse.parse(query.strip())[0]
                if not parsed.tokens:
                    raise ValueError("Empty query")
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Invalid query syntax: {str(e)}"
                    )]
                )
            
            # Get query plan using EXPLAIN QUERY PLAN
            explain_query = f"EXPLAIN QUERY PLAN {query}"
            
            plan_results = await self.db_manager.execute_query(explain_query)
            
            # Also get basic EXPLAIN for more detailed info
            basic_explain_query = f"EXPLAIN {query}"
            basic_explain = await self.db_manager.execute_query(basic_explain_query)
            
            response = {
                "original_query": query,
                "query_plan": plan_results,
                "detailed_explain": basic_explain[:10] if basic_explain else [],  # Limit output
                "analysis": self._analyze_query_plan(plan_results),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error explaining query plan: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error explaining query plan: {str(e)}"
                )]
            )
    
    def _analyze_query_plan(self, plan_results: List[Dict]) -> Dict[str, Any]:
        """Analyze query execution plan and provide insights."""
        if not plan_results:
            return {"analysis": "No plan data available"}
        
        analysis = {
            "operations_count": len(plan_results),
            "operations": [],
            "recommendations": []
        }
        
        # Analyze each operation in the plan
        for step in plan_results:
            detail = step.get('detail', '')
            operation = {
                "step": step.get('id', 0),
                "parent": step.get('parent', 0),
                "detail": detail
            }
            
            # Look for performance indicators
            if 'SCAN' in detail.upper() and 'INDEX' not in detail.upper():
                operation["warning"] = "Table scan detected - consider adding indexes"
                analysis["recommendations"].append(f"Consider creating index for table scan in: {detail}")
            
            if 'TEMP' in detail.upper():
                operation["info"] = "Temporary table/index used"
            
            if 'SORT' in detail.upper():
                operation["info"] = "Sorting operation detected"
            
            analysis["operations"].append(operation)
        
        # General recommendations based on plan complexity
        if len(plan_results) > 10:
            analysis["recommendations"].append("Complex query with many operations - consider simplification")
        
        return analysis