"""
Data Ingestion Tools for MCP Server.

Provides tools for loading and ingesting data from various formats (CSV, JSON)
into the database with schema inference and validation capabilities.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import csv

import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from database.connection import get_database_manager

logger = logging.getLogger(__name__)


class DataIngestionTools:
    """Tools for data ingestion and loading operations."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    async def load_csv_data(
        self, 
        file_path: str, 
        table_name: str, 
        schema_inference: bool = True,
        delimiter: str = ",",
        encoding: str = "utf-8",
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> CallToolResult:
        """
        Load data from a CSV file into a database table.
        
        Args:
            file_path: Path to the CSV file
            table_name: Name of the target database table
            schema_inference: Whether to automatically infer data types
            delimiter: CSV delimiter character
            encoding: File encoding
            skip_rows: Number of rows to skip at the beginning
            max_rows: Maximum number of rows to load
            
        Returns:
            Result containing success status and details
        """
        try:
            # Validate file path
            if not os.path.exists(file_path):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: File not found: {file_path}"
                    )]
                )
            
            # Read CSV file
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                skiprows=skip_rows,
                nrows=max_rows
            )
            
            if df.empty:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: CSV file is empty or contains no valid data"
                    )]
                )
            
            # Infer schema if requested
            if schema_inference:
                schema_info = self._infer_schema(df)
            else:
                schema_info = {col: "TEXT" for col in df.columns}
            
            # Create table with inferred schema
            await self._create_table_with_schema(table_name, schema_info)
            
            # Insert data
            rows_inserted = await self._insert_dataframe(df, table_name)
            
            # Log the operation
            await self._log_data_ingestion(
                table_name, file_path, "CSV", len(df), rows_inserted
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": f"Successfully loaded {rows_inserted} rows into table '{table_name}'",
                        "table_name": table_name,
                        "rows_loaded": rows_inserted,
                        "columns": list(df.columns),
                        "schema": schema_info
                    }, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error loading CSV data: {str(e)}"
                )]
            )
    
    async def load_json_data(
        self, 
        file_path: str, 
        table_name: str, 
        normalize_nested: bool = True,
        json_lines: bool = False,
        max_records: Optional[int] = None
    ) -> CallToolResult:
        """
        Load data from a JSON file into a database table.
        
        Args:
            file_path: Path to the JSON file
            table_name: Name of the target database table
            normalize_nested: Whether to normalize nested JSON objects
            json_lines: Whether the file is in JSON Lines format
            max_records: Maximum number of records to load
            
        Returns:
            Result containing success status and details
        """
        try:
            # Validate file path
            if not os.path.exists(file_path):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: File not found: {file_path}"
                    )]
                )
            
            # Read JSON file
            if json_lines:
                # Read JSON Lines format
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if max_records and i >= max_records:
                            break
                        if line.strip():
                            data.append(json.loads(line))
                df = pd.DataFrame(data)
            else:
                # Read regular JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data[:max_records] if max_records else data)
                else:
                    # Single object, convert to single-row DataFrame
                    df = pd.DataFrame([data])
            
            if df.empty:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: JSON file is empty or contains no valid data"
                    )]
                )
            
            # Normalize nested objects if requested
            if normalize_nested:
                df = pd.json_normalize(df.to_dict('records'))
            
            # Infer schema
            schema_info = self._infer_schema(df)
            
            # Create table with inferred schema
            await self._create_table_with_schema(table_name, schema_info)
            
            # Insert data
            rows_inserted = await self._insert_dataframe(df, table_name)
            
            # Log the operation
            await self._log_data_ingestion(
                table_name, file_path, "JSON", len(df), rows_inserted
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": f"Successfully loaded {rows_inserted} rows into table '{table_name}'",
                        "table_name": table_name,
                        "rows_loaded": rows_inserted,
                        "columns": list(df.columns),
                        "schema": schema_info
                    }, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error loading JSON data: {str(e)}"
                )]
            )
    
    async def create_table_from_schema(
        self, 
        table_name: str, 
        columns: List[str], 
        data_types: List[str],
        constraints: Optional[List[str]] = None
    ) -> CallToolResult:
        """
        Create a database table with specified schema.
        
        Args:
            table_name: Name of the table to create
            columns: List of column names
            data_types: List of data types for each column
            constraints: Optional list of constraints (PRIMARY KEY, UNIQUE, etc.)
            
        Returns:
            Result containing success status and details
        """
        try:
            if len(columns) != len(data_types):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: Number of columns must match number of data types"
                    )]
                )
            
            # Build CREATE TABLE statement
            column_definitions = [
                f"{col} {dtype}" for col, dtype in zip(columns, data_types)
            ]
            
            if constraints:
                column_definitions.extend(constraints)
            
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(column_definitions)}
            )
            """
            
            # Execute table creation
            await self.db_manager.execute_non_query(create_query)
            
            # Register table in metadata
            schema_info = dict(zip(columns, data_types))
            await self._register_table_metadata(table_name, schema_info)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": f"Table '{table_name}' created successfully",
                        "table_name": table_name,
                        "columns": columns,
                        "data_types": data_types,
                        "constraints": constraints or []
                    }, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error creating table: {str(e)}"
                )]
            )
    
    async def validate_data_format(
        self, 
        file_path: str, 
        file_type: str,
        expected_schema: Optional[Dict[str, str]] = None
    ) -> CallToolResult:
        """
        Validate data file format and schema.
        
        Args:
            file_path: Path to the data file
            file_type: Type of file (csv, json)
            expected_schema: Expected column names and data types
            
        Returns:
            Validation results
        """
        try:
            if not os.path.exists(file_path):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: File not found: {file_path}"
                    )]
                )
            
            validation_results = {
                "file_exists": True,
                "file_size": os.path.getsize(file_path),
                "file_type": file_type,
                "valid_format": False,
                "schema_match": False,
                "issues": []
            }
            
            # Validate file format
            try:
                if file_type.lower() == "csv":
                    df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows
                elif file_type.lower() == "json":
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list) and data:
                        df = pd.DataFrame(data[:5])
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                    else:
                        raise ValueError("Invalid JSON structure")
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                validation_results["valid_format"] = True
                validation_results["columns"] = list(df.columns)
                validation_results["sample_rows"] = len(df)
                
                # Validate schema if provided
                if expected_schema:
                    schema_issues = []
                    actual_columns = set(df.columns)
                    expected_columns = set(expected_schema.keys())
                    
                    missing_columns = expected_columns - actual_columns
                    extra_columns = actual_columns - expected_columns
                    
                    if missing_columns:
                        schema_issues.append(f"Missing columns: {list(missing_columns)}")
                    
                    if extra_columns:
                        schema_issues.append(f"Extra columns: {list(extra_columns)}")
                    
                    validation_results["schema_match"] = len(schema_issues) == 0
                    validation_results["schema_issues"] = schema_issues
                
            except Exception as format_error:
                validation_results["issues"].append(f"Format error: {str(format_error)}")
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(validation_results, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error validating data format: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error validating data format: {str(e)}"
                )]
            )
    
    async def infer_schema(self, file_path: str, file_type: str) -> CallToolResult:
        """
        Infer schema from data file.
        
        Args:
            file_path: Path to the data file
            file_type: Type of file (csv, json)
            
        Returns:
            Inferred schema information
        """
        try:
            if not os.path.exists(file_path):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: File not found: {file_path}"
                    )]
                )
            
            # Read sample data
            if file_type.lower() == "csv":
                df = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows
            elif file_type.lower() == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data[:1000])
                else:
                    df = pd.DataFrame([data])
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: Unsupported file type: {file_type}"
                    )]
                )
            
            # Infer schema
            schema_info = self._infer_schema(df)
            
            # Get additional statistics
            stats = {
                "total_columns": len(df.columns),
                "sample_rows": len(df),
                "null_counts": df.isnull().sum().to_dict(),
                "unique_counts": df.nunique().to_dict()
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "schema": schema_info,
                        "statistics": stats,
                        "columns": list(df.columns)
                    }, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error inferring schema: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error inferring schema: {str(e)}"
                )]
            )
    
    def _infer_schema(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer SQL data types from pandas DataFrame."""
        schema = {}
        
        for column in df.columns:
            dtype = df[column].dtype
            
            if pd.api.types.is_integer_dtype(dtype):
                schema[column] = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                schema[column] = "REAL"
            elif pd.api.types.is_bool_dtype(dtype):
                schema[column] = "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema[column] = "TIMESTAMP"
            else:
                # Check if it's a date string
                if self._is_date_column(df[column]):
                    schema[column] = "DATE"
                # Check if it's numeric but stored as string
                elif self._is_numeric_string(df[column]):
                    schema[column] = "REAL"
                else:
                    # Determine text length
                    max_length = df[column].astype(str).str.len().max()
                    if max_length > 255:
                        schema[column] = "TEXT"
                    else:
                        schema[column] = f"VARCHAR({max_length + 50})"  # Add buffer
        
        return schema
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a series contains date strings."""
        try:
            pd.to_datetime(series.dropna().head(10))
            return True
        except:
            return False
    
    def _is_numeric_string(self, series: pd.Series) -> bool:
        """Check if a series contains numeric values stored as strings."""
        try:
            pd.to_numeric(series.dropna().head(10))
            return True
        except:
            return False
    
    async def _create_table_with_schema(self, table_name: str, schema: Dict[str, str]):
        """Create table with specified schema."""
        columns_def = [f"{col} {dtype}" for col, dtype in schema.items()]
        create_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {', '.join(columns_def)}
        )
        """
        await self.db_manager.execute_non_query(create_query)
    
    async def _insert_dataframe(self, df: pd.DataFrame, table_name: str) -> int:
        """Insert DataFrame data into table."""
        # Replace NaN values with None
        df = df.replace({np.nan: None})
        
        # Convert to list of dictionaries
        records = df.to_dict('records')
        
        if not records:
            return 0
        
        # Build INSERT query
        columns = list(records[0].keys())
        placeholders = ", ".join([f":{col}" for col in columns])
        insert_query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({placeholders})
        """
        
        # Execute batch insert
        total_inserted = 0
        batch_size = 1000
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            for record in batch:
                await self.db_manager.execute_non_query(insert_query, record)
                total_inserted += 1
        
        return total_inserted
    
    async def _register_table_metadata(self, table_name: str, schema_info: Dict[str, str]):
        """Register table metadata in the data_tables table."""
        query = """
        INSERT OR REPLACE INTO data_tables (table_name, schema_info, created_at, updated_at)
        VALUES (:table_name, :schema_info, datetime('now'), datetime('now'))
        """
        
        await self.db_manager.execute_non_query(query, {
            'table_name': table_name,
            'schema_info': json.dumps(schema_info)
        })
    
    async def _log_data_ingestion(
        self, 
        table_name: str, 
        source_file: str, 
        file_type: str, 
        total_rows: int,
        inserted_rows: int
    ):
        """Log data ingestion operation."""
        query = """
        INSERT INTO query_logs (
            session_id, natural_query, sql_query, query_type, 
            rows_affected, success, created_at
        )
        VALUES (:session_id, :natural_query, :sql_query, :query_type, 
                :rows_affected, :success, datetime('now'))
        """
        
        await self.db_manager.execute_non_query(query, {
            'session_id': 'data_ingestion',
            'natural_query': f"Load {file_type} file: {source_file}",
            'sql_query': f"Data ingestion into table: {table_name}",
            'query_type': 'INSERT',
            'rows_affected': inserted_rows,
            'success': True
        })