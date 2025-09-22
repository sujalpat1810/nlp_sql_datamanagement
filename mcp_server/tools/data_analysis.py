"""
Data Analysis Tools for MCP Server.

Provides comprehensive tools for data analysis, statistical operations,
data quality assessment, and insight generation.
"""

import json
import logging
import statistics
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import math

import pandas as pd
import numpy as np
from mcp.types import CallToolResult, TextContent

from database.connection import get_database_manager

logger = logging.getLogger(__name__)


class DataAnalysisTools:
    """Tools for comprehensive data analysis and statistics."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    async def get_table_schema(self, table_name: str) -> CallToolResult:
        """
        Get detailed schema information for a database table.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Comprehensive table schema information
        """
        try:
            # Get basic table info
            table_info = await self.db_manager.get_table_info(table_name)
            
            if not table_info.get('exists', False):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: Table '{table_name}' does not exist"
                    )]
                )
            
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            count_result = await self.db_manager.execute_query(count_query)
            row_count = count_result[0]['row_count'] if count_result else 0
            
            # Get sample data for additional analysis
            sample_query = f"SELECT * FROM {table_name} LIMIT 5"
            sample_data = await self.db_manager.execute_query(sample_query)
            
            # Enhanced schema information
            schema_info = {
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(table_info['columns']),
                "columns": table_info['columns'],
                "sample_data": sample_data,
                "table_size_info": await self._get_table_size_info(table_name),
                "indexes": await self._get_table_indexes(table_name)
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(schema_info, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error getting table schema: {str(e)}"
                )]
            )
    
    async def get_data_summary(
        self, 
        table_name: str, 
        columns: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> CallToolResult:
        """
        Generate comprehensive data summary statistics.
        
        Args:
            table_name: Name of the table to summarize
            columns: Specific columns to analyze (None for all)
            sample_size: Number of rows to sample for analysis
            
        Returns:
            Detailed data summary with statistics
        """
        try:
            # Validate table exists
            table_info = await self.db_manager.get_table_info(table_name)
            if not table_info.get('exists', False):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: Table '{table_name}' does not exist"
                    )]
                )
            
            # Build query
            column_list = "*"
            if columns:
                column_list = ", ".join(columns)
            
            query = f"SELECT {column_list} FROM {table_name}"
            if sample_size:
                query += f" ORDER BY RANDOM() LIMIT {sample_size}"
            
            # Execute query
            data = await self.db_manager.execute_query(query)
            
            if not data:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"No data found in table '{table_name}'"
                    )]
                )
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(data)
            
            # Generate summary statistics
            summary = {
                "table_name": table_name,
                "total_rows": len(data),
                "columns_analyzed": list(df.columns),
                "summary_statistics": {},
                "data_types": {},
                "null_analysis": {},
                "unique_analysis": {},
                "data_quality_score": 0
            }
            
            # Analyze each column
            for column in df.columns:
                col_data = df[column]
                
                # Basic statistics
                col_summary = {
                    "count": len(col_data),
                    "null_count": col_data.isnull().sum(),
                    "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100,
                    "unique_count": col_data.nunique(),
                    "unique_percentage": (col_data.nunique() / len(col_data)) * 100
                }
                
                # Data type analysis
                inferred_type = self._infer_column_type(col_data)
                summary["data_types"][column] = inferred_type
                
                # Type-specific analysis
                if inferred_type in ['integer', 'float']:
                    numeric_stats = self._calculate_numeric_stats(col_data)
                    col_summary.update(numeric_stats)
                elif inferred_type == 'datetime':
                    date_stats = self._calculate_date_stats(col_data)
                    col_summary.update(date_stats)
                elif inferred_type == 'string':
                    text_stats = self._calculate_text_stats(col_data)
                    col_summary.update(text_stats)
                
                summary["summary_statistics"][column] = col_summary
                summary["null_analysis"][column] = col_summary["null_percentage"]
                summary["unique_analysis"][column] = col_summary["unique_percentage"]
            
            # Calculate overall data quality score
            summary["data_quality_score"] = self._calculate_data_quality_score(summary)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(summary, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error generating data summary: {str(e)}"
                )]
            )
    
    async def calculate_statistics(
        self, 
        table_name: str, 
        column_name: str, 
        stats: List[str] = ['mean', 'median', 'std', 'min', 'max']
    ) -> CallToolResult:
        """
        Calculate specific statistical measures for a column.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to analyze
            stats: List of statistics to calculate
            
        Returns:
            Calculated statistics
        """
        try:
            # Get column data
            query = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL"
            data = await self.db_manager.execute_query(query)
            
            if not data:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"No non-null data found in column '{column_name}'"
                    )]
                )
            
            # Extract values
            values = [row[column_name] for row in data]
            
            # Try to convert to numeric
            try:
                numeric_values = [float(v) for v in values if v is not None]
            except (ValueError, TypeError):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Column '{column_name}' contains non-numeric data"
                    )]
                )
            
            if not numeric_values:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"No numeric values found in column '{column_name}'"
                    )]
                )
            
            # Calculate requested statistics
            results = {
                "table_name": table_name,
                "column_name": column_name,
                "sample_size": len(numeric_values),
                "statistics": {}
            }
            
            for stat in stats:
                if stat == 'mean':
                    results["statistics"]["mean"] = statistics.mean(numeric_values)
                elif stat == 'median':
                    results["statistics"]["median"] = statistics.median(numeric_values)
                elif stat == 'mode':
                    try:
                        results["statistics"]["mode"] = statistics.mode(numeric_values)
                    except statistics.StatisticsError:
                        results["statistics"]["mode"] = "No unique mode"
                elif stat == 'std':
                    if len(numeric_values) > 1:
                        results["statistics"]["std"] = statistics.stdev(numeric_values)
                    else:
                        results["statistics"]["std"] = 0
                elif stat == 'var':
                    if len(numeric_values) > 1:
                        results["statistics"]["variance"] = statistics.variance(numeric_values)
                    else:
                        results["statistics"]["variance"] = 0
                elif stat == 'min':
                    results["statistics"]["min"] = min(numeric_values)
                elif stat == 'max':
                    results["statistics"]["max"] = max(numeric_values)
                elif stat == 'range':
                    results["statistics"]["range"] = max(numeric_values) - min(numeric_values)
                elif stat == 'q1':
                    results["statistics"]["q1"] = np.percentile(numeric_values, 25)
                elif stat == 'q3':
                    results["statistics"]["q3"] = np.percentile(numeric_values, 75)
                elif stat == 'iqr':
                    q1 = np.percentile(numeric_values, 25)
                    q3 = np.percentile(numeric_values, 75)
                    results["statistics"]["iqr"] = q3 - q1
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(results, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error calculating statistics: {str(e)}"
                )]
            )
    
    async def find_null_values(self, table_name: str) -> CallToolResult:
        """
        Find and analyze null values across all columns.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Null value analysis results
        """
        try:
            # Get table schema
            table_info = await self.db_manager.get_table_info(table_name)
            if not table_info.get('exists', False):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: Table '{table_name}' does not exist"
                    )]
                )
            
            # Get total row count
            total_query = f"SELECT COUNT(*) as total FROM {table_name}"
            total_result = await self.db_manager.execute_query(total_query)
            total_rows = total_result[0]['total'] if total_result else 0
            
            if total_rows == 0:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Table '{table_name}' is empty"
                    )]
                )
            
            null_analysis = {
                "table_name": table_name,
                "total_rows": total_rows,
                "columns": {},
                "summary": {
                    "columns_with_nulls": 0,
                    "total_null_values": 0,
                    "highest_null_percentage": 0,
                    "lowest_null_percentage": 100
                }
            }
            
            # Analyze each column for null values
            columns = [col['name'] if isinstance(col, dict) else str(col) 
                      for col in table_info['columns']]
            
            for column in columns:
                if column.lower() in ['id', 'rowid']:  # Skip system columns
                    continue
                
                # Count null values
                null_query = f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT({column}) as non_null,
                    COUNT(*) - COUNT({column}) as null_count
                FROM {table_name}
                """
                
                result = await self.db_manager.execute_query(null_query)
                if result:
                    null_count = result[0]['null_count']
                    null_percentage = (null_count / total_rows) * 100
                    
                    null_analysis["columns"][column] = {
                        "null_count": null_count,
                        "non_null_count": result[0]['non_null'],
                        "null_percentage": round(null_percentage, 2)
                    }
                    
                    # Update summary
                    if null_count > 0:
                        null_analysis["summary"]["columns_with_nulls"] += 1
                        null_analysis["summary"]["total_null_values"] += null_count
                    
                    if null_percentage > null_analysis["summary"]["highest_null_percentage"]:
                        null_analysis["summary"]["highest_null_percentage"] = null_percentage
                    
                    if null_percentage < null_analysis["summary"]["lowest_null_percentage"]:
                        null_analysis["summary"]["lowest_null_percentage"] = null_percentage
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(null_analysis, indent=2)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error finding null values: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error finding null values: {str(e)}"
                )]
            )
    
    async def detect_outliers(
        self, 
        table_name: str, 
        column_name: str, 
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> CallToolResult:
        """
        Detect outliers in a numeric column using various methods.
        
        Args:
            table_name: Name of the table
            column_name: Name of the numeric column
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold value for outlier detection
            
        Returns:
            Outlier detection results
        """
        try:
            # Get numeric data
            query = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL"
            data = await self.db_manager.execute_query(query)
            
            if not data:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"No data found in column '{column_name}'"
                    )]
                )
            
            # Convert to numeric values
            try:
                values = [float(row[column_name]) for row in data]
            except (ValueError, TypeError):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Column '{column_name}' contains non-numeric data"
                    )]
                )
            
            if len(values) < 3:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Insufficient data for outlier detection (need at least 3 values)"
                    )]
                )
            
            # Detect outliers using specified method
            outliers = []
            outlier_info = {
                "table_name": table_name,
                "column_name": column_name,
                "method": method,
                "threshold": threshold,
                "total_values": len(values),
                "outliers": [],
                "outlier_count": 0,
                "outlier_percentage": 0,
                "statistics": {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0
                }
            }
            
            if method == 'iqr':
                outliers = self._detect_iqr_outliers(values, threshold)
            elif method == 'zscore':
                outliers = self._detect_zscore_outliers(values, threshold)
            elif method == 'modified_zscore':
                outliers = self._detect_modified_zscore_outliers(values, threshold)
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Unknown outlier detection method: {method}"
                    )]
                )
            
            outlier_info["outliers"] = outliers
            outlier_info["outlier_count"] = len(outliers)
            outlier_info["outlier_percentage"] = round((len(outliers) / len(values)) * 100, 2)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(outlier_info, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error detecting outliers: {str(e)}"
                )]
            )
    
    async def get_unique_values(
        self, 
        table_name: str, 
        column_name: str, 
        limit: int = 100
    ) -> CallToolResult:
        """
        Get unique values and their frequencies for a column.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            limit: Maximum number of unique values to return
            
        Returns:
            Unique values with frequency analysis
        """
        try:
            # Get unique values with counts
            query = f"""
            SELECT {column_name}, COUNT(*) as frequency
            FROM {table_name} 
            WHERE {column_name} IS NOT NULL
            GROUP BY {column_name}
            ORDER BY frequency DESC
            LIMIT {limit}
            """
            
            results = await self.db_manager.execute_query(query)
            
            if not results:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"No data found in column '{column_name}'"
                    )]
                )
            
            # Get total non-null count
            total_query = f"SELECT COUNT({column_name}) as total FROM {table_name}"
            total_result = await self.db_manager.execute_query(total_query)
            total_count = total_result[0]['total'] if total_result else 0
            
            # Process results
            unique_analysis = {
                "table_name": table_name,
                "column_name": column_name,
                "total_non_null_values": total_count,
                "unique_values_count": len(results),
                "showing_top": limit,
                "values": []
            }
            
            for row in results:
                value = row[column_name]
                frequency = row['frequency']
                percentage = round((frequency / total_count) * 100, 2) if total_count > 0 else 0
                
                unique_analysis["values"].append({
                    "value": value,
                    "frequency": frequency,
                    "percentage": percentage
                })
            
            # Add summary statistics
            if results:
                frequencies = [row['frequency'] for row in results]
                unique_analysis["frequency_statistics"] = {
                    "mean_frequency": round(statistics.mean(frequencies), 2),
                    "median_frequency": statistics.median(frequencies),
                    "max_frequency": max(frequencies),
                    "min_frequency": min(frequencies)
                }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(unique_analysis, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error getting unique values: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error getting unique values: {str(e)}"
                )]
            )
    
    async def analyze_data_quality(self, table_name: str) -> CallToolResult:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Comprehensive data quality report
        """
        try:
            # Get basic table info
            table_info = await self.db_manager.get_table_info(table_name)
            if not table_info.get('exists', False):
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: Table '{table_name}' does not exist"
                    )]
                )
            
            # Get sample data for analysis
            sample_query = f"SELECT * FROM {table_name} LIMIT 1000"
            sample_data = await self.db_manager.execute_query(sample_query)
            
            if not sample_data:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"No data found in table '{table_name}'"
                    )]
                )
            
            df = pd.DataFrame(sample_data)
            
            quality_report = {
                "table_name": table_name,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "sample_size": len(sample_data),
                "total_columns": len(df.columns),
                "overall_quality_score": 0,
                "quality_dimensions": {},
                "issues": [],
                "recommendations": []
            }
            
            # Completeness Analysis
            completeness_scores = []
            for column in df.columns:
                null_percentage = (df[column].isnull().sum() / len(df)) * 100
                completeness_score = 100 - null_percentage
                completeness_scores.append(completeness_score)
                
                if null_percentage > 50:
                    quality_report["issues"].append(f"Column '{column}' has {null_percentage:.1f}% null values")
                    quality_report["recommendations"].append(f"Consider data imputation or removal for column '{column}'")
            
            quality_report["quality_dimensions"]["completeness"] = {
                "score": round(statistics.mean(completeness_scores), 2),
                "column_scores": dict(zip(df.columns, completeness_scores))
            }
            
            # Uniqueness Analysis
            uniqueness_scores = []
            for column in df.columns:
                unique_percentage = (df[column].nunique() / len(df)) * 100
                uniqueness_scores.append(unique_percentage)
                
                if unique_percentage == 100 and column.lower() != 'id':
                    quality_report["issues"].append(f"Column '{column}' has all unique values (potential identifier)")
                elif unique_percentage < 1:
                    quality_report["issues"].append(f"Column '{column}' has very low uniqueness ({unique_percentage:.1f}%)")
            
            quality_report["quality_dimensions"]["uniqueness"] = {
                "average_uniqueness": round(statistics.mean(uniqueness_scores), 2),
                "column_uniqueness": dict(zip(df.columns, uniqueness_scores))
            }
            
            # Consistency Analysis (data type consistency)
            consistency_issues = []
            for column in df.columns:
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    # Check for mixed data types
                    types = col_data.apply(type).unique()
                    if len(types) > 1:
                        consistency_issues.append(f"Column '{column}' has mixed data types")
            
            quality_report["quality_dimensions"]["consistency"] = {
                "issues_found": len(consistency_issues),
                "details": consistency_issues
            }
            
            # Validity Analysis (basic format checks)
            validity_issues = []
            for column in df.columns:
                col_data = df[column].dropna().astype(str)
                
                # Check for common validity issues
                if 'email' in column.lower():
                    invalid_emails = col_data[~col_data.str.contains(r'^[^@]+@[^@]+\.[^@]+$', na=False)]
                    if len(invalid_emails) > 0:
                        validity_issues.append(f"Column '{column}' contains {len(invalid_emails)} invalid email formats")
                
                if 'phone' in column.lower():
                    # Basic phone number validation
                    invalid_phones = col_data[~col_data.str.match(r'^[\d\-\+\(\)\s]+$', na=False)]
                    if len(invalid_phones) > 0:
                        validity_issues.append(f"Column '{column}' contains {len(invalid_phones)} invalid phone formats")
            
            quality_report["quality_dimensions"]["validity"] = {
                "issues_found": len(validity_issues),
                "details": validity_issues
            }
            
            # Calculate overall quality score
            completeness_avg = quality_report["quality_dimensions"]["completeness"]["score"]
            consistency_score = max(0, 100 - len(consistency_issues) * 10)
            validity_score = max(0, 100 - len(validity_issues) * 5)
            
            overall_score = (completeness_avg + consistency_score + validity_score) / 3
            quality_report["overall_quality_score"] = round(overall_score, 2)
            
            # Add overall recommendations
            if overall_score < 70:
                quality_report["recommendations"].append("Data quality needs significant improvement")
            elif overall_score < 85:
                quality_report["recommendations"].append("Data quality could be improved")
            else:
                quality_report["recommendations"].append("Data quality is good")
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(quality_report, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error analyzing data quality: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error analyzing data quality: {str(e)}"
                )]
            )
    
    # Helper methods
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer the data type of a pandas Series."""
        if pd.api.types.is_integer_dtype(series):
            return 'integer'
        elif pd.api.types.is_float_dtype(series):
            return 'float'
        elif pd.api.types.is_bool_dtype(series):
            return 'boolean'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        else:
            # Check if string data could be dates
            try:
                pd.to_datetime(series.dropna().head(10))
                return 'datetime'
            except:
                return 'string'
    
    def _calculate_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for numeric columns."""
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) == 0:
            return {}
        
        return {
            "mean": float(numeric_series.mean()),
            "median": float(numeric_series.median()),
            "std": float(numeric_series.std()) if len(numeric_series) > 1 else 0,
            "min": float(numeric_series.min()),
            "max": float(numeric_series.max()),
            "q1": float(numeric_series.quantile(0.25)),
            "q3": float(numeric_series.quantile(0.75))
        }
    
    def _calculate_date_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for date columns."""
        try:
            date_series = pd.to_datetime(series, errors='coerce').dropna()
            
            if len(date_series) == 0:
                return {}
            
            return {
                "earliest_date": date_series.min().isoformat(),
                "latest_date": date_series.max().isoformat(),
                "date_range_days": (date_series.max() - date_series.min()).days
            }
        except:
            return {}
    
    def _calculate_text_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for text columns."""
        text_series = series.dropna().astype(str)
        
        if len(text_series) == 0:
            return {}
        
        lengths = text_series.str.len()
        
        return {
            "avg_length": float(lengths.mean()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "total_characters": int(lengths.sum())
        }
    
    def _calculate_data_quality_score(self, summary: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        scores = []
        
        # Completeness score (based on null percentages)
        null_percentages = list(summary["null_analysis"].values())
        if null_percentages:
            avg_null_pct = statistics.mean(null_percentages)
            completeness_score = max(0, 100 - avg_null_pct)
            scores.append(completeness_score)
        
        # Uniqueness diversity score
        unique_percentages = list(summary["unique_analysis"].values())
        if unique_percentages:
            # Prefer columns with moderate uniqueness (not too low, not always 100%)
            uniqueness_score = statistics.mean([
                100 - abs(50 - pct) * 2 if pct != 100 else 80 
                for pct in unique_percentages
            ])
            scores.append(max(0, uniqueness_score))
        
        return round(statistics.mean(scores) if scores else 0, 2)
    
    def _detect_iqr_outliers(self, values: List[float], threshold: float = 1.5) -> List[float]:
        """Detect outliers using IQR method."""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        return [v for v in values if v < lower_bound or v > upper_bound]
    
    def _detect_zscore_outliers(self, values: List[float], threshold: float = 3.0) -> List[float]:
        """Detect outliers using Z-score method."""
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        
        if std == 0:
            return []
        
        return [v for v in values if abs((v - mean) / std) > threshold]
    
    def _detect_modified_zscore_outliers(self, values: List[float], threshold: float = 3.5) -> List[float]:
        """Detect outliers using modified Z-score method."""
        median = statistics.median(values)
        mad = statistics.median([abs(v - median) for v in values])
        
        if mad == 0:
            return []
        
        modified_z_scores = [0.6745 * (v - median) / mad for v in values]
        return [values[i] for i, score in enumerate(modified_z_scores) if abs(score) > threshold]
    
    async def _get_table_size_info(self, table_name: str) -> Dict[str, Any]:
        """Get table size information."""
        try:
            # This is database-specific - simplified for SQLite
            if 'sqlite' in self.db_manager.database_url:
                query = f"SELECT COUNT(*) as row_count FROM {table_name}"
                result = await self.db_manager.execute_query(query)
                return {
                    "row_count": result[0]['row_count'] if result else 0,
                    "estimated_size": "N/A for SQLite"
                }
            else:
                # PostgreSQL version
                query = f"""
                SELECT 
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation
                FROM pg_stats 
                WHERE tablename = '{table_name}'
                """
                result = await self.db_manager.execute_query(query)
                return {"pg_stats": result}
        except:
            return {"error": "Could not retrieve size information"}
    
    async def _get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table index information."""
        try:
            if 'sqlite' in self.db_manager.database_url:
                query = f"PRAGMA index_list({table_name})"
                result = await self.db_manager.execute_query(query)
                return result
            else:
                # PostgreSQL version
                query = f"""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = '{table_name}'
                """
                result = await self.db_manager.execute_query(query)
                return result
        except:
            return []