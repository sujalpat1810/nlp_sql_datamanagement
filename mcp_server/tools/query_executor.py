"""
Query executor for running SQL queries on in-memory data
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class InMemoryQueryExecutor:
    """Execute SQL queries on JSON/CSV data loaded into SQLite in-memory database"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.conn = None
        self.tables = {}
        
    def load_data(self) -> Dict[str, int]:
        """Load all data files into in-memory SQLite database"""
        self.conn = sqlite3.connect(':memory:')
        loaded_tables = {}
        
        # Load all JSON files
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if isinstance(data, list) and len(data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    
                    # Create table name from filename
                    table_name = json_file.stem
                    
                    # Write to SQLite
                    df.to_sql(table_name, self.conn, if_exists='replace', index=False)
                    loaded_tables[table_name] = len(data)
                    self.tables[table_name] = list(df.columns)
                    
                    logger.info(f"Loaded {len(data)} records into table '{table_name}'")
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                
        return loaded_tables
    
    def execute_query(self, sql: str) -> Tuple[List[Dict], List[str], str]:
        """
        Execute SQL query and return results
        
        Returns:
            Tuple of (results, columns, error_message)
        """
        if not self.conn:
            self.load_data()
            
        try:
            # Execute query
            cursor = self.conn.cursor()
            cursor.execute(sql)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert to list of dicts
            results = []
            for row in rows:
                result_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # Convert to JSON-serializable types
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        result_dict[col] = value
                    else:
                        result_dict[col] = str(value)
                results.append(result_dict)
            
            return results, columns, None
            
        except sqlite3.Error as e:
            error_msg = f"SQL Error: {str(e)}"
            logger.error(error_msg)
            return [], [], error_msg
        except Exception as e:
            error_msg = f"Execution Error: {str(e)}"
            logger.error(error_msg)
            return [], [], error_msg
    
    def get_table_info(self) -> Dict[str, List[str]]:
        """Get information about available tables and their columns"""
        if not self.conn:
            self.load_data()
        return self.tables
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


# Singleton instance
_executor = None

def get_query_executor(data_dir: Path) -> InMemoryQueryExecutor:
    """Get or create the query executor instance"""
    global _executor
    if _executor is None:
        _executor = InMemoryQueryExecutor(data_dir)
    return _executor