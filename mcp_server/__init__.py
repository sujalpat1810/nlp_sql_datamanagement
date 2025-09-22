"""
MCP Server for NLP to SQL Data Management System.

This module implements a Model Context Protocol server that provides tools for:
- Data ingestion from CSV and JSON files
- SQL query execution and validation
- Data analysis and statistical operations
- Advanced database operations like joins and aggregations
"""

__version__ = "0.1.0"
__author__ = "Sujal P"

from .main import main, create_server
from .tools import *

__all__ = ["main", "create_server"]