"""
Database layer for NLP to SQL Data Management System.

This module provides database connectivity, ORM models, and data management utilities
using SQLAlchemy with support for SQLite and PostgreSQL databases.
"""

# Don't import by default to avoid dependency issues
__all__ = [
    "DatabaseManager",
    "get_database_manager", 
    "Base",
    "DataTable",
    "QueryLog",
    "UserSession"
]