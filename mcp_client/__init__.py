"""
MCP Client for NLP to SQL System.

This module provides the client-side implementation that communicates with 
the MCP server and Claude API for natural language query processing.
"""

from .client import NLPSQLClient, create_client, quick_query
from .claude_interface import ClaudeInterface, NLQueryRequest, SQLQueryResponse
from .session_manager import SessionManager, QueryHistoryEntry, SessionContext

__all__ = [
    "NLPSQLClient",
    "ClaudeInterface", 
    "SessionManager",
    "NLQueryRequest",
    "SQLQueryResponse",
    "QueryHistoryEntry",
    "SessionContext",
    "create_client",
    "quick_query"
]