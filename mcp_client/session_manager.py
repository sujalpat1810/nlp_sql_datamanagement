"""
Session Manager for NLP to SQL Client.

This module manages user sessions, query history, and conversation context
for the natural language to SQL conversion system.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class QueryHistoryEntry:
    """Single entry in query history."""
    id: str
    timestamp: datetime
    natural_language_query: str
    generated_sql: str
    execution_result: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class SessionContext(BaseModel):
    """Session context information."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime
    last_activity: datetime
    database_schema: Optional[Dict[str, Any]] = None
    active_tables: Set[str] = set()
    query_count: int = 0
    successful_queries: int = 0
    preferences: Dict[str, Any] = {}


class SessionManager:
    """Manages user sessions and query history."""
    
    def __init__(self, session_storage_dir: Optional[str] = None):
        """
        Initialize session manager.
        
        Args:
            session_storage_dir: Directory to store session data (None for memory-only)
        """
        self.sessions: Dict[str, SessionContext] = {}
        self.query_history: Dict[str, List[QueryHistoryEntry]] = {}
        self.storage_dir = Path(session_storage_dir) if session_storage_dir else None
        
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_sessions()
    
    async def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        session = SessionContext(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now
        )
        
        self.sessions[session_id] = session
        self.query_history[session_id] = []
        
        if self.storage_dir:
            await self._save_session(session_id)
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session context or None if not found
        """
        session = self.sessions.get(session_id)
        if session:
            # Update last activity
            session.last_activity = datetime.utcnow()
            if self.storage_dir:
                await self._save_session(session_id)
        
        return session
    
    async def update_session_schema(
        self, 
        session_id: str, 
        schema_info: Dict[str, Any]
    ) -> bool:
        """
        Update database schema information for a session.
        
        Args:
            session_id: Session identifier
            schema_info: Database schema information
            
        Returns:
            True if successful, False if session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.database_schema = schema_info
        session.last_activity = datetime.utcnow()
        
        # Extract table names for quick reference
        if "tables" in schema_info:
            session.active_tables = set(schema_info["tables"].keys())
        
        if self.storage_dir:
            await self._save_session(session_id)
        
        logger.info(f"Updated schema for session {session_id}")
        return True
    
    async def add_query_to_history(
        self,
        session_id: str,
        natural_language_query: str,
        generated_sql: str,
        execution_result: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[float] = None
    ) -> str:
        """
        Add a query to session history.
        
        Args:
            session_id: Session identifier
            natural_language_query: Original natural language query
            generated_sql: Generated SQL query
            execution_result: Query execution results
            success: Whether query executed successfully
            error_message: Error message if failed
            execution_time_ms: Query execution time
            
        Returns:
            Query entry ID
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return ""
        
        # Create query history entry
        entry_id = str(uuid.uuid4())
        entry = QueryHistoryEntry(
            id=entry_id,
            timestamp=datetime.utcnow(),
            natural_language_query=natural_language_query,
            generated_sql=generated_sql,
            execution_result=execution_result,
            success=success,
            error_message=error_message,
            execution_time_ms=execution_time_ms
        )
        
        # Add to session history
        if session_id not in self.query_history:
            self.query_history[session_id] = []
        
        self.query_history[session_id].append(entry)
        
        # Update session statistics
        session.query_count += 1
        if success:
            session.successful_queries += 1
        session.last_activity = datetime.utcnow()
        
        # Limit history size (keep last 100 queries)
        if len(self.query_history[session_id]) > 100:
            self.query_history[session_id] = self.query_history[session_id][-100:]
        
        if self.storage_dir:
            await self._save_session(session_id)
            await self._save_query_history(session_id)
        
        logger.info(f"Added query to history for session {session_id}: {entry_id}")
        return entry_id
    
    async def get_query_history(
        self, 
        session_id: str, 
        limit: int = 20,
        include_failed: bool = True
    ) -> List[QueryHistoryEntry]:
        """
        Get query history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of entries to return
            include_failed: Whether to include failed queries
            
        Returns:
            List of query history entries
        """
        history = self.query_history.get(session_id, [])
        
        if not include_failed:
            history = [entry for entry in history if entry.success]
        
        # Return most recent entries first
        return sorted(history, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    async def get_recent_successful_queries(
        self, 
        session_id: str, 
        limit: int = 5
    ) -> List[str]:
        """
        Get recent successful SQL queries for context.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of queries to return
            
        Returns:
            List of SQL queries
        """
        history = await self.get_query_history(session_id, limit * 2, include_failed=False)
        return [entry.generated_sql for entry in history[:limit]]
    
    async def get_session_statistics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session statistics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session statistics or None if session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        history = self.query_history.get(session_id, [])
        
        # Calculate success rate
        success_rate = 0.0
        if session.query_count > 0:
            success_rate = (session.successful_queries / session.query_count) * 100
        
        # Calculate average execution time
        execution_times = [
            entry.execution_time_ms 
            for entry in history 
            if entry.execution_time_ms is not None and entry.success
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Get query types distribution
        query_types = {}
        for entry in history:
            sql_upper = entry.generated_sql.upper().strip()
            query_type = "UNKNOWN"
            
            if sql_upper.startswith('SELECT'):
                query_type = 'SELECT'
            elif sql_upper.startswith('INSERT'):
                query_type = 'INSERT'
            elif sql_upper.startswith('UPDATE'):
                query_type = 'UPDATE'
            elif sql_upper.startswith('DELETE'):
                query_type = 'DELETE'
            
            query_types[query_type] = query_types.get(query_type, 0) + 1
        
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "query_count": session.query_count,
            "successful_queries": session.successful_queries,
            "success_rate": round(success_rate, 2),
            "avg_execution_time_ms": round(avg_execution_time, 2),
            "active_tables": list(session.active_tables),
            "query_types": query_types
        }
    
    async def update_session_preferences(
        self, 
        session_id: str, 
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Update session preferences.
        
        Args:
            session_id: Session identifier
            preferences: User preferences
            
        Returns:
            True if successful, False if session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.preferences.update(preferences)
        session.last_activity = datetime.utcnow()
        
        if self.storage_dir:
            await self._save_session(session_id)
        
        logger.info(f"Updated preferences for session {session_id}")
        return True
    
    async def cleanup_expired_sessions(self, expiry_hours: int = 24) -> int:
        """
        Clean up expired sessions.
        
        Args:
            expiry_hours: Hours after which a session is considered expired
            
        Returns:
            Number of sessions cleaned up
        """
        expiry_time = datetime.utcnow() - timedelta(hours=expiry_hours)
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.last_activity < expiry_time:
                expired_sessions.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_sessions:
            del self.sessions[session_id]
            if session_id in self.query_history:
                del self.query_history[session_id]
            
            # Remove from disk storage
            if self.storage_dir:
                session_file = self.storage_dir / f"session_{session_id}.json"
                history_file = self.storage_dir / f"history_{session_id}.json"
                
                if session_file.exists():
                    session_file.unlink()
                if history_file.exists():
                    history_file.unlink()
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    async def export_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Export all session data for backup or analysis.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Complete session data or None if session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        history = self.query_history.get(session_id, [])
        
        return {
            "session": asdict(session),
            "query_history": [asdict(entry) for entry in history],
            "export_timestamp": datetime.utcnow().isoformat()
        }
    
    # Private methods for persistence
    
    async def _save_session(self, session_id: str) -> None:
        """Save session to disk."""
        if not self.storage_dir:
            return
        
        session = self.sessions.get(session_id)
        if not session:
            return
        
        session_file = self.storage_dir / f"session_{session_id}.json"
        
        try:
            with open(session_file, 'w') as f:
                # Convert session to dict for JSON serialization
                session_dict = {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "database_schema": session.database_schema,
                    "active_tables": list(session.active_tables),
                    "query_count": session.query_count,
                    "successful_queries": session.successful_queries,
                    "preferences": session.preferences
                }
                json.dump(session_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    async def _save_query_history(self, session_id: str) -> None:
        """Save query history to disk."""
        if not self.storage_dir:
            return
        
        history = self.query_history.get(session_id, [])
        history_file = self.storage_dir / f"history_{session_id}.json"
        
        try:
            with open(history_file, 'w') as f:
                # Convert history entries to dicts
                history_dicts = []
                for entry in history:
                    entry_dict = asdict(entry)
                    entry_dict['timestamp'] = entry.timestamp.isoformat()
                    history_dicts.append(entry_dict)
                
                json.dump(history_dicts, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save query history for session {session_id}: {e}")
    
    def _load_sessions(self) -> None:
        """Load sessions from disk."""
        if not self.storage_dir or not self.storage_dir.exists():
            return
        
        # Load session files
        for session_file in self.storage_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_dict = json.load(f)
                
                # Convert back to SessionContext
                session = SessionContext(
                    session_id=session_dict["session_id"],
                    user_id=session_dict.get("user_id"),
                    created_at=datetime.fromisoformat(session_dict["created_at"]),
                    last_activity=datetime.fromisoformat(session_dict["last_activity"]),
                    database_schema=session_dict.get("database_schema"),
                    active_tables=set(session_dict.get("active_tables", [])),
                    query_count=session_dict.get("query_count", 0),
                    successful_queries=session_dict.get("successful_queries", 0),
                    preferences=session_dict.get("preferences", {})
                )
                
                self.sessions[session.session_id] = session
                
            except Exception as e:
                logger.error(f"Failed to load session from {session_file}: {e}")
        
        # Load history files
        for history_file in self.storage_dir.glob("history_*.json"):
            try:
                session_id = history_file.stem.replace("history_", "")
                
                with open(history_file, 'r') as f:
                    history_dicts = json.load(f)
                
                # Convert back to QueryHistoryEntry objects
                history_entries = []
                for entry_dict in history_dicts:
                    entry = QueryHistoryEntry(
                        id=entry_dict["id"],
                        timestamp=datetime.fromisoformat(entry_dict["timestamp"]),
                        natural_language_query=entry_dict["natural_language_query"],
                        generated_sql=entry_dict["generated_sql"],
                        execution_result=entry_dict.get("execution_result"),
                        success=entry_dict["success"],
                        error_message=entry_dict.get("error_message"),
                        execution_time_ms=entry_dict.get("execution_time_ms")
                    )
                    history_entries.append(entry)
                
                self.query_history[session_id] = history_entries
                
            except Exception as e:
                logger.error(f"Failed to load query history from {history_file}: {e}")
        
        logger.info(f"Loaded {len(self.sessions)} sessions from disk")