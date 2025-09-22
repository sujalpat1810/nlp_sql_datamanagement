"""
Database connection management for NLP to SQL system.

Provides database connectivity, session management, and connection pooling
for SQLite and PostgreSQL databases using SQLAlchemy.
"""

import os
import asyncio
from typing import Optional, Any, Dict, List, Tuple
from contextlib import asynccontextmanager
import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            database_url: Database connection URL. If None, uses environment variable.
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'sqlite+aiosqlite:///data/nlp_sql.db'
        )
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        if self._initialized:
            return
            
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=os.getenv('DATABASE_ECHO', 'false').lower() == 'true',
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Import models to ensure they're registered
            from .models import Base
            
            # Create all tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._initialized = True
            logger.info(f"Database initialized: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_factory = None
            self._initialized = False
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session context manager."""
        if not self._initialized:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        fetch_all: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            fetch_all: Whether to fetch all results or just the first
            
        Returns:
            Query results as list of dictionaries
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), parameters or {})
                
                if fetch_all:
                    rows = result.fetchall()
                else:
                    rows = [result.fetchone()] if result.fetchone() else []
                
                # Convert to list of dictionaries
                if rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in rows]
                return []
                
        except SQLAlchemyError as e:
            logger.error(f"Database query error: {e}")
            raise
    
    async def execute_non_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Execute a non-SELECT query (INSERT, UPDATE, DELETE).
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            Number of affected rows
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), parameters or {})
                return result.rowcount
                
        except SQLAlchemyError as e:
            logger.error(f"Database execution error: {e}")
            raise
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a database table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table information including columns, types, and constraints
        """
        try:
            async with self.get_session() as session:
                # Get column information
                if 'sqlite' in self.database_url:
                    query = f"PRAGMA table_info({table_name})"
                else:
                    query = f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                    """
                
                result = await session.execute(text(query))
                columns = result.fetchall()
                
                return {
                    'table_name': table_name,
                    'columns': [dict(zip(result.keys(), col)) for col in columns],
                    'exists': len(columns) > 0
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            raise
    
    async def get_table_names(self) -> List[str]:
        """Get list of all table names in the database."""
        try:
            async with self.get_session() as session:
                if 'sqlite' in self.database_url:
                    query = "SELECT name FROM sqlite_master WHERE type='table'"
                else:
                    query = """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    """
                
                result = await session.execute(text(query))
                rows = result.fetchall()
                return [row[0] for row in rows]
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting table names: {e}")
            raise
    
    async def validate_connection(self) -> bool:
        """Validate database connection."""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection validation failed: {e}")
            return False


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager