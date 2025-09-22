"""
SQLAlchemy models for NLP to SQL Data Management System.

Defines database models for storing data tables metadata, query logs,
user sessions, and other application data.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Text, Integer, DateTime, Boolean, JSON, 
    ForeignKey, Index, UniqueConstraint
)


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class DataTable(Base):
    """Model for storing information about data tables."""
    
    __tablename__ = 'data_tables'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    table_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    schema_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    row_count: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    query_logs: Mapped[List["QueryLog"]] = relationship(
        "QueryLog", back_populates="data_table"
    )
    
    def __repr__(self) -> str:
        return f"<DataTable(name='{self.table_name}', rows={self.row_count})>"


class QueryLog(Base):
    """Model for logging executed queries."""
    
    __tablename__ = 'query_logs'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), index=True)
    natural_query: Mapped[Optional[str]] = mapped_column(Text)
    sql_query: Mapped[str] = mapped_column(Text)
    query_type: Mapped[str] = mapped_column(String(50))  # SELECT, INSERT, UPDATE, DELETE
    parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    rows_affected: Mapped[Optional[int]] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Foreign key to data table
    data_table_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey('data_tables.id')
    )
    
    # Relationships
    data_table: Mapped[Optional["DataTable"]] = relationship(
        "DataTable", back_populates="query_logs"
    )
    user_session: Mapped[Optional["UserSession"]] = relationship(
        "UserSession", back_populates="query_logs"
    )
    
    __table_args__ = (
        Index('idx_query_logs_session_created', 'session_id', 'created_at'),
        Index('idx_query_logs_type_success', 'query_type', 'success'),
    )
    
    def __repr__(self) -> str:
        return f"<QueryLog(type='{self.query_type}', success={self.success})>"


class UserSession(Base):
    """Model for tracking user sessions."""
    
    __tablename__ = 'user_sessions'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    user_identifier: Mapped[Optional[str]] = mapped_column(String(255))
    context_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    preferences: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_activity: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    query_logs: Mapped[List["QueryLog"]] = relationship(
        "QueryLog", back_populates="user_session"
    )
    
    def __repr__(self) -> str:
        return f"<UserSession(id='{self.session_id}', active={self.is_active})>"


class DataView(Base):
    """Model for storing custom data views."""
    
    __tablename__ = 'data_views'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    view_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    sql_definition: Mapped[str] = mapped_column(Text)
    source_tables: Mapped[List[str]] = mapped_column(JSON)
    column_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_by: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    def __repr__(self) -> str:
        return f"<DataView(name='{self.view_name}', tables={self.source_tables})>"


class DataIndex(Base):
    """Model for tracking database indexes."""
    
    __tablename__ = 'data_indexes'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    table_name: Mapped[str] = mapped_column(String(255), index=True)
    index_name: Mapped[str] = mapped_column(String(255))
    columns: Mapped[List[str]] = mapped_column(JSON)
    is_unique: Mapped[bool] = mapped_column(Boolean, default=False)
    index_type: Mapped[str] = mapped_column(String(50))  # btree, hash, etc.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('table_name', 'index_name', name='uq_table_index'),
    )
    
    def __repr__(self) -> str:
        return f"<DataIndex(table='{self.table_name}', name='{self.index_name}')>"


class DataQualityReport(Base):
    """Model for storing data quality assessment reports."""
    
    __tablename__ = 'data_quality_reports'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    table_name: Mapped[str] = mapped_column(String(255), index=True)
    report_type: Mapped[str] = mapped_column(String(100))  # completeness, consistency, etc.
    quality_score: Mapped[Optional[float]] = mapped_column()
    issues_found: Mapped[int] = mapped_column(Integer, default=0)
    report_data: Mapped[Dict[str, Any]] = mapped_column(JSON)
    recommendations: Mapped[Optional[List[str]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_quality_reports_table_type', 'table_name', 'report_type'),
    )
    
    def __repr__(self) -> str:
        return f"<DataQualityReport(table='{self.table_name}', score={self.quality_score})>"