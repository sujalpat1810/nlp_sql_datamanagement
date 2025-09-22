"""
Database Configuration.

Provides database-specific configuration and connection settings.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from .settings import get_settings


class DatabaseConfig:
    """Database configuration manager."""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
    
    def get_connection_config(self, for_testing: bool = False) -> Dict[str, Any]:
        """Get database connection configuration."""
        if for_testing:
            return {
                "url": self.settings.test_database_url,
                "echo": False,
                "pool_pre_ping": True,
                "pool_recycle": 300
            }
        
        url = self.settings.database_url
        config = {
            "url": url,
            "echo": self.settings.database_echo,
            "pool_pre_ping": True,
            "pool_recycle": 3600
        }
        
        # Add connection pool settings for non-SQLite databases
        if not url.startswith("sqlite"):
            config.update({
                "pool_size": self.settings.database_pool_size,
                "max_overflow": self.settings.database_max_overflow,
                "pool_timeout": 30,
                "pool_reset_on_return": "commit"
            })
        
        return config
    
    def get_migration_config(self) -> Dict[str, Any]:
        """Get Alembic migration configuration."""
        return {
            "script_location": "database/migrations",
            "sqlalchemy.url": self.settings.database_url,
            "version_table": "alembic_version",
            "version_table_schema": None,
            "compare_type": True,
            "compare_server_default": True,
            "render_as_batch": True  # For SQLite compatibility
        }
    
    def ensure_data_directory(self) -> None:
        """Ensure the data directory exists for file-based databases."""
        if self.settings.database_url.startswith("sqlite"):
            # Extract directory from SQLite URL
            if ":///" in self.settings.database_url:
                db_path = self.settings.database_url.split(":///")[1]
                if db_path != ":memory:":
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def get_backup_settings(self) -> Dict[str, Any]:
        """Get database backup configuration."""
        return {
            "backup_dir": Path(self.settings.backup_dir),
            "max_backups": 10,
            "backup_prefix": "nlp_sql_backup",
            "compress": True,
            "include_data": True
        }
    
    def get_test_data_config(self) -> Dict[str, Any]:
        """Get test data configuration."""
        return {
            "sample_users": 100,
            "sample_orders": 500,
            "sample_products": 50,
            "sample_departments": 5,
            "seed": 42  # For reproducible test data
        }
    
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.settings.database_url.startswith("sqlite")
    
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL database."""
        return "postgresql" in self.settings.database_url
    
    def is_in_memory(self) -> bool:
        """Check if using in-memory database."""
        return ":memory:" in self.settings.database_url