"""
Application Settings Configuration.

Loads and validates configuration from environment variables and .env files.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    claude_api_key: Optional[str] = Field(default=None, env="CLAUDE_API_KEY")
    claude_model: str = Field(default="claude-3-sonnet-20241022", env="CLAUDE_MODEL")
    claude_max_tokens: int = Field(default=4000, env="CLAUDE_MAX_TOKENS")
    claude_temperature: float = Field(default=0.1, env="CLAUDE_TEMPERATURE")
    
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./data/nlp_sql.db", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    database_pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")
    
    # PostgreSQL Configuration
    postgres_user: str = Field(default="nlp_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="secure_password", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="nlp_sql_db", env="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    
    # MCP Server Configuration
    mcp_server_host: str = Field(default="localhost", env="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=3000, env="MCP_SERVER_PORT")
    mcp_server_name: str = Field(default="nlp-sql-server", env="MCP_SERVER_NAME")
    mcp_server_version: str = Field(default="1.0.0", env="MCP_SERVER_VERSION")
    mcp_server_secret_key: str = Field(default="dev-secret-key", env="MCP_SERVER_SECRET_KEY")
    mcp_allowed_origins: str = Field(default="http://localhost:3000,http://127.0.0.1:3000", env="MCP_ALLOWED_ORIGINS")
    
    # Application Configuration
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/nlp_sql.log", env="LOG_FILE")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    # Session Management
    session_timeout: int = Field(default=3600, env="SESSION_TIMEOUT")
    max_sessions: int = Field(default=100, env="MAX_SESSIONS")
    session_cleanup_interval: int = Field(default=300, env="SESSION_CLEANUP_INTERVAL")
    
    # NLP Configuration
    intent_confidence_threshold: float = Field(default=0.7, env="INTENT_CONFIDENCE_THRESHOLD")
    max_intent_suggestions: int = Field(default=5, env="MAX_INTENT_SUGGESTIONS")
    enable_ml_intent_classification: bool = Field(default=True, env="ENABLE_ML_INTENT_CLASSIFICATION")
    
    # SQL Configuration
    sql_safety_checks: bool = Field(default=True, env="SQL_SAFETY_CHECKS")
    max_query_complexity: str = Field(default="moderate", env="MAX_QUERY_COMPLEXITY")
    enable_query_optimization: bool = Field(default=True, env="ENABLE_QUERY_OPTIMIZATION")
    max_result_rows: int = Field(default=1000, env="MAX_RESULT_ROWS")
    
    # Security Configuration
    enable_sql_injection_protection: bool = Field(default=True, env="ENABLE_SQL_INJECTION_PROTECTION")
    allowed_sql_operations: str = Field(default="SELECT,INSERT,UPDATE,DELETE", env="ALLOWED_SQL_OPERATIONS")
    blocked_sql_keywords: str = Field(default="DROP,TRUNCATE,ALTER,CREATE_USER,GRANT,REVOKE", env="BLOCKED_SQL_KEYWORDS")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # Performance Configuration
    enable_query_cache: bool = Field(default=True, env="ENABLE_QUERY_CACHE")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")
    max_cache_size: int = Field(default=1000, env="MAX_CACHE_SIZE")
    max_concurrent_queries: int = Field(default=10, env="MAX_CONCURRENT_QUERIES")
    query_timeout: int = Field(default=30, env="QUERY_TIMEOUT")
    
    # Testing Configuration
    mock_claude_responses: bool = Field(default=False, env="MOCK_CLAUDE_RESPONSES")
    mock_database_operations: bool = Field(default=False, env="MOCK_DATABASE_OPERATIONS")
    test_database_url: str = Field(default="sqlite:///:memory:", env="TEST_DATABASE_URL")
    test_log_level: str = Field(default="WARNING", env="TEST_LOG_LEVEL")
    
    # Sample Data Configuration
    load_sample_data: bool = Field(default=True, env="LOAD_SAMPLE_DATA")
    sample_data_path: str = Field(default="./data/sample_data.json", env="SAMPLE_DATA_PATH")
    
    # File Paths
    data_dir: str = Field(default="./data", env="DATA_DIR")
    logs_dir: str = Field(default="./logs", env="LOGS_DIR")
    backup_dir: str = Field(default="./backups", env="BACKUP_DIR")
    temp_dir: str = Field(default="./temp", env="TEMP_DIR")
    config_dir: str = Field(default="./config", env="CONFIG_DIR")
    schema_dir: str = Field(default="./config/schemas", env="SCHEMA_DIR")
    templates_dir: str = Field(default="./config/templates", env="TEMPLATES_DIR")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env file
        
    def validate_api_keys(self) -> bool:
        """Check if at least one API key is configured."""
        return bool(self.claude_api_key or self.openai_api_key)
    
    def get_database_url(self, for_testing: bool = False) -> str:
        """Get the appropriate database URL."""
        if for_testing or self.testing:
            return self.test_database_url
        return self.database_url
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.logs_dir,
            self.backup_dir,
            self.temp_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"
    
    def get_allowed_sql_operations(self) -> List[str]:
        """Get allowed SQL operations as a list."""
        return [op.strip().upper() for op in self.allowed_sql_operations.split(",")]
    
    def get_blocked_sql_keywords(self) -> List[str]:
        """Get blocked SQL keywords as a list."""
        return [keyword.strip().upper() for keyword in self.blocked_sql_keywords.split(",")]


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.create_directories()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global _settings
    _settings = None
    return get_settings()