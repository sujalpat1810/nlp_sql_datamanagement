"""
Pytest Configuration and Shared Fixtures.

Provides common fixtures and configuration for all test modules.
"""

import pytest
import asyncio
import tempfile
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import DatabaseManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def temp_database() -> AsyncGenerator[DatabaseManager, None]:
    """Create a temporary test database with sample data."""
    # Create temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    # Initialize database manager
    db_manager = DatabaseManager(f"sqlite:///{db_path}")
    await db_manager.initialize()
    
    # Create basic test schema
    await db_manager.execute("""
        CREATE TABLE IF NOT EXISTS test_users (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE,
            age INTEGER,
            salary DECIMAL(10,2),
            department VARCHAR(50),
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    await db_manager.execute("""
        CREATE TABLE IF NOT EXISTS test_departments (
            id INTEGER PRIMARY KEY,
            name VARCHAR(50) NOT NULL UNIQUE,
            budget DECIMAL(12,2),
            manager_id INTEGER,
            FOREIGN KEY (manager_id) REFERENCES test_users (id)
        )
    """)
    
    # Insert sample data
    await db_manager.execute("""
        INSERT INTO test_departments (id, name, budget) VALUES
        (1, 'Engineering', 1000000.00),
        (2, 'Marketing', 500000.00),
        (3, 'Sales', 750000.00),
        (4, 'HR', 300000.00)
    """)
    
    await db_manager.execute("""
        INSERT INTO test_users (name, email, age, salary, department, is_active) VALUES
        ('Alice Johnson', 'alice@company.com', 28, 75000.00, 'Engineering', 1),
        ('Bob Smith', 'bob@company.com', 32, 82000.00, 'Engineering', 1),
        ('Carol Davis', 'carol@company.com', 29, 68000.00, 'Marketing', 1),
        ('David Wilson', 'david@company.com', 35, 90000.00, 'Engineering', 1),
        ('Eve Brown', 'eve@company.com', 26, 58000.00, 'Sales', 1),
        ('Frank Miller', 'frank@company.com', 40, 95000.00, 'Engineering', 0),
        ('Grace Taylor', 'grace@company.com', 24, 52000.00, 'HR', 1)
    """)
    
    yield db_manager
    
    # Cleanup
    await db_manager.close()
    try:
        Path(db_path).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_claude_client() -> Mock:
    """Create a mock Claude API client."""
    client = Mock()
    
    # Mock successful SQL generation
    client.generate_sql = AsyncMock(return_value={
        "sql": "SELECT COUNT(*) as total FROM test_users WHERE is_active = 1",
        "confidence": 0.95,
        "explanation": "Counting active users",
        "parameters": []
    })
    
    # Mock result explanation
    client.explain_result = AsyncMock(return_value={
        "explanation": "The query returned the total count of active users.",
        "insights": ["7 users are currently active", "Good user retention"]
    })
    
    # Mock suggestions
    client.get_suggestions = AsyncMock(return_value={
        "suggestions": [
            "Try: Show users by department",
            "Try: Average salary by department",
            "Try: Users hired in the last year"
        ]
    })
    
    return client


@pytest.fixture
def sample_csv_data() -> str:
    """Sample CSV data for testing uploads."""
    return """name,age,salary,department,email
John Doe,30,50000,Engineering,john@test.com
Jane Smith,28,48000,Marketing,jane@test.com
Bob Johnson,35,65000,Engineering,bob@test.com
Alice Brown,26,45000,Sales,alice@test.com"""


@pytest.fixture
def sample_json_data() -> str:
    """Sample JSON data for testing."""
    return """[
    {"name": "Product A", "price": 99.99, "category": "Electronics"},
    {"name": "Product B", "price": 149.99, "category": "Electronics"},
    {"name": "Product C", "price": 29.99, "category": "Books"},
    {"name": "Product D", "price": 79.99, "category": "Clothing"}
]"""


@pytest.fixture
def temp_csv_file(sample_csv_data) -> Generator[str, None, None]:
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        tmp_file.write(sample_csv_data)
        tmp_file_path = tmp_file.name
    
    yield tmp_file_path
    
    # Cleanup
    try:
        Path(tmp_file_path).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def environment_variables():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        'CLAUDE_API_KEY': 'test_api_key_12345',
        'DATABASE_URL': 'sqlite:///test.db',
        'MCP_SERVER_PORT': '9000',
        'LOG_LEVEL': 'DEBUG',
        'TESTING': 'true'
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    from mcp_client.session_manager import SessionManager, UserSession
    
    manager = SessionManager()
    
    # Add a test session
    test_session = UserSession("test_session_123")
    test_session.add_query(
        nl_query="Show all users",
        sql_query="SELECT * FROM test_users",
        success=True,
        data=[{"id": 1, "name": "Test User"}],
        execution_time=0.05
    )
    manager.sessions["test_session_123"] = test_session
    
    return manager


@pytest.fixture
def mock_intent_classifier():
    """Create a mock intent classifier."""
    from nlp.intent_classifier import IntentClassifier, IntentResult, QueryIntent
    
    classifier = Mock(spec=IntentClassifier)
    
    # Default classification result
    classifier.classify_intent.return_value = IntentResult(
        intent=QueryIntent.SELECT_ALL,
        confidence=0.85,
        suggested_sql_template="SELECT * FROM {table}",
        required_entities=["table"]
    )
    
    classifier.get_intent_suggestions.return_value = [
        IntentResult(QueryIntent.SELECT_ALL, 0.85, "SELECT * FROM {table}", ["table"]),
        IntentResult(QueryIntent.COUNT_RECORDS, 0.70, "SELECT COUNT(*) FROM {table}", ["table"])
    ]
    
    classifier.analyze_query_complexity.return_value = {
        "complexity_level": "simple",
        "complexity_score": 0.2,
        "indicators": ["single table", "no joins", "basic selection"]
    }
    
    return classifier


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    import logging
    
    # Set up basic logging configuration
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress some noisy loggers
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


@pytest.fixture
def performance_data():
    """Generate performance test data."""
    return {
        'users': [
            {
                'id': i,
                'name': f'User_{i:04d}',
                'email': f'user{i:04d}@test.com',
                'age': 20 + (i % 50),
                'salary': 40000 + (i * 100),
                'department': ['Engineering', 'Marketing', 'Sales', 'HR'][i % 4]
            }
            for i in range(1, 1001)  # 1000 users
        ],
        'orders': [
            {
                'id': i,
                'user_id': (i % 1000) + 1,
                'amount': 10.00 + (i * 0.50),
                'status': ['pending', 'completed', 'cancelled'][i % 3],
                'created_at': f'2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}'
            }
            for i in range(1, 5001)  # 5000 orders
        ]
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "database: marks tests that require database")
    config.addinivalue_line("markers", "api: marks tests that require API access")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add markers based on file paths
        if "test_integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        elif "test_mcp_tools" in item.fspath.basename or "test_database" in item.fspath.basename:
            item.add_marker(pytest.mark.database)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ['performance', 'large', 'concurrent']):
            item.add_marker(pytest.mark.slow)