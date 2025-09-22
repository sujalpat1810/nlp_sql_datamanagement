"""
Quick Test Script for NLP to SQL System

This script provides a simple way to test the core functionality
of the NLP to SQL system without needing the full MCP server setup.

Usage:
    python examples/quick_test.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_client.claude_interface import ClaudeInterface, NLQueryRequest
from database.connection import DatabaseManager
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_database_setup():
    """Test basic database setup and operations."""
    print("🔧 Testing Database Setup...")
    
    try:
        # Test database connection
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Create a simple test table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS test_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            age INTEGER,
            department TEXT,
            salary REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        await db_manager.execute_query(create_table_sql)
        print("✅ Database connection and table creation successful")
        
        # Insert sample data
        sample_users = [
            ("Alice Johnson", "alice@company.com", 28, "Engineering", 75000.00),
            ("Bob Smith", "bob@company.com", 34, "Marketing", 65000.00),
            ("Carol Davis", "carol@company.com", 29, "Engineering", 80000.00),
            ("David Wilson", "david@company.com", 42, "Sales", 70000.00),
            ("Eve Brown", "eve@company.com", 26, "Engineering", 72000.00),
        ]
        
        for user in sample_users:
            try:
                insert_sql = """
                INSERT OR IGNORE INTO test_users (name, email, age, department, salary) 
                VALUES (?, ?, ?, ?, ?)
                """
                await db_manager.execute_query(insert_sql, user)
            except Exception as e:
                # User might already exist, that's okay
                pass
        
        # Test a simple query
        result = await db_manager.execute_query("SELECT COUNT(*) as count FROM test_users")
        user_count = result[0]['count'] if result else 0
        print(f"✅ Sample data loaded: {user_count} users in test table")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


async def test_claude_interface():
    """Test Claude API interface (requires API key)."""
    print("\n🤖 Testing Claude Interface...")
    
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if not claude_api_key:
        print("⚠️  CLAUDE_API_KEY not set. Skipping Claude API tests.")
        print("   To test Claude integration, set: export CLAUDE_API_KEY='your-key'")
        return False
    
    try:
        claude = ClaudeInterface(claude_api_key)
        
        # Test natural language query processing
        test_request = NLQueryRequest(
            query="Show me all users in the Engineering department",
            schema_info={
                "tables": {
                    "test_users": {
                        "columns": ["id", "name", "email", "age", "department", "salary"],
                        "types": ["INTEGER", "TEXT", "TEXT", "INTEGER", "TEXT", "REAL"]
                    }
                }
            }
        )
        
        response = await claude.process_natural_language_query(test_request)
        
        print(f"✅ Claude API connection successful")
        print(f"   Generated SQL: {response.sql_query}")
        print(f"   Confidence: {response.confidence_score}")
        print(f"   Query Type: {response.query_type}")
        
        if response.warnings:
            print(f"   Warnings: {response.warnings}")
        
        return True
        
    except Exception as e:
        print(f"❌ Claude API test failed: {e}")
        return False


async def test_intent_classification():
    """Test intent classification without Claude API."""
    print("\n🧠 Testing Intent Classification...")
    
    try:
        from nlp.intent_classifier import IntentClassifier
        
        classifier = IntentClassifier()
        
        # Test various query types
        test_queries = [
            "Show me all users",
            "Find users where age > 30",
            "Count how many users are in Engineering",
            "Update user salary to 85000 where name is Alice",
            "Delete users where department is Sales",
            "What is the average salary by department?",
            "Create a new user record"
        ]
        
        print("Intent Classification Results:")
        for query in test_queries:
            result = classifier.classify_intent(query)
            print(f"  '{query}'")
            print(f"    → Intent: {result.intent.value}")
            print(f"    → Confidence: {result.confidence:.2f}")
            print(f"    → Template: {result.suggested_sql_template}")
            print()
        
        print("✅ Intent classification working")
        return True
        
    except Exception as e:
        print(f"❌ Intent classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_end_to_end_simulation():
    """Simulate end-to-end processing without MCP server."""
    print("\n🔄 Testing End-to-End Simulation...")
    
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if not claude_api_key:
        print("⚠️  Skipping end-to-end test (no Claude API key)")
        return False
    
    try:
        # Initialize components
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        claude = ClaudeInterface(claude_api_key)
        
        # Get table schema
        table_info = await db_manager.get_table_info("test_users")
        
        # Test query: "Show me all engineers earning more than 70000"
        natural_query = "Show me all engineers earning more than 70000"
        
        request = NLQueryRequest(
            query=natural_query,
            schema_info={
                "tables": {
                    "test_users": table_info
                }
            }
        )
        
        # Generate SQL with Claude
        sql_response = await claude.process_natural_language_query(request)
        print(f"Natural Language: {natural_query}")
        print(f"Generated SQL: {sql_response.sql_query}")
        
        # Execute the generated SQL
        if sql_response.sql_query and not sql_response.sql_query.startswith("-- Error"):
            results = await db_manager.execute_query(sql_response.sql_query)
            print(f"Query Results: {len(results) if results else 0} rows")
            
            if results:
                print("Sample Results:")
                for i, row in enumerate(results[:3]):  # Show first 3 rows
                    print(f"  Row {i+1}: {dict(row)}")
            
            print("✅ End-to-end simulation successful")
            
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ End-to-end simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_test_summary(results):
    """Print test summary."""
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Check configuration and dependencies.")


async def main():
    """Run all tests."""
    print("🧪 NLP to SQL System - Quick Test Suite")
    print("="*60)
    
    # Check dependencies
    print("📋 Checking Dependencies...")
    missing_deps = []
    
    try:
        import anthropic
        print("✅ anthropic (Claude API)")
    except ImportError:
        missing_deps.append("anthropic")
        print("❌ anthropic (Claude API) - run: pip install anthropic")
    
    try:
        import spacy
        print("✅ spacy (NLP)")
    except ImportError:
        missing_deps.append("spacy")
        print("❌ spacy (NLP) - run: pip install spacy")
    
    try:
        import sklearn
        print("✅ scikit-learn (ML)")
    except ImportError:
        missing_deps.append("scikit-learn")
        print("❌ scikit-learn (ML) - run: pip install scikit-learn")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
    
    print()
    
    # Run tests
    test_results = {}
    
    test_results["Database Setup"] = await test_database_setup()
    test_results["Claude Interface"] = await test_claude_interface()
    test_results["Intent Classification"] = await test_intent_classification()
    test_results["End-to-End Simulation"] = await test_end_to_end_simulation()
    
    print_test_summary(test_results)


if __name__ == "__main__":
    asyncio.run(main())