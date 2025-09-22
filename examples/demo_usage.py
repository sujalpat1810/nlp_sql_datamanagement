"""
Demo Usage Script for NLP to SQL System

This script demonstrates how to use the complete NLP to SQL system
with real examples and sample data.

Usage:
    1. Set your Claude API key: export CLAUDE_API_KEY='your-key'
    2. Run: python examples/demo_usage.py
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

from mcp_client import create_client, NLPSQLClient
from database.connection import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_demo_database():
    """Set up database with sample data for demo."""
    print("🏗️  Setting up demo database...")
    
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    # Create employees table
    await db_manager.execute_query("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            age INTEGER,
            department TEXT,
            position TEXT,
            salary REAL,
            hire_date DATE,
            manager_id INTEGER,
            city TEXT,
            country TEXT DEFAULT 'USA'
        )
    """)
    
    # Create departments table
    await db_manager.execute_query("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            budget REAL,
            location TEXT,
            head_count INTEGER DEFAULT 0
        )
    """)
    
    # Create projects table
    await db_manager.execute_query("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT,
            budget REAL,
            start_date DATE,
            end_date DATE,
            status TEXT DEFAULT 'active'
        )
    """)
    
    # Insert sample departments
    departments = [
        ("Engineering", 2500000.00, "San Francisco", 25),
        ("Marketing", 800000.00, "New York", 12),
        ("Sales", 1200000.00, "Chicago", 18),
        ("HR", 400000.00, "Austin", 8),
        ("Finance", 600000.00, "Boston", 10)
    ]
    
    for dept in departments:
        try:
            await db_manager.execute_query(
                "INSERT OR IGNORE INTO departments (name, budget, location, head_count) VALUES (?, ?, ?, ?)",
                dept
            )
        except:
            pass
    
    # Insert sample employees
    employees = [
        ("Alice Johnson", "alice@company.com", 28, "Engineering", "Senior Developer", 95000.00, "2022-01-15", None, "San Francisco"),
        ("Bob Smith", "bob@company.com", 34, "Marketing", "Marketing Manager", 78000.00, "2021-03-10", None, "New York"),
        ("Carol Davis", "carol@company.com", 29, "Engineering", "DevOps Engineer", 88000.00, "2022-06-01", 1, "San Francisco"),
        ("David Wilson", "david@company.com", 42, "Sales", "Sales Director", 95000.00, "2020-11-20", None, "Chicago"),
        ("Eve Brown", "eve@company.com", 26, "Engineering", "Junior Developer", 72000.00, "2023-02-01", 1, "San Francisco"),
        ("Frank Miller", "frank@company.com", 35, "HR", "HR Manager", 75000.00, "2021-09-15", None, "Austin"),
        ("Grace Lee", "grace@company.com", 31, "Finance", "Financial Analyst", 68000.00, "2022-04-12", None, "Boston"),
        ("Henry Clark", "henry@company.com", 27, "Marketing", "Content Writer", 55000.00, "2023-01-08", 2, "New York"),
        ("Isabel Garcia", "isabel@company.com", 33, "Sales", "Account Manager", 72000.00, "2021-12-03", 4, "Chicago"),
        ("Jack Thompson", "jack@company.com", 29, "Engineering", "Full Stack Developer", 85000.00, "2022-08-22", 1, "San Francisco")
    ]
    
    for emp in employees:
        try:
            await db_manager.execute_query(
                "INSERT OR IGNORE INTO employees (name, email, age, department, position, salary, hire_date, manager_id, city) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                emp
            )
        except:
            pass
    
    # Insert sample projects
    projects = [
        ("Mobile App Redesign", "Engineering", 500000.00, "2024-01-01", "2024-12-31", "active"),
        ("Q4 Marketing Campaign", "Marketing", 150000.00, "2024-10-01", "2024-12-31", "active"),
        ("CRM Integration", "Sales", 200000.00, "2024-03-01", "2024-09-30", "completed"),
        ("Employee Portal", "HR", 100000.00, "2024-02-01", "2024-08-31", "active"),
        ("Budget Analysis Tool", "Finance", 75000.00, "2024-01-15", "2024-06-30", "completed")
    ]
    
    for project in projects:
        try:
            await db_manager.execute_query(
                "INSERT OR IGNORE INTO projects (name, department, budget, start_date, end_date, status) VALUES (?, ?, ?, ?, ?, ?)",
                project
            )
        except:
            pass
    
    await db_manager.close()
    print("✅ Demo database setup complete!")


async def demo_basic_queries(client: NLPSQLClient, session_id: str):
    """Demonstrate basic query functionality."""
    print("\n📊 Demo: Basic Queries")
    print("-" * 40)
    
    basic_queries = [
        "Show me all employees",
        "How many employees do we have?",
        "List all departments with their budgets",
        "Find employees in the Engineering department",
        "Show me employees earning more than 80000"
    ]
    
    for query in basic_queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            result = await client.process_natural_language_query(
                session_id=session_id,
                natural_language_query=query,
                execute=True
            )
            
            print(f"   SQL: {result.generated_sql}")
            print(f"   Success: {result.execution_success}")
            
            if result.execution_success and result.execution_results:
                data = result.execution_results.get('data', [])
                print(f"   Results: {len(data)} rows")
                
                # Show sample results
                if data and len(data) <= 5:
                    for row in data[:3]:
                        print(f"   Sample: {dict(row)}")
                elif data:
                    print(f"   Sample: {dict(data[0])}")
                    
        except Exception as e:
            print(f"   Error: {e}")


async def demo_advanced_queries(client: NLPSQLClient, session_id: str):
    """Demonstrate advanced query functionality."""
    print("\n🚀 Demo: Advanced Queries")
    print("-" * 40)
    
    advanced_queries = [
        "What is the average salary by department?",
        "Show me the top 3 highest paid employees",
        "Find departments with more than 10 employees", 
        "Calculate total budget across all departments",
        "Show me employees and their managers",
        "Which departments have active projects?"
    ]
    
    for query in advanced_queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            result = await client.process_natural_language_query(
                session_id=session_id,
                natural_language_query=query,
                execute=True
            )
            
            print(f"   SQL: {result.generated_sql}")
            print(f"   Confidence: {result.confidence_score:.2f}")
            
            if result.execution_success and result.execution_results:
                data = result.execution_results.get('data', [])
                print(f"   Results: {len(data)} rows")
                
                if data:
                    for row in data[:2]:  # Show first 2 rows
                        print(f"   Result: {dict(row)}")
                        
        except Exception as e:
            print(f"   Error: {e}")


async def demo_data_analysis(client: NLPSQLClient, session_id: str):
    """Demonstrate data analysis capabilities."""
    print("\n📈 Demo: Data Analysis")
    print("-" * 40)
    
    try:
        # Get table schema
        schema = await client.get_table_schema("employees")
        print(f"✅ Retrieved schema for employees table")
        print(f"   Columns: {len(schema.get('columns', []))}")
        
        # Analyze data quality
        quality = await client.analyze_data_quality("employees")
        if quality:
            print(f"✅ Data quality score: {quality.get('overall_quality_score', 'N/A')}")
        
        # Get session statistics
        stats = await client.get_session_statistics(session_id)
        if stats:
            print(f"✅ Session stats: {stats['query_count']} queries, {stats['success_rate']}% success rate")
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")


async def demo_query_safety(client: NLPSQLClient, session_id: str):
    """Demonstrate query safety validation."""
    print("\n🔒 Demo: Query Safety Validation")
    print("-" * 40)
    
    # Test potentially dangerous queries
    dangerous_queries = [
        "Delete all employees",
        "Update all employee salaries to 100000",
        "Drop the employees table"
    ]
    
    for query in dangerous_queries:
        print(f"\n⚠️  Testing: '{query}'")
        
        try:
            # Generate SQL but don't execute
            result = await client.process_natural_language_query(
                session_id=session_id,
                natural_language_query=query,
                execute=False  # Safety: don't execute dangerous queries
            )
            
            print(f"   Generated SQL: {result.generated_sql}")
            print(f"   Warnings: {result.warnings}")
            
            # Check safety
            safety = await client.validate_sql_safety(result.generated_sql)
            print(f"   Risk Level: {safety.get('risk_level', 'unknown')}")
            print(f"   Is Safe: {safety.get('is_safe', False)}")
            
        except Exception as e:
            print(f"   Error: {e}")


async def demo_session_history(client: NLPSQLClient, session_id: str):
    """Demonstrate session history functionality."""
    print("\n📚 Demo: Session History")
    print("-" * 40)
    
    try:
        history = await client.get_session_history(session_id, limit=5)
        print(f"✅ Retrieved {len(history)} queries from session history")
        
        for i, entry in enumerate(history[-3:], 1):  # Show last 3 queries
            print(f"\n   Query {i}:")
            print(f"     NL: {entry['natural_language_query']}")
            print(f"     SQL: {entry['generated_sql']}")
            print(f"     Success: {entry['success']}")
            
    except Exception as e:
        print(f"❌ History error: {e}")


async def main():
    """Run the complete demo."""
    print("🎭 NLP to SQL System - Complete Demo")
    print("=" * 60)
    
    # Check API key
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if not claude_api_key:
        print("❌ CLAUDE_API_KEY environment variable not set!")
        print("   Set it with: export CLAUDE_API_KEY='your-anthropic-api-key'")
        return
    
    # Setup demo database
    await setup_demo_database()
    
    try:
        # Create client and session
        async with await create_client(claude_api_key) as client:
            session_id = await client.create_session("demo-user")
            print(f"✅ Created session: {session_id}")
            
            # Run demo sections
            await demo_basic_queries(client, session_id)
            await demo_advanced_queries(client, session_id)
            await demo_data_analysis(client, session_id)
            await demo_query_safety(client, session_id)
            await demo_session_history(client, session_id)
            
            print("\n🎉 Demo completed successfully!")
            print("\n💡 Tips for using the system:")
            print("   - Use natural language like 'Show me...' or 'Find...'")
            print("   - The system can handle JOINs, aggregations, and complex queries")
            print("   - All queries are validated for safety before execution")
            print("   - Session history is maintained for context")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())