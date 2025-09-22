"""
Mock Claude API Responses for Testing.

Provides hardcoded responses to test the system without real API calls.
"""

from typing import Dict, List, Any, Optional
import random


# Mock Claude responses for different query types
MOCK_CLAUDE_RESPONSES = {
    "select_all_users": {
        "sql": "SELECT * FROM users WHERE is_active = 1",
        "confidence": 0.95,
        "explanation": "This query selects all active users from the users table.",
        "parameters": [],
        "query_type": "SELECT_ALL",
        "tables_involved": ["users"],
        "estimated_rows": 25
    },
    
    "count_employees": {
        "sql": "SELECT COUNT(*) as total_employees FROM employees WHERE is_active = 1",
        "confidence": 0.90,
        "explanation": "This query counts the total number of active employees.",
        "parameters": [],
        "query_type": "COUNT_RECORDS",
        "tables_involved": ["employees"],
        "estimated_rows": 1
    },
    
    "average_salary_by_department": {
        "sql": """
            SELECT 
                department,
                COUNT(*) as employee_count,
                AVG(salary) as average_salary,
                MIN(salary) as min_salary,
                MAX(salary) as max_salary
            FROM employees 
            WHERE is_active = 1 
            GROUP BY department 
            ORDER BY average_salary DESC
        """,
        "confidence": 0.85,
        "explanation": "This query calculates salary statistics grouped by department for active employees.",
        "parameters": [],
        "query_type": "SELECT_AGGREGATED",
        "tables_involved": ["employees"],
        "estimated_rows": 5
    },
    
    "users_with_recent_orders": {
        "sql": """
            SELECT 
                u.id,
                u.name,
                u.email,
                COUNT(o.id) as order_count,
                SUM(o.total_amount) as total_spent
            FROM users u
            JOIN orders o ON u.id = o.user_id
            WHERE o.created_at >= DATE('now', '-30 days')
            GROUP BY u.id, u.name, u.email
            HAVING COUNT(o.id) > 0
            ORDER BY total_spent DESC
        """,
        "confidence": 0.88,
        "explanation": "This query finds users who have placed orders in the last 30 days with their order statistics.",
        "parameters": [],
        "query_type": "SELECT_JOINED",
        "tables_involved": ["users", "orders"],
        "estimated_rows": 15
    },
    
    "insert_new_user": {
        "sql": "INSERT INTO users (name, email, age, department, salary, is_active) VALUES (?, ?, ?, ?, ?, ?)",
        "confidence": 0.92,
        "explanation": "This query inserts a new user record with the specified values.",
        "parameters": ["John Smith", "john.smith@company.com", 30, "Engineering", 75000.00, 1],
        "query_type": "INSERT_SINGLE",
        "tables_involved": ["users"],
        "estimated_rows": 1
    },
    
    "update_user_salary": {
        "sql": "UPDATE employees SET salary = ? WHERE id = ? AND is_active = 1",
        "confidence": 0.87,
        "explanation": "This query updates the salary for a specific active employee.",
        "parameters": [85000.00, 123],
        "query_type": "UPDATE_FILTERED",
        "tables_involved": ["employees"],
        "estimated_rows": 1
    },
    
    "delete_inactive_users": {
        "sql": "DELETE FROM users WHERE is_active = 0 AND last_login < DATE('now', '-365 days')",
        "confidence": 0.75,
        "explanation": "This query deletes inactive users who haven't logged in for over a year.",
        "parameters": [],
        "query_type": "DELETE_FILTERED",
        "tables_involved": ["users"],
        "estimated_rows": 8
    },
    
    "complex_reporting_query": {
        "sql": """
            WITH department_stats AS (
                SELECT 
                    department,
                    COUNT(*) as emp_count,
                    AVG(salary) as avg_salary
                FROM employees 
                WHERE is_active = 1 
                GROUP BY department
            ),
            project_stats AS (
                SELECT 
                    p.department_id,
                    COUNT(*) as project_count,
                    SUM(p.budget) as total_budget
                FROM projects p 
                WHERE p.status = 'active'
                GROUP BY p.department_id
            )
            SELECT 
                d.name as department_name,
                ds.emp_count,
                ds.avg_salary,
                COALESCE(ps.project_count, 0) as active_projects,
                COALESCE(ps.total_budget, 0) as project_budget,
                ROUND(ps.total_budget / ds.emp_count, 2) as budget_per_employee
            FROM departments d
            LEFT JOIN department_stats ds ON d.name = ds.department
            LEFT JOIN project_stats ps ON d.id = ps.department_id
            ORDER BY ds.emp_count DESC
        """,
        "confidence": 0.82,
        "explanation": "This complex query provides a comprehensive report combining employee statistics with project data by department.",
        "parameters": [],
        "query_type": "SELECT_COMPLEX",
        "tables_involved": ["departments", "employees", "projects"],
        "estimated_rows": 6
    }
}


# Mock result explanations
MOCK_EXPLANATIONS = {
    "select_all_users": {
        "explanation": "The query returned all active users in the system. There are currently 25 active users.",
        "insights": [
            "25 users are currently active in the system",
            "Most users are from the Engineering department",
            "Average user age is 32 years",
            "User registration has been steady over the past year"
        ],
        "suggestions": [
            "Try: Show users by department",
            "Try: Find users who joined recently", 
            "Try: Show user activity statistics"
        ]
    },
    
    "count_employees": {
        "explanation": "The query counted all active employees. The company currently has 47 active employees.",
        "insights": [
            "47 employees are currently active",
            "This represents a 12% increase from last quarter",
            "Engineering department has the most employees",
            "Recent hiring has been focused on technical roles"
        ],
        "suggestions": [
            "Try: Show employee breakdown by department",
            "Try: Calculate average employee tenure",
            "Try: Show recent hires"
        ]
    },
    
    "average_salary_by_department": {
        "explanation": "The salary analysis shows Engineering has the highest average salary at $82,500, followed by Sales at $68,900.",
        "insights": [
            "Engineering department has highest average salary ($82,500)",
            "Sales department averages $68,900 per employee",
            "Marketing department has most employees (15)",
            "Salary range varies significantly across departments"
        ],
        "suggestions": [
            "Try: Show salary distribution within departments",
            "Try: Find employees above/below department average",
            "Try: Compare salaries by experience level"
        ]
    }
}


# Mock error responses
MOCK_ERROR_RESPONSES = {
    "table_not_found": {
        "error": "Table 'nonexistent_table' doesn't exist in the database",
        "error_type": "TABLE_NOT_FOUND",
        "suggestions": [
            "Check available tables with: SHOW TABLES",
            "Verify table name spelling",
            "Ensure you have access to the table"
        ]
    },
    
    "sql_syntax_error": {
        "error": "SQL syntax error near 'SELCT' - did you mean 'SELECT'?",
        "error_type": "SYNTAX_ERROR", 
        "suggestions": [
            "Check SQL syntax",
            "Verify spelling of SQL keywords",
            "Use proper quotation marks for strings"
        ]
    },
    
    "permission_denied": {
        "error": "Permission denied for DROP operation on table 'users'",
        "error_type": "PERMISSION_DENIED",
        "suggestions": [
            "DROP operations are restricted for safety",
            "Contact administrator for schema changes",
            "Use UPDATE or DELETE for data modifications"
        ]
    }
}


def get_mock_claude_responses() -> Dict[str, Any]:
    """Get all mock Claude responses."""
    return MOCK_CLAUDE_RESPONSES


def get_mock_response_for_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Get appropriate mock response based on query content.
    
    Args:
        query: Natural language query string
        
    Returns:
        Mock response dict or None if no match found
    """
    query_lower = query.lower().strip()
    
    # Simple keyword matching for different query types
    if any(word in query_lower for word in ["show all users", "list all users", "get all users"]):
        return MOCK_CLAUDE_RESPONSES["select_all_users"]
    
    elif any(word in query_lower for word in ["count employees", "how many employees", "number of employees"]):
        return MOCK_CLAUDE_RESPONSES["count_employees"]
    
    elif any(word in query_lower for word in ["average salary", "salary by department", "department salary"]):
        return MOCK_CLAUDE_RESPONSES["average_salary_by_department"]
    
    elif any(word in query_lower for word in ["recent orders", "users with orders", "order statistics"]):
        return MOCK_CLAUDE_RESPONSES["users_with_recent_orders"]
    
    elif any(word in query_lower for word in ["add user", "create user", "insert user", "new user"]):
        return MOCK_CLAUDE_RESPONSES["insert_new_user"]
    
    elif any(word in query_lower for word in ["update salary", "change salary", "modify salary"]):
        return MOCK_CLAUDE_RESPONSES["update_user_salary"]
    
    elif any(word in query_lower for word in ["delete inactive", "remove inactive", "cleanup users"]):
        return MOCK_CLAUDE_RESPONSES["delete_inactive_users"]
    
    elif any(word in query_lower for word in ["department report", "comprehensive report", "department statistics"]):
        return MOCK_CLAUDE_RESPONSES["complex_reporting_query"]
    
    # Default to a simple select response
    return {
        "sql": f"SELECT * FROM users WHERE name LIKE '%{query}%' LIMIT 10",
        "confidence": 0.60,
        "explanation": f"This query searches for records matching '{query}'.",
        "parameters": [],
        "query_type": "SELECT_FILTERED",
        "tables_involved": ["users"],
        "estimated_rows": 5
    }


def get_mock_explanation_for_query(query_type: str) -> Dict[str, Any]:
    """Get mock explanation for a query type."""
    return MOCK_EXPLANATIONS.get(query_type, {
        "explanation": "The query was executed successfully and returned the requested data.",
        "insights": ["Query executed successfully", "Data retrieved from database"],
        "suggestions": ["Try modifying the query parameters", "Consider adding filters for more specific results"]
    })


def get_random_mock_error() -> Dict[str, Any]:
    """Get a random mock error response for testing error handling."""
    error_keys = list(MOCK_ERROR_RESPONSES.keys())
    error_key = random.choice(error_keys)
    return MOCK_ERROR_RESPONSES[error_key]


def simulate_api_delay() -> float:
    """Simulate API response delay (in seconds) for realistic testing."""
    return random.uniform(0.1, 2.0)  # 100ms to 2 seconds