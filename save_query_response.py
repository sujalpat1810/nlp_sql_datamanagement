#!/usr/bin/env python3
"""
Save specific query response to file
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_client.claude_interface import ClaudeInterface, NLQueryRequest
from config.settings import get_settings

async def save_query_response(query, filename=None):
    """Test query and save response to file"""
    
    # Load sample data for schema context
    with open("data/sample_employees.json") as f:
        employees = json.load(f)
    
    schema_context = {
        "employees_table": {
            "description": "Employee records table",
            "structure": "employees(id, first_name, last_name, email, phone, hire_date, job_title, department, salary, manager_id, is_active, location, skills, performance_rating, last_promotion)",
            "sample_record": employees[0],
            "total_records": len(employees)
        }
    }
    
    settings = get_settings()
    claude = ClaudeInterface(
        api_key=settings.claude_api_key,
        model=settings.claude_model,
        mock_mode=False
    )
    
    print(f"🔍 Processing Query: '{query}'")
    print("⏳ Getting response from Claude API...")
    
    request = NLQueryRequest(
        query=query,
        schema_info=schema_context
    )
    
    response = await claude.process_natural_language_query(request)
    
    # Create response data
    response_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": {
            "sql_query": response.sql_query,
            "explanation": response.explanation,
            "confidence_score": response.confidence_score,
            "query_type": response.query_type,
            "estimated_complexity": response.estimated_complexity,
            "warnings": response.warnings,
            "suggestions": response.suggestions,
            "tables_used": response.tables_used,
            "estimated_rows": response.estimated_rows
        },
        "sample_data_context": {
            "total_employees": len(employees),
            "highest_salary": max(emp['salary'] for emp in employees),
            "highest_paid_employee": next(emp for emp in employees if emp['salary'] == max(emp['salary'] for emp in employees))
        }
    }
    
    # Generate filename if not provided
    if not filename:
        safe_query = "".join(c for c in query[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"query_response_{safe_query.replace(' ', '_')}.json"
    
    # Save to file
    output_file = PROJECT_ROOT / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Response saved to: {output_file}")
    
    # Also print the response
    print(f"\n📊 Confidence: {response.confidence_score:.1%}")
    print(f"📝 Generated SQL:")
    print("-" * 50)
    print(response.sql_query)
    print("-" * 50)
    print(f"\n💡 Explanation:")
    print(response.explanation)
    
    return response_data

async def main():
    """Save the specific query you asked about"""
    query = "list down the employees working in the company and who has the highest salary"
    
    response_data = await save_query_response(query, "your_query_response.json")
    
    print(f"\n🎯 Your query response has been saved!")
    print(f"📁 File: your_query_response.json")
    print(f"📊 You can view it anytime to see the SQL, explanation, and confidence scores.")

if __name__ == "__main__":
    asyncio.run(main())