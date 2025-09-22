# NLP to SQL System Validation Guide

## 🚀 How to Check Your System

This guide provides multiple ways to validate and test your NLP to SQL system with company data.

## 📋 Quick System Health Check

### 1. Configuration Check
```powershell
# Verify environment configuration
python -c "from config.settings import get_settings; settings = get_settings(); print(f'Claude API: {\"✅\" if settings.claude_api_key else \"❌\"}, Model: {settings.claude_model}')"
```

### 2. Basic Claude API Test
```powershell
# Test Claude API connectivity
python test_real_claude.py
```

### 3. Sample Data Verification
```powershell
# Check sample data is loaded
python -c "import json; print('📊 Sample Data:'); print(f'Employees: {len(json.load(open(\"data/sample_employees.json\")))}'); print(f'Projects: {len(json.load(open(\"data/sample_projects.json\")))}'); print(f'Departments: {len(json.load(open(\"data/sample_departments.json\")))}')"
```

## 🧪 Comprehensive Testing Options

### Option 1: Quick Mock Test (No API Calls)
```powershell
# Fast test with mock responses
python test_claude_with_data.py
```
- **Time**: ~30 seconds
- **Purpose**: Validate system structure without API costs
- **Expected**: 100% success rate with mock responses

### Option 2: Real API Test (Recommended)
```powershell
# Full test with real Claude API
python test_real_api_with_data.py
```
- **Time**: ~2-3 minutes
- **Purpose**: Complete end-to-end validation
- **Expected**: 100% success rate, 70-90% confidence scores

### Option 3: Manual Interactive Testing
```powershell
# Start Python interactive session
python
```
```python
# In Python shell:
from mcp_client.claude_interface import ClaudeInterface, NLQueryRequest
from config.settings import get_settings

# Initialize
settings = get_settings()
claude = ClaudeInterface(api_key=settings.claude_api_key, model=settings.claude_model, mock_mode=False)

# Test a query
request = NLQueryRequest(query="Show me all employees in Engineering with salaries over $90,000")
response = await claude.process_natural_language_query(request)
print(f"SQL: {response.sql_query}")
print(f"Confidence: {response.confidence_score}")
```

## 📊 Expected Results Guide

### ✅ Success Indicators

**Mock Test Results:**
```
📊 TEST REPORT SUMMARY
   Total Scenarios: 10
   Successful: 10
   Success Rate: 100.0%
   Average Confidence: 62.5%
```

**Real API Test Results:**
```
📊 OVERALL RESULTS:
   Total Scenarios: 8
   Successful: 8
   Success Rate: 100.0%
   Average Confidence: 75.6%
   Average SQL Quality: 5.9/10
```

### 🎯 Quality Examples

**High-Quality SQL (8-10/10):**
```sql
-- Department analysis with joins
SELECT e.department, AVG(e.salary) as avg_salary, 
       COUNT(e.id) as employee_count, d.budget as total_budget
FROM employees e JOIN departments d ON d.name = e.department
WHERE e.is_active = true
GROUP BY e.department, d.budget
HAVING AVG(e.performance_rating) > 4.0
ORDER BY avg_salary DESC
```

**Good SQL with Improvements (5-7/10):**
```sql
-- Employee search with manager join
SELECT e1.first_name, e1.last_name, e1.salary, e1.performance_rating,
       CONCAT(e2.first_name, ' ', e2.last_name) as manager_name
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.id
WHERE e1.department = 'Engineering' AND e1.is_active = true
ORDER BY e1.salary DESC
```

## 🔍 Test Your Own Queries

### Business Scenarios to Try

1. **HR Analytics:**
   - "Show me employees with performance ratings above 4.5"
   - "List all employees in Marketing department with their hire dates"
   - "Find employees who haven't been promoted in over 2 years"

2. **Project Management:**
   - "Which projects are over budget?"
   - "Show active projects with team sizes and budget utilization"
   - "List completed projects in Engineering department"

3. **Financial Analysis:**
   - "What's the average salary by department?"
   - "Show departments with total budget over $1 million"
   - "Calculate total project spending by department"

4. **Skills Inventory:**
   - "Find employees with Python or SQL skills"
   - "Show developers with React experience"
   - "List employees with AWS certifications"

### Custom Query Testing Template
```python
# Copy this template for testing your own queries
from mcp_client.claude_interface import ClaudeInterface, NLQueryRequest
import asyncio

async def test_custom_query(query_text):
    claude = ClaudeInterface(mock_mode=False)  # Set True for mock mode
    request = NLQueryRequest(query=query_text)
    response = await claude.process_natural_language_query(request)
    
    print(f"Query: {query_text}")
    print(f"SQL: {response.sql_query}")
    print(f"Confidence: {response.confidence_score:.1%}")
    print(f"Explanation: {response.explanation}")
    if response.warnings:
        print(f"Warnings: {response.warnings}")
    return response

# Test your query
# asyncio.run(test_custom_query("Your natural language query here"))
```

## 📈 Performance Benchmarks

### Response Time Expectations
- **Mock Mode**: < 2 seconds per query
- **Real API Mode**: 5-15 seconds per query (depends on Claude API)
- **Batch Testing**: ~2-3 minutes for 8 comprehensive scenarios

### Confidence Score Interpretation
- **90-100%**: Excellent - High confidence, likely accurate
- **70-89%**: Good - Reliable with minor improvements possible  
- **50-69%**: Fair - May need review or additional context
- **<50%**: Poor - Requires attention or query clarification

### SQL Quality Scoring
- **8-10/10**: Production-ready, follows best practices
- **5-7/10**: Good foundation, minor optimizations needed
- **3-4/10**: Functional but requires improvements
- **0-2/10**: Significant issues, needs revision

## 🐛 Troubleshooting

### Common Issues & Solutions

**❌ "No module named 'anthropic'"**
```powershell
pip install anthropic pydantic-settings
```

**❌ "Claude API Error: 401 Unauthorized"**
- Check your API key in `.env` file
- Verify key format: `sk-ant-api03-...`

**❌ "SQLQueryResponse object has no attribute 'error'"**
- Update test scripts to use correct response model fields
- Check `confidence_score` instead of `confidence`

**❌ Low confidence scores**
- Add more context to your queries
- Specify table names or column details
- Use business terminology consistently

**❌ Poor SQL quality**
- Provide schema information in requests
- Include sample data context
- Use specific field names in queries

## 📁 Generated Reports

After running tests, check these generated reports:

1. **`claude_company_data_report.json`** - Mock test results
2. **`real_api_company_test_report.json`** - Real API test results  
3. **`test_report.json`** - Basic API connectivity results

## 🎯 Success Checklist

- [ ] Environment configuration verified
- [ ] Dependencies installed successfully
- [ ] Sample data files present (15 employees, 8 projects, 5 departments)
- [ ] Mock testing passes (100% success rate)
- [ ] Real API testing passes (100% success rate) 
- [ ] Average confidence scores >70%
- [ ] SQL quality scores >5/10 average
- [ ] Custom queries work as expected
- [ ] Reports generated successfully

## 🚀 Ready for Production

Once all checks pass, your NLP to SQL system is ready for:
- Integration with existing databases
- Custom business logic implementation  
- User interface development
- Production deployment with MCP architecture

## 📞 Support

If you encounter issues:
1. Check the error messages in terminal output
2. Review the generated JSON reports for detailed diagnostics
3. Verify your `.env` configuration matches the expected format
4. Test with mock mode first before using real API
5. Check sample data integrity and format

---
*Last Updated: September 2025*