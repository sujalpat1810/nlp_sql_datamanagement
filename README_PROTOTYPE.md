# NLP to SQL Prototype

A working prototype that converts natural language queries into SQL using Claude AI and MCP (Model Context Protocol) architecture.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)
The system works in both mock mode and real API mode. For real API mode, ensure your `.env` file contains:
```
CLAUDE_API_KEY=your_actual_api_key_here
```

### 3. Launch the Prototype
```bash
python launch_prototype.py
```

This will:
- Start the FastAPI backend server on http://localhost:8000
- Open the frontend UI in your default browser
- Load sample company data (employees, projects, departments)

## 🎯 Features

### Frontend UI
- **Modern Web Interface**: Clean, responsive design with dark mode SQL display
- **Example Queries**: Click on pre-built examples to test the system
- **Real-time Processing**: See your natural language converted to SQL instantly
- **Confidence Scores**: Visual indicators showing query processing confidence
- **Mock/Real Mode Toggle**: Switch between mock responses and real Claude API

### Backend API
- **RESTful Endpoints**: Full API with documentation at http://localhost:8000/docs
- **WebSocket Support**: Real-time query processing (experimental)
- **Schema Context**: Automatically includes database schema for better SQL generation
- **Error Handling**: Comprehensive error messages and suggestions

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | System health and status check |
| `/schema` | GET | Get database schema information |
| `/query` | POST | Convert natural language to SQL |
| `/examples` | GET | Get example queries |
| `/stats` | GET | Get data statistics |
| `/ws` | WebSocket | Real-time query processing |

## 🧪 Testing the System

### Using the UI
1. Open the frontend in your browser
2. Click on any example query or type your own
3. Toggle between Mock/Real mode
4. Click "Convert to SQL" to see results

### Using the API Directly
```bash
# Health check
curl http://localhost:8000/health

# Process a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show all employees with salary above 100000", "use_mock": true}'
```

### Example Natural Language Queries
- "Show all employees in the Engineering department"
- "Who has the highest salary in the company?"
- "List projects that are over budget"
- "Find employees with Python skills"
- "What is the average salary by department?"

## 📊 Sample Data

The prototype includes sample data for testing:
- **15 Employees**: With names, salaries, departments, skills, and performance ratings
- **8 Projects**: Including budgets, statuses, and team assignments
- **5 Departments**: Engineering, Sales, Marketing, HR, and Finance

## 🔧 Configuration

### Environment Variables
- `CLAUDE_API_KEY`: Your Anthropic Claude API key
- `CLAUDE_MODEL`: Model to use (default: claude-3-5-sonnet-20241022)
- `USE_MOCK_MODE`: Force mock mode even with API key
- `DATABASE_URL`: SQLite database path

### Mock Mode
When no API key is configured or when mock mode is enabled, the system provides:
- Pre-defined responses for common query patterns
- Consistent SQL generation for testing
- No API costs or rate limits

## 🛠️ Troubleshooting

### Port Already in Use
If you see "Port 8000 is already in use", either:
1. Stop the existing server: `Ctrl+C` in the terminal
2. Or kill the process: `kill $(lsof -t -i:8000)` (Linux/Mac)

### Frontend Not Loading
If the frontend doesn't open automatically:
1. Navigate to `frontend/index.html` 
2. Open it manually in your browser
3. Ensure JavaScript is enabled

### API Connection Issues
- Check the status indicator in the UI header
- Verify backend is running at http://localhost:8000
- Check browser console for errors

## 📝 Development

### Project Structure
```
nlp_sql_datamangement/
├── backend/
│   └── app.py          # FastAPI server
├── frontend/
│   └── index.html      # Web UI (all-in-one)
├── mcp_client/         # Claude interface
├── config/             # Settings and configuration
├── data/               # Sample JSON data
└── launch_prototype.py # Launcher script
```

### Adding New Features
1. **Backend**: Modify `backend/app.py` to add new endpoints
2. **Frontend**: Edit `frontend/index.html` for UI changes
3. **Query Processing**: Update `mcp_client/claude_interface.py`

## 🚦 Next Steps

This is a working prototype. To productionize:
1. Add authentication and user sessions
2. Implement real database connections
3. Add query result execution and display
4. Enhance error handling and logging
5. Deploy to cloud infrastructure
6. Add query history and favorites
7. Implement data visualization for results

## 🤝 Support

For issues or questions:
1. Check the console logs for detailed errors
2. Verify all dependencies are installed
3. Ensure your API key is valid (if using real mode)
4. Review the API documentation at http://localhost:8000/docs