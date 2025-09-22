# NLP to SQL Data Management System 🚀

A powerful data management system that converts natural language queries into SQL operations using Claude AI API. Upload CSV/JSON files and query them using plain English!

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Claude AI](https://img.shields.io/badge/Claude-3.5_Sonnet-purple.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-181717?logo=github)](https://github.com/sujalpat1810/nlp_sql_datamanagement)

## 🌟 Features

- **Natural Language Queries**: Ask questions in plain English
- **File Upload Support**: Upload CSV or JSON files via drag-and-drop
- **Real-time SQL Generation**: Converts your questions to optimized SQL queries
- **Query Execution**: Execute queries and see results instantly
- **Export Functionality**: Export query results as CSV
- **Mock Mode**: Test without API key using intelligent mock responses
- **WebSocket Support**: Real-time communication between frontend and backend

## 🎯 Upcoming Features

- [ ] Data Visualization (Charts & Graphs)
- [ ] Multi-CSV File Joins and Relationships
- [ ] Advanced Data Manipulation Tools
- [ ] Google Sheets Integration
- [ ] Google Drive Upload/Sync
- [ ] Query History & Saved Queries
- [ ] Team Collaboration Features
- [ ] API Endpoints for External Integration

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Claude API key (optional - can use mock mode)

### Installation

1. Clone the repository
```bash
git clone https://github.com/sujalpat1810/nlp_sql_datamanagement.git
cd nlp_sql_datamanagement
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create `.env` file (optional for real API)
```env
CLAUDE_API_KEY=your_claude_api_key_here
```

4. Start the backend server
```bash
python start_backend.py
```

5. Open your browser to `http://localhost:8000`

## 💻 Usage

1. **Upload Data**: Drag and drop a CSV or JSON file
2. **Ask Questions**: Type queries like:
   - "Show all employees with salary above 100000"
   - "Who has the highest salary?"
   - "List employees in Engineering department"
3. **View Results**: See SQL query and data results
4. **Export**: Download results as CSV

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Frontend UI   │────▶│  FastAPI Backend│────▶│  Claude AI API  │
│   (HTML/JS)     │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        
         │                       ▼                        
         │              ┌─────────────────┐              
         │              │                 │              
         └─────────────▶│  SQLite (In-    │              
                        │  Memory DB)     │              
                        └─────────────────┘              
```

## 📁 Project Structure

```
nlp_sql_datamanagement/
├── backend/            # FastAPI backend server
│   └── app.py         # Main application
├── frontend/          # Web UI
│   └── index.html     # Single-page application
├── mcp_client/        # Claude API interface
│   ├── claude_interface.py
│   └── client.py
├── database/          # Database operations
│   ├── connection.py
│   ├── models.py
│   └── query_executor.py
├── config/            # Configuration
│   ├── settings.py
│   └── mock_responses.py
├── data/              # Data storage
│   └── sample_*.json  # Sample data files
├── tests/             # Test files
├── requirements.txt   # Python dependencies
└── start_backend.py   # Server launcher
```

## 🔧 Configuration

### Environment Variables (.env)
```env
# Claude API Configuration
CLAUDE_API_KEY=your_api_key_here  # Get from https://console.anthropic.com

# Optional Configuration
CLAUDE_MODEL=claude-3-5-sonnet-20241022
DATABASE_URL=sqlite:///./data/nlp_sql.db
LOG_LEVEL=INFO
```

## 📊 Example Queries

```
- "Show all employees"
- "Who has the highest salary in the company?"
- "List employees in the Engineering department"
- "Show average salary by department"
- "Find employees hired after 2020"
- "Count employees in each team"
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Test with mock data
python test_mock_claude.py

# Test real API (requires API key)
python test_real_claude.py
```

## 🛡️ Security

- SQL injection prevention through parameterized queries
- Input validation and sanitization
- Secure API key management (.env not in repo)
- CORS protection in production

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Claude AI by Anthropic for natural language processing
- FastAPI for the backend framework
- SQLite for lightweight database operations

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/nlp_sql_datamanagement](https://github.com/yourusername/nlp_sql_datamanagement)

---

⭐ Don't forget to star this repo if you find it useful!