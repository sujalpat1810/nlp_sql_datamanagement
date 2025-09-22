#!/usr/bin/env python3
"""
FastAPI Backend for NLP to SQL System
Provides REST API endpoints for natural language query processing
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import uvicorn

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from mcp_client.claude_interface import ClaudeInterface, NLQueryRequest
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.query_executor import get_query_executor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NLP to SQL API",
    description="Natural Language to SQL Query Converter using Claude AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
settings = get_settings()
claude_interface = None

# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    use_mock: bool = Field(False, description="Use mock mode instead of real API")
    include_schema: bool = Field(True, description="Include schema context")
    execute_query: bool = Field(True, description="Execute the generated SQL and return results")
    
class QueryResponse(BaseModel):
    success: bool
    query: str
    sql: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    query_type: Optional[str] = None
    complexity: Optional[str] = None
    warnings: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    error: Optional[str] = None
    timestamp: str
    execution_time: Optional[float] = None
    results: Optional[List[Dict[str, Any]]] = None
    result_count: Optional[int] = None
    columns: Optional[List[str]] = None

class SchemaInfo(BaseModel):
    tables: List[Dict[str, Any]]
    total_records: Dict[str, int]
    sample_data: Dict[str, Any]

class SystemStatus(BaseModel):
    status: str
    api_configured: bool
    mock_available: bool
    sample_data_loaded: bool
    version: str

class FileUpload(BaseModel):
    type: str = Field(..., description="File type: 'json' or 'csv'")
    fileName: str = Field(..., description="Original file name")
    data: Any = Field(..., description="Parsed file data")

class UploadRequest(BaseModel):
    files: List[FileUpload] = Field(..., description="List of uploaded files")

class UploadResponse(BaseModel):
    success: bool
    message: str
    processed_files: List[str]
    total_records: Dict[str, int]

# Initialize Claude interface
@app.on_event("startup")
async def startup_event():
    global claude_interface
    try:
        # Check if real API is available
        has_api_key = bool(settings.claude_api_key and settings.claude_api_key != "test_api_key")
        claude_interface = ClaudeInterface(
            api_key=settings.claude_api_key if has_api_key else None,
            model=settings.claude_model,
            mock_mode=not has_api_key
        )
        logger.info(f"Claude interface initialized (Mock mode: {not has_api_key})")
    except Exception as e:
        logger.error(f"Failed to initialize Claude interface: {e}")

# Load sample data for schema context
def get_schema_context():
    """Load and return schema context from all data files"""
    schema_context = {}
    data_dir = PROJECT_ROOT / "data"
    
    # Check if data directory exists
    if not data_dir.exists():
        return schema_context
    
    # Load all JSON files in the data directory
    for json_file in data_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                
            # Determine table name from filename
            table_name = json_file.stem
            
            # Handle both list and single object formats
            if isinstance(data, list) and len(data) > 0:
                schema_context[table_name] = {
                    "description": f"{table_name.replace('_', ' ').title()} table",
                    "fields": list(data[0].keys()),
                    "sample_record": data[0],
                    "total_records": len(data)
                }
            elif isinstance(data, dict):
                schema_context[table_name] = {
                    "description": f"{table_name.replace('_', ' ').title()} data",
                    "fields": list(data.keys()),
                    "sample_record": data,
                    "total_records": 1
                }
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")
            continue
    
    return schema_context

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NLP to SQL API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    has_api_key = bool(settings.claude_api_key and settings.claude_api_key != "test_api_key")
    schema_context = get_schema_context()
    
    return SystemStatus(
        status="healthy",
        api_configured=has_api_key,
        mock_available=True,
        sample_data_loaded=bool(schema_context),
        version="1.0.0"
    )

@app.get("/schema")
async def get_schema():
    """Get database schema information"""
    schema_context = get_schema_context()
    
    tables = []
    total_records = {}
    sample_data = {}
    
    for table_name, info in schema_context.items():
        tables.append({
            "name": table_name,
            "description": info["description"],
            "fields": info["fields"],
            "record_count": info["total_records"]
        })
        total_records[table_name] = info["total_records"]
        sample_data[table_name] = info["sample_record"]
    
    return SchemaInfo(
        tables=tables,
        total_records=total_records,
        sample_data=sample_data
    )

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process natural language query and return SQL"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Get schema context if requested
        schema_info = get_schema_context() if request.include_schema else None
        
        # Create Claude request
        claude_request = NLQueryRequest(
            query=request.query,
            schema_info=schema_info
        )
        
        # Use appropriate interface
        if request.use_mock:
            mock_interface = ClaudeInterface(mock_mode=True)
            response = await mock_interface.process_natural_language_query(claude_request)
        else:
            response = await claude_interface.process_natural_language_query(claude_request)
        
        # Execute the query if requested
        results = None
        result_count = None
        columns = None
        
        if request.execute_query and response.sql_query:
            try:
                # Get query executor
                data_dir = PROJECT_ROOT / "data"
                executor = get_query_executor(data_dir)
                
                # Execute the query
                query_results, query_columns, exec_error = executor.execute_query(response.sql_query)
                
                if exec_error:
                    if not response.warnings:
                        response.warnings = []
                    response.warnings.append(f"Query execution warning: {exec_error}")
                else:
                    results = query_results
                    result_count = len(query_results)
                    columns = query_columns
                    logger.info(f"Query executed successfully, returned {result_count} results")
                    
            except Exception as e:
                logger.error(f"Error executing SQL query: {e}")
                if not response.warnings:
                    response.warnings = []
                response.warnings.append(f"Could not execute query: {str(e)}")
        
        # Calculate execution time
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return QueryResponse(
            success=True,
            query=request.query,
            sql=response.sql_query,
            explanation=response.explanation,
            confidence=response.confidence_score,
            query_type=response.query_type,
            complexity=response.estimated_complexity,
            warnings=response.warnings,
            suggestions=response.suggestions,
            timestamp=datetime.now().isoformat(),
            execution_time=round(execution_time, 2),
            results=results,
            result_count=result_count,
            columns=columns
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(
            success=False,
            query=request.query,
            error=str(e),
            timestamp=datetime.now().isoformat(),
            execution_time=round(asyncio.get_event_loop().time() - start_time, 2)
        )

@app.get("/examples")
async def get_example_queries():
    """Get example queries for testing"""
    return {
        "examples": [
            {
                "category": "Employee Queries",
                "queries": [
                    "Show all employees in the Engineering department",
                    "Who has the highest salary in the company?",
                    "Find employees with performance rating above 4.5",
                    "List employees hired in the last 2 years"
                ]
            },
            {
                "category": "Project Queries", 
                "queries": [
                    "Which projects are over budget?",
                    "Show all active projects with their budgets",
                    "List projects by department",
                    "Find projects ending this year"
                ]
            },
            {
                "category": "Department Analytics",
                "queries": [
                    "What is the average salary by department?",
                    "Show department budgets and employee counts",
                    "Which department has the most employees?",
                    "Calculate total budget across all departments"
                ]
            },
            {
                "category": "Skills Analysis",
                "queries": [
                    "Find employees with Python skills",
                    "List employees who know SQL or Python",
                    "Show developers with React experience",
                    "Which skills are most common?"
                ]
            }
        ]
    }

@app.get("/stats")
async def get_statistics():
    """Get data statistics for all loaded tables"""
    schema_context = get_schema_context()
    
    stats = {
        "total_tables": len(schema_context),
        "data_loaded": bool(schema_context),
        "tables": {}
    }
    
    # Add stats for each table
    for table_name, table_info in schema_context.items():
        stats["tables"][table_name] = {
            "record_count": table_info["total_records"],
            "field_count": len(table_info["fields"]),
            "fields": table_info["fields"]
        }
    
    # Backward compatibility
    stats["total_employees"] = schema_context.get("employees", {}).get("total_records", 0) or schema_context.get("sample_employees", {}).get("total_records", 0)
    stats["total_projects"] = schema_context.get("projects", {}).get("total_records", 0) or schema_context.get("sample_projects", {}).get("total_records", 0)
    stats["total_departments"] = schema_context.get("departments", {}).get("total_records", 0) or schema_context.get("sample_departments", {}).get("total_records", 0)
    
    # Add detailed employee stats if available
    employees_table = "employees" if "employees" in schema_context else "sample_employees" if "sample_employees" in schema_context else None
    if employees_table:
        data_dir = PROJECT_ROOT / "data"
        try:
            with open(data_dir / f"{employees_table}.json") as f:
                employees = json.load(f)
                
            if employees and all('salary' in emp for emp in employees):
                stats["salary_range"] = {
                    "min": min(emp["salary"] for emp in employees),
                    "max": max(emp["salary"] for emp in employees),
                    "average": sum(emp["salary"] for emp in employees) / len(employees)
                }
                
                # Department distribution
                if all('department' in emp for emp in employees):
                    dept_counts = {}
                    for emp in employees:
                        dept = emp["department"]
                        dept_counts[dept] = dept_counts.get(dept, 0) + 1
                    stats["employees_by_department"] = dept_counts
        except Exception as e:
            logger.warning(f"Could not load detailed employee stats: {e}")
    
    return stats

@app.post("/upload")
async def upload_data(request: UploadRequest):
    """Handle file uploads and process data"""
    try:
        processed_files = []
        total_records = {}
        data_dir = PROJECT_ROOT / "data"
        temp_dir = PROJECT_ROOT / "temp"
        
        # Create directories if they don't exist
        data_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)
        
        for file_upload in request.files:
            filename = file_upload.fileName
            file_type = file_upload.type
            data = file_upload.data
            
            # Determine table name from filename
            table_name = Path(filename).stem.lower()
            if 'employee' in table_name:
                table_name = 'employees'
            elif 'project' in table_name:
                table_name = 'projects'
            elif 'department' in table_name:
                table_name = 'departments'
            else:
                # Save to temp directory for custom tables
                table_name = Path(filename).stem.replace(' ', '_').lower()
            
            # Save the data
            if file_type == 'json':
                # Save JSON data
                save_path = data_dir / f"{table_name}.json"
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                total_records[table_name] = len(data) if isinstance(data, list) else 1
                
            elif file_type == 'csv':
                # Convert CSV data to JSON format and save
                save_path = data_dir / f"{table_name}.json"
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                total_records[table_name] = len(data)
            
            processed_files.append(filename)
            logger.info(f"Processed file: {filename} -> {save_path}")
        
        # Clear cache to reload schema
        get_schema_context.cache_clear() if hasattr(get_schema_context, 'cache_clear') else None
        
        return UploadResponse(
            success=True,
            message=f"Successfully processed {len(processed_files)} file(s)",
            processed_files=processed_files,
            total_records=total_records
        )
        
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time queries
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time query processing"""
    await websocket.accept()
    try:
        while True:
            # Receive query
            data = await websocket.receive_json()
            query = data.get("query", "")
            use_mock = data.get("use_mock", False)
            
            # Send processing status
            await websocket.send_json({
                "type": "status",
                "message": "Processing query..."
            })
            
            # Process query
            try:
                schema_info = get_schema_context()
                claude_request = NLQueryRequest(
                    query=query,
                    schema_info=schema_info
                )
                
                if use_mock:
                    mock_interface = ClaudeInterface(mock_mode=True)
                    response = await mock_interface.process_natural_language_query(claude_request)
                else:
                    response = await claude_interface.process_natural_language_query(claude_request)
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "success": True,
                    "data": {
                        "sql": response.sql_query,
                        "explanation": response.explanation,
                        "confidence": response.confidence_score,
                        "query_type": response.query_type,
                        "complexity": response.estimated_complexity,
                        "warnings": response.warnings,
                        "suggestions": response.suggestions
                    }
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )