# NLP to SQL Project with MCP Architecture

This project implements an intelligent data management system that converts natural language queries into SQL operations using Claude API and MCP (Model Context Protocol) server-client architecture.

## Project Structure

### MCP Server (`mcp_server/`)
- Provides tools for data ingestion, SQL execution, and analysis
- Handles database connections and operations
- Implements security and validation

### MCP Client (`mcp_client/`) 
- Interfaces with Claude API for natural language processing
- Converts NLP queries to SQL operations
- Manages user sessions and responses

### Database Layer (`database/`)
- SQLite/PostgreSQL integration with SQLAlchemy
- Data models and schema management
- Migration and seeding utilities

### NLP Pipeline (`nlp/`)
- Intent classification and entity extraction
- SQL generation from natural language
- Query validation and optimization

### Testing (`tests/`)
- Unit tests for MCP tools and client-server communication
- Integration tests and security tests
- Performance and load testing

## Development Guidelines

- Use Python 3.11+ with type hints
- Follow MCP protocol specifications
- Implement proper error handling and logging
- Ensure security best practices for SQL injection prevention
- Use environment variables for sensitive configuration