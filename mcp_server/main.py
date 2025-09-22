"""
Main entry point for the NLP to SQL MCP Server.

This server provides comprehensive data management tools through the Model Context Protocol,
enabling natural language to SQL query conversion and database operations.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import Implementation, ServerInfo

from database.connection import DatabaseManager
from .tools import (
    DataIngestionTools,
    QueryExecutionTools, 
    DataAnalysisTools,
    AdvancedOperationsTools
)


# Configure logging to stderr (not stdout for MCP compliance)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Application lifespan manager."""
    logger.info("Starting NLP to SQL MCP Server...")
    
    # Initialize database connection
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    # Store database manager in app state
    app.state = {"db_manager": db_manager}
    
    logger.info("MCP Server initialized successfully")
    yield
    
    # Cleanup
    await db_manager.close()
    logger.info("MCP Server shutdown complete")


def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    
    # Initialize FastMCP server
    server = FastMCP(
        "nlp-sql-server",
        lifespan=lifespan
    )
    
    # Server information
    server.server_info = ServerInfo(
        name="NLP to SQL MCP Server",
        version="0.1.0"
    )
    
    return server


def register_tools(server: FastMCP) -> None:
    """Register all MCP tools with the server."""
    
    # Initialize tool handlers
    data_ingestion = DataIngestionTools()
    query_execution = QueryExecutionTools()
    data_analysis = DataAnalysisTools()
    advanced_ops = AdvancedOperationsTools()
    
    # Register Data Ingestion Tools
    server.add_tool(data_ingestion.load_csv_data)
    server.add_tool(data_ingestion.load_json_data)
    server.add_tool(data_ingestion.create_table_from_schema)
    server.add_tool(data_ingestion.validate_data_format)
    server.add_tool(data_ingestion.infer_schema)
    
    # Register Query Execution Tools
    server.add_tool(query_execution.execute_select_query)
    server.add_tool(query_execution.execute_insert_query)
    server.add_tool(query_execution.execute_update_query)
    server.add_tool(query_execution.execute_delete_query)
    server.add_tool(query_execution.execute_custom_sql)
    server.add_tool(query_execution.validate_sql_syntax)
    
    # Register Data Analysis Tools
    server.add_tool(data_analysis.get_table_schema)
    server.add_tool(data_analysis.get_data_summary)
    server.add_tool(data_analysis.calculate_statistics)
    server.add_tool(data_analysis.find_null_values)
    server.add_tool(data_analysis.detect_outliers)
    server.add_tool(data_analysis.get_unique_values)
    server.add_tool(data_analysis.analyze_data_quality)
    
    # Register Advanced Operations Tools
    server.add_tool(advanced_ops.create_join_query)
    server.add_tool(advanced_ops.create_aggregation_query)
    server.add_tool(advanced_ops.create_database_view)
    server.add_tool(advanced_ops.create_database_index)
    server.add_tool(advanced_ops.create_subquery)
    server.add_tool(advanced_ops.execute_union_query)
    server.add_tool(advanced_ops.create_common_table_expression)
    server.add_tool(advanced_ops.manage_database_views)
    server.add_tool(advanced_ops.manage_database_indexes)
    server.add_tool(advanced_ops.explain_query_plan)
    
    logger.info(f"Registered {len(server.tools)} MCP tools")


async def main():
    """Main entry point for the MCP server."""
    try:
        # Create server instance
        server = create_server()
        
        # Register all tools
        register_tools(server)
        
        # Run server with stdio transport
        await server.run(transport='stdio')
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())