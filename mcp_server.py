#!/usr#!/usr/bin/env python3
"""
MCP Server for Weaviate RAG Pipeline using FastMCP 2
Clean server implementation using decorator-based tool registration
"""
import os
from fastmcp import FastMCP

# ============================================
# Create FastMCP Server Instance (SINGLE INSTANCE)
# ============================================

mcp = FastMCP("weaviate-rag-server")

# ============================================
# Import Tools AFTER mcp is created
# ============================================

# Import the module first (before decorators are applied)
import tools

# Set the mcp instance in rag_tool
tools.mcp = mcp

# Now call the registration function to apply decorators
tools.register_tools()

# ============================================
# Server Entry Point
# ============================================

if __name__ == "__main__":
     port = int(os.getenv("PORT", 8000))
    
    # FastMCP runs on stdio by default, but we need SSE for remote access
    # Use uvicorn to expose it as an HTTP server
    import uvicorn
    
    mcp.run_server(
        host="0.0.0.0",  # Listen on all interfaces
        port=port,
        log_level="info"
    )
