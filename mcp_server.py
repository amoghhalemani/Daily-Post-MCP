#!/usr/bin/env python3
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
    
    import uvicorn
    
    mcp.run_server(
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
