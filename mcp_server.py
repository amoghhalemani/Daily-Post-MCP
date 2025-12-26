#!/usr/bin/env python3
"""
MCP Server for Weaviate RAG Pipeline using FastMCP 2
"""
import os
import uvicorn
from fastmcp import FastMCP

# ============================================
# Create FastMCP Server Instance
# ============================================

mcp = FastMCP("weaviate-rag-server")

# ============================================
# Import Tools
# ============================================

import tools

# Set the mcp instance in tools
tools.mcp = mcp

# Register tools
tools.register_tools()

# ============================================
# Server Entry Point
# ============================================

if __name__ == "__main__":
    railway_port = os.getenv("PORT")
    
    if railway_port:
        print(f"🚀 Starting in SSE mode on port {railway_port}...")
        try:
            mcp.run(transport="sse", host="0.0.0.0", port=int(railway_port))
        except TypeError:
            print("⚠️ Transport argument not supported, checking for ASGI app...")
            pass 
    else:
        mcp.run()
