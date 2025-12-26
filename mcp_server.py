import os
import sys
from fastmcp import FastMCP

# ============================================
# Create FastMCP Server Instance
# ============================================

mcp = FastMCP("weaviate-rag-server")

# ============================================
# Import Tools AFTER mcp is created
# ============================================

import tools

# Set the mcp instance in tools
tools.mcp = mcp

# Now call the registration function to apply decorators
tools.register_tools()

# ============================================
# Server Entry Point
# ============================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    # Check if running in cloud (Railway sets PORT)
    is_cloud = os.getenv("PORT") or os.getenv("K_SERVICE")
    
    if is_cloud:
        print(f"🚀 Starting MCP Server (Streamable HTTP + Stateless) on 0.0.0.0:{port}...")
        mcp.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=port,
            path="/mcp",
            log_level="debug",
        )
    else:
        print("🚀 Starting MCP Server in STDIO mode (local)...", file=sys.stderr)
        mcp.run()
