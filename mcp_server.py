import os
from fastapi import FastAPI
import uvicorn
import tools

app = FastAPI()

# Register tools
tools.register_tools()

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
