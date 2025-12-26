# Use a lightweight Python version
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (usually none for this stack)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------
# OPTIMIZATION: Pre-download the embedding model
# This prevents downloading it on every server restart
# -----------------------------------------------------
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Copy the rest of your application code
COPY . .

# Expose the port (Railway ignores this technically, but good practice)
EXPOSE 8080

# Run the server
CMD ["python", "mcp_server.py"]
