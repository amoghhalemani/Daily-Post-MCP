# Use a standard lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git is sometimes needed for certain pip packages)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies globally
# This ensures python can find 'sentence_transformers' immediately
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model (Now this will work because the package is installed)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Copy the rest of your application code
COPY . .

# Create the style guide file to prevent startup errors
RUN touch sanjay_sahay_style.txt

# Expose the port
EXPOSE 8080

# Run the server
CMD ["python", "mcp_server.py"]
