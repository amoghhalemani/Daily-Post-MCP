# ==========================================
# Stage 1: Builder (Compilers & Downloads)
# ==========================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install git/build tools needed for pip
RUN apt-get update && apt-get install -y git

# Install dependencies into a specific location
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download the model to a specific cache directory
ENV HF_HOME=/app/model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# ==========================================
# Stage 2: Runtime (Clean & Minimal)
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy the downloaded model cache from builder
COPY --from=builder /app/model_cache /root/.cache/huggingface

# Copy app code
COPY . .

# Create the style guide file
RUN touch sanjay_sahay_style.txt

EXPOSE 8080

# Environment variable to ensure model finds the cached files
ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "mcp_server.py"]
