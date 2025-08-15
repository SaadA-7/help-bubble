FROM python:3.11-slim

WORKDIR /app

# Set memory optimization environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code
COPY . .

# Create cache directories
RUN mkdir -p /tmp/transformers_cache /tmp/huggingface

EXPOSE 8000

# Use uvicorn with optimized settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]