# Dockerfile for CXR-Report-Metric Azure Deployment - Optimized for GPU
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies optimized for Azure
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    unzip \
    sqlite3 \
    git \
    curl \
    htop \
    nvtop \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables for optimal GPU performance
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONPATH=/app:$PYTHONPATH
ENV TORCH_HOME=/app/models/.torch
ENV TRANSFORMERS_CACHE=/app/models/.transformers
ENV HF_HOME=/app/models/.huggingface

# Copy requirements and install optimized dependencies
COPY requirements.txt .
COPY requirements-azure.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-azure.txt

# Install additional Azure-optimized dependencies
RUN pip install --no-cache-dir \
    azure-storage-blob==12.19.0 \
    azure-keyvault-secrets==4.7.0 \
    azure-identity==1.15.0 \
    azure-monitor-opentelemetry==1.2.0 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    gunicorn==21.2.0 \
    python-multipart==0.0.6 \
    redis==5.0.1 \
    psutil==5.9.6

# Copy the entire codebase
COPY . .

# Create optimized directories for model files and caching
RUN mkdir -p /app/models /app/data /app/outputs /app/cache /app/logs && \
    mkdir -p /app/models/.torch /app/models/.transformers /app/models/.huggingface && \
    chmod -R 755 /app

# Set up model caching and optimization
ENV MODEL_CACHE_DIR=/app/models
ENV RESULTS_CACHE_DIR=/app/cache
ENV LOG_LEVEL=INFO

# Pre-download commonly used models for faster startup
RUN python -c "\
import torch; \
from transformers import AutoTokenizer, AutoModel; \
import ssl; \
ssl._create_default_https_context = ssl._create_unverified_context; \
try: \
    print('Downloading BERT models...'); \
    AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='/app/models/.transformers'); \
    AutoModel.from_pretrained('bert-base-uncased', cache_dir='/app/models/.transformers'); \
    print('Models cached successfully'); \
except Exception as e: \
    print(f'Model caching failed: {e}'); \
"

# Expose ports for web API and monitoring
EXPOSE 8000 8080 9090

# Health check for Azure
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Optimize container startup
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command optimized for Azure
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "api_server:app", "--bind", "0.0.0.0:8000", "--timeout", "300", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "100", "--preload"]
