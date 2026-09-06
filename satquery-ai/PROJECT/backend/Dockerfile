# SatQuery AI - Backend Dockerfile
# Base: Lightweight Python 3.11 Slim
FROM python:3.11-slim

LABEL maintainer="SatQuery AI Team <team@satquery.ai>"
LABEL description="SatQuery AI FastAPI Backend Service for SIH 2026"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEMO_MODE=true \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies for OpenCV, GDAL/Rasterio, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend codebase
COPY . /app/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI application via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
