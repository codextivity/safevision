# Dockerfile
# CPU-only inference image — no GPU needed for serving
# GPU is only needed for training which runs locally

FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

# System dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch for inference
# Much smaller than GPU version — reduces image size by ~2GB
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/trained/ ./models/trained/

# Create required directories
RUN mkdir -p data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7860 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]