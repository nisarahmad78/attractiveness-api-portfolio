# ===========================================
# 📦 Attractiveness API - Dockerfile
# ✅ Fixed OpenCV runtime dependencies (libGL)
# ✅ Uses Python 3.11 slim base (small size)
# ✅ Properly exposes FastAPI on port 8000
# ✅ Includes models folder (YuNet ONNX)
# ===========================================

# Base image: Python 3.11 slim
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies (includes OpenCV runtime libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt . 
COPY config ./config

# Install Python dependencies (no cache for smaller image)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code into container
COPY app ./app

# ✅ Copy models folder into container
COPY models ./models

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app with Uvicorn
# Use 0.0.0.0 so it can be accessed from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
