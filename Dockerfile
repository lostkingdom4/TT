# Multi-stage Dockerfile for PyTorch Intel GPU MLIR Extraction
# Supports NVIDIA GPUs with CUDA 12.1
# Works on both Linux and Windows with Docker Desktop

# Stage 1: Base image with CUDA
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python environment
FROM base AS python-base

# Install Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Stage 3: Dependencies
FROM python-base AS dependencies

WORKDIR /tmp

# Copy only requirements file first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support first
RUN pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Triton
RUN pip install triton==3.3.1

# Install remaining dependencies
RUN pip install -r requirements.txt

# Stage 4: Final application
FROM dependencies AS app

# Create app directory
WORKDIR /workspace

# Copy application code
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/Data/triton_cache \
    /workspace/Data/performance \
    /workspace/Data/tuning_results \
    /workspace/runs

# Set Triton cache directory
ENV TRITON_CACHE_DIR=/workspace/Data/triton_cache \
    TRITON_PRINT_AUTOTUNING=1 \
    TRITON_CACHE_AUTOTUNING=1

# Verify CUDA is available
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Default command - drop into bash for interactive use
CMD ["/bin/bash"]
