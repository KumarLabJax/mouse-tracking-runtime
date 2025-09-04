FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Upgrade pip/wheel
RUN python -m pip install --upgrade pip wheel

RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
        vim \
        ffmpeg \
        libjpeg8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# --- Versions ---
ARG TORCH_VER=2.5.1
ARG TORCHVISION_VER=0.20.1
ARG TORCHAUDIO_VER=2.5.1
ARG TORCH_CUDA_TAG=cu126
ARG TENSORFLOW_VER=2.19.0

# Install PyTorch + CUDA (bundled runtime)
RUN pip install \
      --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
      torch==${TORCH_VER} torchvision==${TORCHVISION_VER} torchaudio==${TORCHAUDIO_VER}

# Install TensorFlow GPU (bundled CUDA)
RUN pip install "tensorflow[and-cuda]==${TENSORFLOW_VER}"

WORKDIR /workspace

COPY LICENSE ./
