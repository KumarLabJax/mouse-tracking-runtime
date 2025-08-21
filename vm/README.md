# Container Support for TensorFlow and PyTorch

This repository supports both Docker and Singularity runtime environments with a unified approach to TensorFlow and PyTorch dependencies.

## Architecture Overview

We use a multi-stage container architecture that supports both ML frameworks:

1. **Base ML Image**: `vm/tf-pytorch/Dockerfile` - Contains both TensorFlow and PyTorch with CUDA support
2. **Application Image**: `Dockerfile` - Builds on the ML base and adds the mouse-tracking application
3. **Singularity Definition**: `vm/singularity.def` - Creates Singularity containers from the Docker images

## Docker Support

### Base ML Image (`vm/tf-pytorch/Dockerfile`)

The base image provides:
- **Python 3.10** runtime environment
- **PyTorch 2.5.1** with CUDA 12.6 support (`cu126`)
- **TensorFlow 2.19.0** with CUDA support
- Essential system dependencies (ffmpeg, libjpeg8-dev, etc.)

Key features:
- Uses PyTorch's official CUDA index for GPU acceleration
- TensorFlow includes bundled CUDA runtime via `tensorflow[and-cuda]`
- Both frameworks can coexist and utilize GPU resources
- Pinned versions prevent dependency conflicts

### Application Image (`Dockerfile`)

The main application container:
- Extends from `aberger4/mouse-tracking-base:python3.10-slim` (published ML base)
- Uses `uv` for fast Python package management
- Installs only runtime dependencies (excludes dev/test/lint groups)
- Provides `mouse-tracking-runtime` CLI as the main entrypoint

## Singularity Support

### Definition File (`vm/singularity.def`)

The Singularity container:
- Bootstraps from the Docker image `aberger4/mouse-tracking:python3.10-slim`
- Inherits all TensorFlow/PyTorch capabilities from the Docker base
- Copies model files into `/workspace/models/` during build
- Provides HPC-compatible runtime environment

### Building Singularity Images

```bash
singularity build mouse-tracking-runtime.sif vm/singularity.def
```

## Framework Compatibility

Both frameworks are configured to work together:

### GPU Access
- **Docker**: Uses NVIDIA runtime with `NVIDIA_VISIBLE_DEVICES=all`
- **Singularity**: Inherits GPU access from host system
- **CUDA**: Both frameworks use compatible CUDA versions (12.6)

### Model Runtimes
- **PyTorch**: Used for HRNet-based pose estimation models
- **TensorFlow**: Handles arena corners, segmentation, and identity tracking

## Usage Examples

### Docker
```bash
# Build and run the application container
docker build -t mouse-tracking-runtime .
docker run --gpus all mouse-tracking-runtime mouse-tracking-runtime --help
```

### Singularity
```bash
# Build and run the Singularity container
singularity build mouse-tracking-runtime.sif vm/singularity.def
singularity run --nv mouse-tracking-runtime.sif mouse-tracking-runtime --help
```

The `--nv` flag enables NVIDIA GPU support in Singularity environments.