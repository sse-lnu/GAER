# REQUIREMENTS

## Hardware Requirements

- CPU: 2 cores, 4+ recommended
- RAM: 8 GB minimum
- GPU: Optional NVIDIA GPU with CUDA 12.9 support (Compute Capability 7.5 or later). 
- Disk Space: Approximately 1 GB for the full datasets and source code

## Software Requirements

- Operating System: Linux, macOS, or Windows (with WSL2). macOS supports CPU only, Linux and Windows support CPU or CUDA
- Dependency Management: uv (https://github.com/astral-sh/uv) (others might work, but not tested)
- Docker: Optional, requires NVIDIA Container Toolkit for GPU support and an Intel x86 CPU
- Git LFS: Required to download large datasets

## Python Dependencies

All Python dependencies (including the Python runtime) are specified with explicit versioning in `pyproject.toml` or as part of the Dockerfiles.
