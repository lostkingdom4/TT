# Docker Setup for PyTorch Intel GPU MLIR Extraction

This Docker setup provides a containerized environment for running the Triton kernel auto-tuning pipeline with NVIDIA GPU support on both Linux and Windows.

## Prerequisites

### Linux
- Docker Engine 20.10+ with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA 12.1 compatible driver

### Windows (WSL2)
- Windows 10/11 (Build 19044+ or Windows 11)
- WSL2 installed and enabled
- Docker Desktop 4.0+ with WSL2 backend **OR** Docker Engine installed directly in WSL2
- NVIDIA GPU with latest Game Ready or Studio drivers (525.60.11+)
- CUDA support in WSL2 (automatically available with latest drivers)

## Windows WSL2 Setup

### Option 1: Using Docker Desktop (Easiest)
1. Install Docker Desktop for Windows
2. Enable WSL2 integration in Docker Desktop settings
3. Open WSL2 terminal (Ubuntu/Debian)
4. Navigate to your project directory
5. Run Docker commands normally

### Option 2: Docker Engine in WSL2 (Native Performance)
1. **Install WSL2 Ubuntu** (if not already):
   ```powershell
   # In PowerShell (Admin)
   wsl --install -d Ubuntu-22.04
   ```

2. **Install Docker Engine in WSL2**:
   ```bash
   # Inside WSL2 terminal
   sudo apt-get update
   sudo apt-get install -y ca-certificates curl gnupg

   # Add Docker's official GPG key
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg

   # Set up repository
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

   # Install Docker Engine
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

   # Add your user to docker group (to avoid sudo)
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. **Install NVIDIA Container Toolkit**:
   ```bash
   # Configure NVIDIA GPG key
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   # Install toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit

   # Configure Docker to use NVIDIA runtime
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. **Verify GPU Access**:
   ```bash
   # Check if GPU is visible in WSL2
   nvidia-smi

   # Test Docker GPU access
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

5. **Clone/Access Your Project**:
   ```bash
   # Option A: Clone directly in WSL2 (better performance)
   cd ~
   git clone <your-repo-url>
   cd PyTorch_Intel_GPU_MLIR-Extraction

   # Option B: Access Windows files (slower due to I/O overhead)
   cd /mnt/c/Users/YourUsername/path/to/PyTorch_Intel_GPU_MLIR-Extraction
   ```

## Quick Start

### 1. Build the Docker Image

```bash
# Build the image
docker compose build

# Or build directly
docker build -t pytorch-intel-gpu-mlir:latest .
```

### 2. Run the Container

#### Using Docker Compose (Recommended)
```bash
# In WSL2 terminal (or Linux)
cd /path/to/PyTorch_Intel_GPU_MLIR-Extraction

# Start container in interactive mode
docker compose run --rm triton-kernel-tuning

# Or start in detached mode and attach
docker compose up -d
docker compose exec triton-kernel-tuning /bin/bash

# Stop when done
docker compose down
```

#### Using Docker Run Directly
```bash
# Linux / WSL2
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  pytorch-intel-gpu-mlir:latest

# Windows PowerShell (with Docker Desktop)
docker run --gpus all -it --rm `
  -v ${PWD}:/workspace `
  pytorch-intel-gpu-mlir:latest

# Windows CMD (with Docker Desktop)
docker run --gpus all -it --rm ^
  -v %cd%:/workspace ^
  pytorch-intel-gpu-mlir:latest
```

### 3. Verify GPU Access

Inside the container:
```bash
# Check NVIDIA GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check Triton
python -c "import triton; print(f'Triton version: {triton.__version__}')"
```

## Running the Pipeline

### Auto-tune a Single Kernel
```bash
python Data/tune_one_file.py Data/triton_modules/YOUR_KERNEL.py
```

### Auto-tune All Kernels
```bash
python Data/auto_tune_all.py --root ./runs/run_single --device cuda
```

### Extract PyTorch Modules
```bash
python operator_fusion.py --path ./runs/run_single
```

### Create Triton Dataset
```bash
python create_dataset.py --input_dir ./Data/pytorch_modules --output_dir ./Data/triton_modules
```

## Directory Structure

The container mounts your project directory at `/workspace` with persistent volumes for:
- `/workspace/Data/triton_cache` - Triton compilation cache
- `/workspace/Data/performance` - Performance results
- `/workspace/Data/tuning_results` - Auto-tuning results

## Troubleshooting

### GPU Not Detected

**Linux:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Windows WSL2:**
- Ensure WSL2 is enabled: `wsl --status` (in PowerShell)
- Verify GPU in WSL2: `nvidia-smi` (inside WSL2 terminal)
- Check NVIDIA driver version (Windows host): Should be 525.60.11 or newer
- If using Docker Desktop:
  - Update to latest version
  - Enable "Use WSL 2 based engine" in Settings > General
  - Enable WSL integration in Settings > Resources > WSL Integration
- If using Docker Engine in WSL2:
  - Restart Docker daemon: `sudo systemctl restart docker`
  - Verify NVIDIA runtime: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Permission Issues (Linux)

If you encounter permission errors with mounted volumes:
```bash
# Run container with your user ID
docker run --gpus all -it --rm \
  --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  pytorch-intel-gpu-mlir:latest
```

### Out of Memory Errors

Reduce batch sizes or adjust timeout settings:
```bash
# Modify timeout in auto_tune_all.py
python Data/auto_tune_all.py --root ./runs/run_single --device cuda --limit 10
```

### Triton Cache Issues

Clear the Triton cache:
```bash
# Inside container
rm -rf /workspace/Data/triton_cache/*

# Or delete the volume
docker volume rm pytorch-intel-gpu-mlir-extraction_triton-cache
```

## Performance Tips

1. **Use SSD for volumes**: Store data on fast storage
2. **Allocate sufficient RAM**: Recommended 16GB+ for Docker
3. **GPU Memory**: Monitor with `nvidia-smi` and adjust batch sizes
4. **Cache warming**: First run compiles kernels, subsequent runs are faster

## Development Workflow

### Edit Code on Host, Run in Container
Your code is mounted in real-time - any changes on your host machine are immediately available in the container.

```bash
# Terminal 1: Start container
docker compose up -d

# Terminal 2: Execute commands
docker compose exec triton-kernel-tuning python Data/tune_one_file.py ...

# When done
docker compose down
```

### WSL2 Performance Tips
1. **Store project in WSL2 filesystem** (`~/projects/...`) instead of Windows filesystem (`/mnt/c/...`)
   - 2-5x faster file I/O
   - Better Docker volume performance

2. **Access WSL2 from Windows Explorer**:
   - Type `\\wsl$\Ubuntu-22.04\home\your_username\` in Explorer address bar
   - Edit files with VS Code: `code .` in WSL2 terminal

3. **VS Code with WSL2**:
   - Install "WSL" extension in VS Code
   - Connect to WSL2: Click green corner icon > "Connect to WSL"
   - Open folder in WSL2 filesystem
   - Use integrated terminal (automatically uses WSL2)

### Persistent Data
Results are automatically saved to your host machine in:
- `Data/performance/`
- `Data/tuning_results/`
- `runs/`

## Cleanup

```bash
# Stop and remove containers
docker compose down

# Remove volumes (WARNING: deletes cached data)
docker compose down -v

# Remove image
docker rmi pytorch-intel-gpu-mlir:latest

# Clean up all Docker resources
docker system prune -a --volumes
```

## WSL2-Specific Troubleshooting

### Docker Daemon Not Running in WSL2
```bash
# Check if Docker is running
sudo systemctl status docker

# Start Docker service
sudo systemctl start docker

# Enable Docker to start automatically
sudo systemctl enable docker
```

### WSL2 Memory Issues
WSL2 can consume excessive memory. Create `.wslconfig` in Windows user directory:

```ini
# C:\Users\YourUsername\.wslconfig
[wsl2]
memory=16GB    # Limit WSL2 memory usage
processors=8   # Limit CPU cores
swap=8GB       # Swap file size
```

Restart WSL2: `wsl --shutdown` (in PowerShell), then reopen terminal.

### File Permission Issues in WSL2
```bash
# If you see permission errors with Docker volumes
# Option 1: Run with your user ID
docker run --gpus all -it --rm \
  --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  pytorch-intel-gpu-mlir:latest

# Option 2: Fix ownership after running container
sudo chown -R $USER:$USER Data/triton_cache Data/performance Data/tuning_results
```

### GPU Not Visible in WSL2
```bash
# 1. Check NVIDIA driver on Windows host (PowerShell as Admin)
nvidia-smi  # Should show driver version 525.60.11+

# 2. Check GPU in WSL2
nvidia-smi  # Should work without installing CUDA in WSL2

# 3. If nvidia-smi fails in WSL2
# Update Windows NVIDIA driver from nvidia.com

# 4. Verify CUDA in WSL2 (no installation needed)
ls -la /usr/lib/wsl/lib/libcuda.so*  # Should exist
```

## Building for Specific CUDA Version

To build for a different CUDA version, modify the base image in `Dockerfile`:

```dockerfile
# For CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

# Then install matching PyTorch
RUN pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Notes

- The container runs with Python 3.12 to match your environment
- All project dependencies from `requirements.txt` are installed
- CUDA 12.1 with cuDNN 8 is included
- Triton 3.3.1 is pre-installed with auto-tuning support
- Environment variables for Triton caching are pre-configured

## Support

For issues related to:
- Docker setup: Check Docker Desktop logs
- GPU access: Verify `nvidia-smi` works in WSL2 (Windows) or host (Linux)
- Pipeline errors: Check application logs in `/workspace`
