# Self-Hosted GitHub Actions Runner with GPU Support

This guide explains how to set up a self-hosted GitHub Actions runner with NVIDIA GPU support for SwipeFlix model training.

## Prerequisites

- Machine with NVIDIA GPU (GTX 1060 or better recommended)
- Ubuntu 20.04+ or similar Linux distribution
- Docker installed
- NVIDIA drivers installed

## Step 1: Install NVIDIA Drivers

### Check Current GPU

```bash
lspci | grep -i nvidia
```

### Install Drivers (Ubuntu)

```bash
# Add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-525

# Reboot
sudo reboot

# Verify installation
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
...
```

## Step 2: Install Docker with NVIDIA Runtime

### Install Docker

```bash
# Uninstall old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
```

### Install NVIDIA Container Toolkit

```bash
# Add NVIDIA GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

# Add repository
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Test GPU in Docker

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Step 3: Set Up GitHub Actions Runner

### Create Runner Directory

```bash
mkdir -p ~/actions-runner
cd ~/actions-runner
```

### Download Runner

Go to your repository on GitHub:
1. Settings → Actions → Runners
2. Click "New self-hosted runner"
3. Select Linux x64
4. Copy the download and configure commands

```bash
# Download (example - use actual link from GitHub)
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
```

### Configure Runner

```bash
# Configure (use actual token from GitHub)
./config.sh --url https://github.com/yourusername/swipeflix \
  --token YOUR_REGISTRATION_TOKEN \
  --labels gpu \
  --name swipeflix-gpu-runner
```

Configuration options:
- **Name**: `swipeflix-gpu-runner`
- **Labels**: `gpu` (important for targeting this runner)
- **Work folder**: Default `_work`

### Install as Service

```bash
# Install service
sudo ./svc.sh install

# Start service
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

### Verify Runner

Go to GitHub repository → Settings → Actions → Runners

You should see your runner listed as "Idle".

## Step 4: Configure CI Workflow

The `.github/workflows/ci.yml` already includes a GPU training job:

```yaml
train_gpu:
  name: Train Model (GPU)
  runs-on: [self-hosted, gpu]
  steps:
    - name: Check GPU availability
      run: nvidia-smi

    - name: Build GPU Docker image
      run: docker build -f Dockerfile.gpu -t swipeflix:gpu .

    - name: Train model on GPU
      run: |
        docker run --gpus all \
          -v $(pwd)/data:/app/data \
          -v $(pwd)/mlruns:/app/mlruns \
          swipeflix:gpu \
          python -m src.swipeflix.ml.train --sample-size 5000 --seed 42
```

This job will:
- Only run on runners with the `gpu` label
- Use the GPU-enabled Dockerfile
- Train model with GPU acceleration

## Step 5: Test GPU Runner

### Trigger Workflow

```bash
# Push to main branch
git push origin main
```

### Monitor Execution

1. Go to GitHub repository → Actions
2. Find your workflow run
3. Check the `train_gpu` job
4. View logs to see GPU usage

Expected log output:
```
Run nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
...
```

## Troubleshooting

### Runner Not Showing in GitHub

```bash
# Check service status
sudo ./svc.sh status

# View logs
sudo journalctl -u actions.runner.* -f

# Restart service
sudo ./svc.sh restart
```

### GPU Not Available in Docker

```bash
# Check nvidia-docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-container-toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

### Permission Issues

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Fix runner directory permissions
sudo chown -R $USER:$USER ~/actions-runner
```

### CUDA Version Mismatch

Check CUDA compatibility:
```bash
nvidia-smi  # Shows max supported CUDA version

# Update Dockerfile.gpu to use compatible CUDA version
# Example: nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

## Maintenance

### Update Runner

```bash
cd ~/actions-runner
sudo ./svc.sh stop
./config.sh remove --token YOUR_REMOVE_TOKEN
# Download new version
# Reconfigure
sudo ./svc.sh install
sudo ./svc.sh start
```

### Monitor GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use nvtop
sudo apt install nvtop
nvtop
```

### Clean Up Docker Images

```bash
# Remove old images
docker image prune -a

# Remove stopped containers
docker container prune
```

## Security Considerations

1. **Firewall**: Ensure runner machine is behind firewall
2. **Updates**: Keep drivers and system updated
3. **Isolation**: Use Docker for isolation
4. **Secrets**: Don't log secrets in workflows
5. **Access**: Limit who can trigger workflows

## Cost Optimization

### Use Conditional Jobs

```yaml
train_gpu:
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

### Schedule Training

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
```

## Performance Benchmarks

Expected training times on different GPUs:

| GPU | Dataset Size | Training Time |
|-----|--------------|---------------|
| GTX 1060 | 10K users | ~5 minutes |
| RTX 3060 | 10K users | ~2 minutes |
| RTX 3090 | 10K users | ~1 minute |
| A100 | 10K users | ~30 seconds |

## Advanced Configuration

### Multiple GPU Support

```bash
# Use all GPUs
docker run --gpus all ...

# Use specific GPU
docker run --gpus '"device=0"' ...

# Use multiple specific GPUs
docker run --gpus '"device=0,1"' ...
```

### Resource Limits

```yaml
- name: Train with resource limits
  run: |
    docker run --gpus all \
      --memory=16g \
      --cpus=8 \
      swipeflix:gpu python -m src.swipeflix.ml.train
```

## References

- [GitHub Actions Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## Support

For issues:
1. Check logs: `sudo journalctl -u actions.runner.* -f`
2. Open GitHub issue with logs