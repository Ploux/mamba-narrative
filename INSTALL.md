# Mamba-130M Baseline Installation Guide

This guide documents the working installation procedure for Mamba-130M on RTX 5090 with CUDA 12.8.

## System Requirements

- NVIDIA RTX 5090 (or other CUDA 12.0+ capable GPU)
- Ubuntu/Pop!_OS (tested on Ubuntu 22.04-based Pop!_OS)
- NVIDIA Driver 570.x or higher
- 16GB+ RAM recommended

## Installation Steps

### 1. Install Miniconda (if not already installed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2. Install CUDA 12.8 Toolkit
```bash
# Download CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run

# Run installer (UNCHECK driver, UNCHECK nvidia-fs)
sudo sh cuda_12.8.0_570.86.10_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version  # Should show 12.8
```

### 3. Create Conda Environment
```bash
conda create -n mamba_narrative python=3.10
conda activate mamba_narrative
```

### 4. Install PyTorch Nightly with CUDA 12.8
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 5. Install Mamba-SSM and Dependencies from Source

**CRITICAL**: Must install from source to compile against your exact PyTorch version.
```bash
# Install build dependencies
pip install ninja packaging

# Install causal-conv1d from source
pip install --no-cache-dir --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d.git

# Install mamba-ssm from source (takes 15-20 minutes)
pip install --no-cache-dir --no-build-isolation git+https://github.com/state-spaces/mamba.git
```

### 6. Install Additional Dependencies
```bash
pip install transformers datasets numpy
```

### 7. Verify Installation
```bash
# Test basic Mamba block
python test_mamba.py

# Test 130M model
python test_mamba_130m.py
```

## 130M Model Configuration

The working configuration for ~130M parameters:
```python
from mamba_ssm.models.config_mamba import MambaConfig

config = MambaConfig(
    d_model=896,
    n_layer=16,
    vocab_size=50257,
)
```

This yields **126.9M parameters**.

## Common Issues

### Issue: "CUDA capability sm_120 is not compatible"
**Solution**: You need CUDA 12.8+ and PyTorch nightly built against cu128.

### Issue: "mamba_ssm is only supported on CUDA 11.6 and above"
**Solution**: Check `nvcc --version` shows 12.8. Update your PATH if needed.

### Issue: "undefined symbol" when importing mamba_ssm
**Solution**: Reinstall from source with `--no-cache-dir --no-build-isolation` to recompile against your PyTorch.

### Issue: nvidia-fs installation fails
**Solution**: Skip nvidia-fs during CUDA installation (uncheck in installer or use `--toolkit` flag).

## GPU Memory Usage

- 130M model inference: ~0.5 GB
- Training (batch size 32): TBD during baseline experiments

## Next Steps

See `mamba_baseline.md` for data preparation and training procedures.