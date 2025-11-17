#!/bin/bash
# Quick setup script for Mamba-130M baseline
# Run this after completing CUDA 12.8 installation

set -e  # Exit on error

echo "Creating conda environment..."
conda create -n mamba_narrative python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mamba_narrative

echo "Installing PyTorch nightly..."
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

echo "Installing build dependencies..."
pip install ninja packaging

echo "Installing causal-conv1d from source (this will take a few minutes)..."
pip install --no-cache-dir --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d.git

echo "Installing mamba-ssm from source (this will take 15-20 minutes)..."
pip install --no-cache-dir --no-build-isolation git+https://github.com/state-spaces/mamba.git

echo "Installing additional dependencies..."
pip install transformers datasets numpy

echo "✅ Installation complete!"
echo "Run 'python test_mamba.py' to verify."