## Establishing Mamba-130M Baseline on Project Gutenberg

### Subtasks Overview

1. Set up Mamba development environment and verify baseline model runs
2. Prepare and preprocess Project Gutenberg corpus for narrative-focused training
3. Configure training parameters appropriate for narrative generation
4. Train baseline Mamba-130M on prepared corpus
5. Implement evaluation metrics for narrative quality
6. Document baseline performance for comparison

### Subtask 1: Environment Setup and Model Verification

**Objective:** Establish a working Mamba installation and verify the 130M parameter model runs correctly before investing time in data preparation.

**Step 1.1: Core Dependencies**

The Mamba implementation requires specific versions due to its custom CUDA kernels:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

```bash
# Create fresh conda environment
conda create -n mamba_narrative python=3.10
conda activate mamba_narrative

nvidia-smi # reveals CUDA 13.0

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# verify_5090.py
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version PyTorch built with: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

```
pip uninstall torch torchvision torchaudio -y

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

```
# Install Mamba and its dependencies
pip install mamba-ssm
pip install causal-conv1d>=1.1.0
pip install transformers datasets
```

**Step 1.2: Verify Mamba Installation**

Test that the selective scan CUDA kernels compile correctly:

```python
# test_mamba.py
import torch
from mamba_ssm import Mamba

# Test basic Mamba block
batch, length, dim = 2, 64, 128
x = torch.randn(batch, length, dim).cuda()

model = Mamba(
    d_model=dim,  # Model dimension d_model
    d_state=16,   # SSM state expansion factor
    d_conv=4,     # Local convolution width
    expand=2,     # Block expansion factor
).cuda()

y = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
assert y.shape == x.shape
print("Mamba block test passed!")
```

**Step 1.3: Clone and Explore Mamba Repository**

The official repository contains training scripts that need modification:

```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba

# Examine the model configurations
ls configs/experiment/
# Look for s4-130m.yaml or similar config files
```

**Step 1.4: Verify 130M Configuration**

Locate or create the 130M parameter configuration. If not available, create based on the architecture pattern:

```yaml
# configs/experiment/mamba-130m.yaml
model:
  d_model: 768
  n_layer: 12
  vocab_size: 50257  # GPT-2 tokenizer size
  ssm_cfg:
    d_state: 16
    d_conv: 4
    expand: 2

training:
  batch_size: 32  # Adjust based on available VRAM
  learning_rate: 6e-4
  weight_decay: 0.1
  warmup_steps: 500
```

**Step 1.5: Test Model Instantiation**

Create a minimal script to verify the 130M model instantiates correctly:

```python
# test_mamba_130m.py
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

config = {
    'd_model': 768,
    'n_layer': 12,
    'vocab_size': 50257,
    'ssm_cfg': {},
    'rms_norm': True,
    'residual_in_fp32': True,
    'fused_add_norm': True,
}

model = MambaLMHeadModel(**config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Approximate size: {total_params / 1e6:.1f}M")

# Should be approximately 130M parameters
assert 120_000_000 < total_params < 140_000_000
```

**Common Issues and Solutions:**

1. **CUDA Kernel Compilation Errors**

    - Ensure CUDA toolkit matches PyTorch CUDA version
    - May need to install ninja: `pip install ninja`
    - On clusters, may need to module load appropriate CUDA version
2. **Out of Memory on Model Instantiation**

    - Reduce batch size in configuration
    - Use gradient checkpointing if available
    - Consider starting with smaller model for testing (70M parameters)
3. **Import Errors for mamba_ssm**

    - Install from source if pip version fails:

    ```bash
    git clone https://github.com/state-spaces/mamba.git
    cd mamba
    pip install -e .
    ```

4. **Missing Configuration Files**

    - The repository structure changes frequently
    - May need to adapt configs from similar-sized models
    - Check closed issues on GitHub for community configurations

**Verification Checkpoint:**

Before proceeding to data preparation, ensure:

- [ ] Mamba CUDA kernels compile and run
- [ ] 130M model instantiates without errors
- [ ] Forward pass completes on dummy data
- [ ] Memory usage is within available VRAM limits
- [ ] Training script location identified and accessible

**Next Steps Preview:**

Once environment verification completes, Subtask 2 will involve:

- Downloading Project Gutenberg books via standardized datasets
- Preprocessing to extract narrative-only content (removing metadata)
- Creating train/validation splits preserving book boundaries
- Tokenization with GPT-2 tokenizer for compatibility

The focus remains on establishing a working baseline before implementing novel architectural modifications. This methodical approach ensures a solid foundation for the research contributions.
