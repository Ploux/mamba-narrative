# Mamba Narrative Research

Research project exploring state-space models (Mamba) for long-form narrative generation.

## Project Goal

Develop specialized small language models for long-form fiction generation using Mamba architecture with narrative-specific modifications.

## Current Status

**Phase 1: Baseline Establishment**

Subtask 1: Environment Setup - Complete
- CUDA 12.8 toolkit installed
- PyTorch nightly with CUDA 12.8 support
- mamba-ssm and causal-conv1d compiled from source
- Mamba-130M configuration verified (d_model=896, n_layer=16, 126.9M parameters)
- RTX 5090 operational

Subtask 2: Data Preparation - In Progress
- Project Gutenberg corpus downloaded (sedthh/gutenberg_english, 48k books)
- Preprocessing and tokenization pipeline running
- Creating train/validation/test splits with 1024-token sequences

Subtask 3: Training Configuration - Next
- Configure training parameters for narrative generation
- Implement baseline training loop
- Profile memory usage and generation quality

## Installation

See [INSTALL.md](INSTALL.md) for detailed setup instructions.

Quick start:
```bash
# After installing CUDA 12.8
./setup.sh
python test_mamba_130m.py
```

## Project Structure

- `INSTALL.md` - Complete installation guide
- `mamba_baseline.md` - Baseline establishment plan
- `research_plan.md` - Full research roadmap
- `test_mamba.py` - Basic Mamba block verification
- `test_mamba_130m.py` - 130M model verification

## Hardware

- NVIDIA RTX 5090 (32GB VRAM)
- CUDA 12.8
- PyTorch nightly (cu128)

## References

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [Mamba GitHub](https://github.com/state-spaces/mamba)
- [Memory-Augmented Transformers](https://arxiv.org/html/2508.10824v1)
- [Advancing Transformer Architecture in Long-Context Large Language Models](https://arxiv.org/html/2311.12351v2)
- See `references.md` for more papers on LLMs and narrative generation.