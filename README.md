# Mamba Narrative Research

Research project exploring state-space models (Mamba) for long-form narrative generation.

## Project Goal

Develop specialized small language models for long-form fiction generation using Mamba architecture with narrative-specific modifications.

## Current Status

**Phase 1: Baseline Establishment - Complete**

Subtask 1: Environment Setup - Complete
- CUDA 12.8 toolkit installed
- PyTorch nightly with CUDA 12.8 support
- mamba-ssm and causal-conv1d compiled from source
- Mamba-130M configuration verified (d_model=896, n_layer=16, 126.9M parameters)
- RTX 5090 operational

Subtask 2: Data Preparation - Complete
- Project Gutenberg corpus downloaded (sedthh/gutenberg_english, 5,000 books)
- Preprocessing and tokenization complete
- Dataset: 502,357 training sequences, 25,311 validation, 26,887 test
- Total training tokens: 514,413,568 (~514M tokens)
- Sequence length: 1024 tokens

Subtask 3: Baseline Training - Complete
- 3 epochs completed in 9.5 hours on RTX 5090
- Batch size: 8, Learning rate: 6e-4
- Final validation perplexity: 19.50

### Baseline Training Results

| Epoch | Train Loss | Val Loss | Val Perplexity | Time (min) |
|-------|------------|----------|----------------|------------|
| 1     | 3.0991     | 3.0261   | 20.62          | 188.9      |
| 2     | 2.9272     | 2.9894   | 19.87          | 190.9      |
| 3     | 2.9008     | 2.9704   | 19.50          | 191.7      |

See [generations_20251119.md](generations_20251119.md) for example outputs from the trained model.

**Phase 2: Multi-Track Architecture - Next**

Implement narrative-specific architectural modifications as described in [research_plan.md](research_plan.md):
- Triple-track SSM blocks (character/plot/scene timescales)
- Narrative-aware routing mechanism
- Cross-track attention for information exchange

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
- `train_baseline.py` - Training script for baseline model
- `generate_text.py` - Text generation inference script
- `generations_20251119.md` - Example generations from trained baseline

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