"""
Convert pretrained Mamba-2.8B to dual-track architecture
Both tracks initialized from pretrained weights, but with different dt_scale
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from dual_track_mamba import create_dual_track_from_pretrained
import sys

def main():
    print("="*60)
    print("Converting Mamba-2.8B to Dual-Track Architecture")
    print("="*60)
    
    # Load pretrained model
    print("\n1. Loading pretrained Mamba-2.8B...")
    pretrained = AutoModelForCausalLM.from_pretrained(
        "state-spaces/mamba-2.8b-hf",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    print(f"   Pretrained parameters: {sum(p.numel() for p in pretrained.parameters()):,}")
    
    # Convert to dual-track
    print("\n2. Creating dual-track architecture...")
    print("   Fast track: dt_scale = 2.0 (rapid forgetting)")
    print("   Slow track: dt_scale = 0.5 (persistent memory)")
    
    dual_track = create_dual_track_from_pretrained(
        pretrained,
        fast_dt_scale=2.0,
        slow_dt_scale=0.5
    )
    
    print(f"   Dual-track parameters: {sum(p.numel() for p in dual_track.parameters()):,}")
    
    # Save converted model
    print("\n3. Saving dual-track model...")
    import os
    save_path = "checkpoints/dual_track_2800m_init"
    os.makedirs(save_path, exist_ok=True)
    
    torch.save({
        'model_state_dict': dual_track.state_dict(),
        'config': dual_track.config.__dict__,
        'fast_dt_scale': 2.0,
        'slow_dt_scale': 0.5
    }, f'{save_path}/model.pt')
    
    print(f"\n✓ Conversion complete!")
    print(f"✓ Dual-track model saved to: {save_path}")
    print(f"\nNext step: Fine-tune with finetune_dual_track_2800m.py")

if __name__ == "__main__":
    main()