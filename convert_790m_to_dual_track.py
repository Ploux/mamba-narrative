"""
Convert pretrained Mamba-790M to dual-track architecture
"""

import torch
from transformers import AutoModelForCausalLM
from dual_track_mamba import create_dual_track_from_pretrained

def main():
    print("="*60)
    print("Converting Mamba-790M to Dual-Track Architecture")
    print("="*60)
    
    # Load pretrained model
    print("\n1. Loading pretrained Mamba-790M...")
    pretrained = AutoModelForCausalLM.from_pretrained(
        "state-spaces/mamba-790m-hf",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    print(f"   Pretrained parameters: {sum(p.numel() for p in pretrained.parameters()):,}")
    
    # Convert to dual-track
    print("\n2. Creating dual-track architecture...")
    print("   Fast track: dt_scale = 2.0")
    print("   Slow track: dt_scale = 0.5")
    
    dual_track = create_dual_track_from_pretrained(
        pretrained,
        fast_dt_scale=2.0,
        slow_dt_scale=0.5
    )
    
    print(f"   Dual-track parameters: {sum(p.numel() for p in dual_track.parameters()):,}")
    
    # Save converted model
    print("\n3. Saving dual-track model...")
    import os
    save_path = "checkpoints/dual_track_790m_init"
    os.makedirs(save_path, exist_ok=True)
    
    torch.save({
        'model_state_dict': dual_track.state_dict(),
        'config': dual_track.config.__dict__,
        'fast_dt_scale': 2.0,
        'slow_dt_scale': 0.5
    }, f'{save_path}/model.pt')
    
    print(f"\n✓ Conversion complete!")
    print(f"✓ Dual-track model saved to: {save_path}")

if __name__ == "__main__":
    main()