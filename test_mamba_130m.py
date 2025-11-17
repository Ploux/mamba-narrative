from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import torch

# Try different configurations to hit ~130M
configs_to_try = [
    {"d_model": 896, "n_layer": 16, "name": "896d x 16L"},
    {"d_model": 1024, "n_layer": 12, "name": "1024d x 12L"},
    {"d_model": 768, "n_layer": 24, "name": "768d x 24L"},
]

for cfg in configs_to_try:
    config = MambaConfig(
        d_model=cfg["d_model"],
        n_layer=cfg["n_layer"],
        vocab_size=50257,
    )
    
    model = MambaLMHeadModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"{cfg['name']}: {total_params:,} ({total_params/1e6:.1f}M)")
    
    if 120_000_000 < total_params < 140_000_000:
        print(f"✅ Found ~130M config: d_model={cfg['d_model']}, n_layer={cfg['n_layer']}")
        
        # Test GPU forward pass
        print("\nTesting GPU forward pass...")
        model = model.cuda()
        dummy_input = torch.randint(0, 50257, (2, 128)).cuda()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.logits.shape}")
        print("✅ GPU forward pass successful!")
        
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        break
