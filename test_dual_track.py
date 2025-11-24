"""
Test dual-track Mamba implementation
"""

import torch
from mamba_ssm.models.config_mamba import MambaConfig
from dual_track_mamba import create_dual_track_model

def test_dual_track():
    """Test that dual-track model instantiates and runs"""
    
    # Create small config for testing
    config = MambaConfig(
        d_model=256,
        n_layer=4,
        vocab_size=50257,
    )
    
    # Create model
    print("Creating dual-track model...")
    model = create_dual_track_model(config, fast_dt_scale=2.0, slow_dt_scale=0.5)
    model = model.cuda()
    model.eval()
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).cuda()
    
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    assert logits.shape == (batch_size, seq_len, 50257)
    print("\nForward pass successful!")
    
    # Test generation
    print("\nTesting generation...")
    input_ids = torch.randint(0, 50257, (1, 10)).cuda()
    
    with torch.no_grad():
        for _ in range(50):
            output = model(input_ids)
            logits = output.logits
            next_token = torch.argmax(logits[0, -1, :])
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    print(f"Generated sequence length: {input_ids.shape[1]}")
    print("\nGeneration successful!")
    
    # Test track weights
    print("\nTrack combination weights:")
    for i, layer in enumerate(model.backbone['layers']):
        weights = torch.softmax(layer.mixer.track_weights, dim=0)
        print(f"Layer {i}: Fast={weights[0]:.3f}, Slow={weights[1]:.3f}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_dual_track()