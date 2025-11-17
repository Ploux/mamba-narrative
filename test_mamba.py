import torch
from mamba_ssm import Mamba

# Test basic Mamba block
batch, length, dim = 2, 64, 128
x = torch.randn(batch, length, dim).cuda()

model = Mamba(
    d_model=dim,
    d_state=16,
    d_conv=4,
    expand=2,
).cuda()

y = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
assert y.shape == x.shape
print("✅ Mamba block test passed!")
