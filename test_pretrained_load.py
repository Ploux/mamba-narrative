"""
Test loading pretrained Mamba model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Attempting to load pretrained Mamba-130M...")

try:
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "state-spaces/mamba-130m-hf",
        trust_remote_code=True
    )
    
    # Load tokenizer (Mamba uses GPT-NeoX tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Move to GPU
    model = model.cuda()
    
    # Test generation
    text = "Once upon a time, in a kingdom far away,"
    print(f"\n✓ Testing generation...")
    print(f"  Prompt: {text}")
    
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.8)
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n✓ Generated text:")
    print(f"  {generated}")
    
    print("\n✓ All tests passed! Ready for fine-tuning.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()