"""
Test baseline with different generation settings
"""

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import GPT2Tokenizer

def generate_improved(model, tokenizer, prompt, max_length=200, temperature=0.7, top_p=0.9, top_k=50):
    """Generate with nucleus sampling and top-k"""
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            
            # Temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Load baseline
config = MambaConfig(d_model=896, n_layer=16, vocab_size=50257)
model = MambaLMHeadModel(config)
checkpoint = torch.load('checkpoints/mamba_baseline/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Test with better sampling
prompt = "Sir Edmund was known throughout the kingdom as the bravest knight who ever lived."
print("Testing with temperature=0.7, top_p=0.9, top_k=50:")
print(generate_improved(model, tokenizer, prompt, temperature=0.7, top_p=0.9, top_k=50))

print("\n" + "="*60 + "\n")

print("Testing with temperature=0.5 (more conservative):")
print(generate_improved(model, tokenizer, prompt, temperature=0.5, top_p=0.9, top_k=40))