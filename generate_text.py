"""
Test text generation from trained Mamba baseline
"""

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import GPT2Tokenizer

def load_model(checkpoint_path, device='cuda'):
    """Load trained model"""
    # Create config
    config = MambaConfig(
        d_model=896,
        n_layer=16,
        vocab_size=50257,
    )
    
    # Create model
    model = MambaLMHeadModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")
    print(f"Val perplexity: {checkpoint['val_perplexity']:.2f}")
    
    return model

def generate_text(model, tokenizer, prompt, max_length=500, temperature=0.8):
    """Generate text from prompt"""
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Get next token logits and apply temperature
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Sample from distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we hit end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = load_model('checkpoints/mamba_baseline/best_model.pt', device)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Test prompts
    prompts = [
        "It was a dark and stormy night when",
        "The old mansion stood at the end of the lane,",
        "She walked into the room and",
        "In the year 1850, a young man named",
    ]
    
    print(f"\n{'='*60}")
    print("GENERATING TEXT FROM MAMBA BASELINE")
    print(f"{'='*60}")
    
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=300, temperature=0.8)
        print(generated)
        print(f"\n{'-'*60}\n")

if __name__ == "__main__":
    main()