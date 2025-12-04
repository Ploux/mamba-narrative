"""
Evaluate baseline model on continuation tasks
"""

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import GPT2Tokenizer
from continuation_eval import evaluate_model

def load_baseline(checkpoint_path='checkpoints/mamba_baseline/best_model_epoch7.pt'):
    config = MambaConfig(
        d_model=896,
        n_layer=16,
        vocab_size=50257,
    )
    
    model = MambaLMHeadModel(config)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    
    print(f"Loaded baseline from epoch {checkpoint['epoch']}")
    return model

if __name__ == "__main__":
    model = load_baseline()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    evaluate_model(model, tokenizer, 'eval_results_baseline.json', device='cuda')
    
    print("\nBaseline evaluation complete!")
    print("Results saved to eval_results_baseline.json")