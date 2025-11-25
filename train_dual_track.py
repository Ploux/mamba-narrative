"""
Train small dual-track model for quick validation
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import GPT2Tokenizer
from tqdm import tqdm
import json
from dual_track_mamba import create_dual_track_model
from continuation_eval import evaluate_model

def train_small_dual_track():
    """
    Train a smaller dual-track model with gradient accumulation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small config for memory constraints
    config = MambaConfig(
        d_model=256,
        n_layer=6,
        vocab_size=50257,
    )
    
    # Create model
    print("Creating dual-track model...")
    model = create_dual_track_model(config, fast_dt_scale=2.0, slow_dt_scale=0.5)
    model = model.to(device)
    
    # Load data
    print("\nLoading data...")
    train_dataset = load_from_disk('data/processed/train')
    val_dataset = load_from_disk('data/processed/val')
    
    train_dataset.set_format(type='torch', columns=['input_ids'])
    val_dataset.set_format(type='torch', columns=['input_ids'])
    
    # Batch size 2 with gradient accumulation
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Training with gradient accumulation
    num_epochs = 2
    accumulation_steps = 2  # Effective batch size = 2 * 2 = 4
    print(f"\nTraining dual-track model for {num_epochs} epochs...")
    print(f"Batch size: 2, Gradient accumulation: {accumulation_steps}, Effective batch: 4")
    print(f"This will take ~2-3 hours per epoch")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids']
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in input_ids])
            input_ids = input_ids.to(device)
            
            outputs = model(input_ids)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} - Training loss: {avg_loss:.4f}")
    
    # Save
    os.makedirs('checkpoints/dual_track_quick', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_loss': avg_loss
    }, 'checkpoints/dual_track_quick/model.pt')
    
    print("Model saved to checkpoints/dual_track_quick/model.pt")
    
    # Evaluate
    print("\nRunning continuation evaluation...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    evaluate_model(model, tokenizer, 'eval_results_dual_track_quick.json', device)
    
    print("\nDone! Now run evaluation on baseline for comparison.")

if __name__ == "__main__":
    train_small_dual_track()