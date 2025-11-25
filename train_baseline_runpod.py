"""
Continue training baseline on Runpod
Loads existing checkpoint and trains for additional epochs
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from tqdm import tqdm
import json
import time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    config = MambaConfig(
        d_model=896,
        n_layer=16,
        vocab_size=50257,
    )
    
    batch_size = 32  # A100 can handle this
    learning_rate = 6e-4
    additional_epochs = 10  # Train 10 more epochs
    save_dir = 'checkpoints/mamba_baseline_continued'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model from checkpoint
    print("Loading baseline checkpoint...")
    model = MambaLMHeadModel(config)
    checkpoint = torch.load('checkpoints/mamba_baseline/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    start_epoch = checkpoint['epoch']
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous val loss: {checkpoint['val_loss']:.4f}")
    
    # Load data
    print("Loading datasets...")
    train_dataset = load_from_disk('data/processed/train')
    val_dataset = load_from_disk('data/processed/val')
    
    train_dataset.set_format(type='torch', columns=['input_ids'])
    val_dataset.set_format(type='torch', columns=['input_ids'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state")
    
    print(f"\n{'='*60}")
    print(f"Continuing training for {additional_epochs} more epochs")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    training_history = []
    
    for epoch in range(start_epoch + 1, start_epoch + additional_epochs + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
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
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"\nStep {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Avg: {avg_loss:.4f}")
        
        train_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
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
                
                total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_loader)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} - Time: {epoch_time/60:.2f} min")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Perplexity: {val_perplexity:.2f}")
        print(f"{'='*60}\n")
        
        # Save checkpoint every epoch
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
        }
        
        torch.save(checkpoint_data, f'{save_dir}/checkpoint_epoch{epoch}.pt')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_data, f'{save_dir}/best_model.pt')
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'epoch_time_minutes': epoch_time / 60
        })
        
        with open(f'{save_dir}/training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}/")

if __name__ == "__main__":
    main()