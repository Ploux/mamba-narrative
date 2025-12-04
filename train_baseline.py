"""
Continue training baseline from epoch 9 checkpoint (local)
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
from torch.optim.lr_scheduler import CosineAnnealingLR

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = MambaConfig(
        d_model=896,
        n_layer=16,
        vocab_size=50257,
    )
    
    batch_size = 8
    learning_rate = 3e-4  # Reduced from original 6e-4
    num_epochs = 50
    
    save_dir = 'checkpoints/mamba_baseline_continued'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load from LOCAL epoch 9 checkpoint
    print("Loading epoch 9 checkpoint (from local training)...")
    model = MambaLMHeadModel(config)
    checkpoint = torch.load('checkpoints/mamba_baseline_continued/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    start_epoch = checkpoint['epoch']
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous val loss: {checkpoint['val_loss']:.4f}")
    print(f"Previous val perplexity: {checkpoint.get('val_perplexity', 'N/A')}")
    
    # Load data
    print("Loading datasets...")
    train_dataset = load_from_disk('data/processed/train')
    val_dataset = load_from_disk('data/processed/val')
    
    train_dataset.set_format(type='torch', columns=['input_ids'])
    val_dataset.set_format(type='torch', columns=['input_ids'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Optimizer with reduced LR
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Load optimizer state (batch sizes match now)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Update learning rate in loaded optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        print(f"Loaded optimizer state and updated LR to {learning_rate:.2e}")
    else:
        print("No optimizer state in checkpoint, starting fresh")

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    print(f"\n{'='*60}")
    print(f"Training from epoch {start_epoch+1} to {start_epoch+num_epochs}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate:.2e}")
    print(f"This will take ~{num_epochs * 3.2:.0f} hours ({num_epochs * 3.2 / 24:.1f} days)")
    print(f"{'='*60}\n")
    
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    training_history = []
    
    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch+num_epochs}")
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
        
        train_loss = total_loss / len(train_loader)
        scheduler.step()
        
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
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} - Time: {epoch_time/60:.2f} min - LR: {current_lr:.2e}")
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

if __name__ == "__main__":
    main()