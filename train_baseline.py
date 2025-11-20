"""
Training script for Mamba-130M baseline on Project Gutenberg
Subtasks 3 & 4 from mamba_baseline.md
"""

import os
import json
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from tqdm import tqdm
import time
import gc
gc.collect()

def create_model():
    """Create Mamba-130M model"""
    config = MambaConfig(
        d_model=896,
        n_layer=16,
        vocab_size=50257,
    )
    
    model = MambaLMHeadModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    return model

def load_data(batch_size=32):
    """Load preprocessed data"""
    print("Loading datasets...")
    train_dataset = load_from_disk('data/processed/train')
    val_dataset = load_from_disk('data/processed/val')
    
    # Set format to torch
    train_dataset.set_format(type='torch', columns=['input_ids'])
    val_dataset.set_format(type='torch', columns=['input_ids'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Handle batch input_ids
        input_ids = batch['input_ids']
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in input_ids])
        input_ids = input_ids.to(device)
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Compute loss (shift for autoregressive prediction)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Log every 100 steps
        if batch_idx % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"\nStep {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Avg: {avg_loss:.4f}")
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Handle batch input_ids
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
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def main():
    # Configuration
    batch_size = 8
    learning_rate = 6e-4
    num_epochs = 3
    warmup_steps = 500
    save_dir = 'checkpoints/mamba_baseline'
    
    # Create checkpoint directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    model = model.to(device)
    
    # Load data
    train_loader, val_loader = load_data(batch_size)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {num_epochs} epochs")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss, val_perplexity = validate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log results
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs} - Time: {epoch_time/60:.2f} min")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Perplexity: {val_perplexity:.2f}")
        print(f"{'='*60}\n")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
        }
        
        torch.save(checkpoint, f'{save_dir}/checkpoint_epoch{epoch}.pt')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, f'{save_dir}/best_model.pt')
            print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        
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
    
    print("\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}/")

if __name__ == "__main__":
    main()