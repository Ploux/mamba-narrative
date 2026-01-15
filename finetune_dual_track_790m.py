"""
Fine-tune dual-track Mamba-790M on narrative continuation task
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
from dual_track_mamba import DualTrackMambaLMHeadModel, MambaConfig
from tqdm import tqdm
import json
import time
import sys
from torch.cuda.amp import autocast

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    log_file = 'finetune_dual_track_790m.log'
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    
    print(f"Logging to: {log_file}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 1 # OLD: 2
    learning_rate = 1e-4
    num_epochs = 10
    max_length = 1024
    
    save_dir = 'checkpoints/finetuned_dual_track_790m'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dual-track model
    print("Loading dual-track Mamba-790M...")
    checkpoint = torch.load('checkpoints/dual_track_790m_init/model.pt')
    
    config_dict = checkpoint['config']
    config = MambaConfig(
        d_model=config_dict['d_model'],
        n_layer=config_dict['n_layer'],
        vocab_size=config_dict['vocab_size'],
        ssm_cfg=config_dict.get('ssm_cfg', {})
    )
    
    model = DualTrackMambaLMHeadModel(
        config=config,
        fast_dt_scale=checkpoint['fast_dt_scale'],
        slow_dt_scale=checkpoint['slow_dt_scale']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model = model.half()  # Convert to FP16
    print(f"Model converted to FP16")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("\nLoading fine-tuning datasets...")
    train_dataset = load_from_disk('data/finetuning/train')
    val_dataset = load_from_disk('data/finetuning/val')
    
    def tokenize_function(examples):
        combined = [p + c for p, c in zip(examples['prompt'], examples['continuation'])]
        tokenized = tokenizer(
            combined,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=['prompt', 'continuation', 'source']
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=['prompt', 'continuation', 'source']
    )
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.1
    )
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning Dual-Track Mamba-790M for up to {num_epochs} epochs")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Fast track dt_scale: {checkpoint['fast_dt_scale']}")
    print(f"Slow track dt_scale: {checkpoint['slow_dt_scale']}")
    print(f"Estimated time per epoch: ~20-25 minutes")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():  # Add autocast context
                outputs = model(input_ids)
                logits = outputs.logits
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n!!! NaN/Inf detected at batch {batch_idx} !!!")
                break
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids)
                logits = outputs.logits
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n!!! NaN/Inf in validation !!!")
                    break
                
                total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_loader)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        epoch_time = time.time() - epoch_start
        
        # Get track weights from first layer
        track_weights = torch.softmax(model.backbone['layers'][0].mixer.track_weights, dim=0)
        fast_weight = track_weights[0].item()
        slow_weight = track_weights[1].item()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} - Time: {epoch_time/60:.2f} min")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Perplexity: {val_perplexity:.2f}")
        print(f"Track weights: Fast={fast_weight:.3f}, Slow={slow_weight:.3f}")
        print(f"{'='*60}\n")
        
        if torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
            print("!!! Training collapsed - stopping !!!")
            break
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'fast_weight': fast_weight,
            'slow_weight': slow_weight
        }
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
            'fast_dt_scale': checkpoint['fast_dt_scale'],
            'slow_dt_scale': checkpoint['slow_dt_scale'],
            **checkpoint_data
        }, f'{save_dir}/checkpoint_epoch{epoch}.pt')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'fast_dt_scale': checkpoint['fast_dt_scale'],
                'slow_dt_scale': checkpoint['slow_dt_scale'],
                **checkpoint_data
            }, f'{save_dir}/best_model.pt')
            print(f"*** Saved BEST model (val_loss: {val_loss:.4f}) ***")
        
        training_history.append(checkpoint_data)
        
        with open(f'{save_dir}/training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Early stopping
        if epoch > 3 and val_loss > best_val_loss + 0.1:
            print(f"\nEarly stopping - validation loss increasing")
            break
    
    print(f"\nFine-tuning complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()