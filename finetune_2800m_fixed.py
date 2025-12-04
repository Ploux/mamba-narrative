"""
Fine-tune Mamba-2.8B with proper mixed precision and logging
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import time
import sys

# Redirect output to log file AND console
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
    # Setup logging
    log_file = 'finetune_2800m.log'
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    
    print(f"Logging to: {log_file}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters - ADJUSTED FOR STABILITY
    batch_size = 1
    accumulation_steps = 2
    learning_rate = 5e-5  # REDUCED from 1e-4
    num_epochs = 8
    max_length = 1024
    
    save_dir = 'checkpoints/finetuned_2800m'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print("Loading pretrained Mamba-2.8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "state-spaces/mamba-2.8b-hf",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model = model.to(device)

    # Enable gradient checkpointing (trades compute for memory)
    model.gradient_checkpointing_enable()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("\nLoading fine-tuning datasets...")
    train_dataset = load_from_disk('data/finetuning/train')
    val_dataset = load_from_disk('data/finetuning/val')
    
    # Tokenize function
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
    
    # Tokenize datasets
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
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning Mamba-2.8B for up to {num_epochs} epochs")
    print(f"Batch size: {batch_size}, Accumulation: {accumulation_steps}")
    print(f"Learning rate: {learning_rate} (REDUCED for stability)")
    print(f"Precision: FP32 (more stable than FP16)")
    print(f"Gradient clipping: 0.5 (aggressive)")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / accumulation_steps
            
            # Check for NaN BEFORE backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n!!! NaN/Inf detected at batch {batch_idx} !!!")
                print(f"Stopping training at epoch {epoch}")
                break
            
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Aggressive clipping
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        train_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n!!! NaN/Inf in validation !!!")
                    break
                
                total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_loader)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} - Time: {epoch_time/60:.2f} min")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Perplexity: {val_perplexity:.2f}")
        print(f"{'='*60}\n")
        
        # Check for NaN in results
        if torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
            print("!!! Training collapsed to NaN/Inf - stopping !!!")
            break
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
        }
        
        model.save_pretrained(f'{save_dir}/checkpoint_epoch{epoch}')
        print(f"Saved checkpoint: epoch{epoch}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(f'{save_dir}/best_model')
            print(f"*** Saved BEST model (val_loss: {val_loss:.4f}) ***")
        
        # Save history
        training_history.append(checkpoint_data)
        
        with open(f'{save_dir}/training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Early stopping
        if epoch > 2 and val_loss > best_val_loss + 0.1:
            print(f"\nEarly stopping - validation loss increasing")
            break
    
    print(f"\nFine-tuning complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()