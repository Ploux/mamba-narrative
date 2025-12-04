"""
Fine-tune pretrained Mamba-2.8B on narrative continuation task
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 1  # Maximum for 2.8B on 32GB
    accumulation_steps = 2  # Effective batch size = 2
    learning_rate = 1e-4
    num_epochs = 8  # Conservative, expect early stopping
    max_length = 1024
    
    save_dir = 'checkpoints/finetuned_2800m'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load pretrained 2.8B model
    print("Loading pretrained Mamba-2.8B...")
    print("(This may take a few minutes...)")
    model = AutoModelForCausalLM.from_pretrained(
        "state-spaces/mamba-2.8b-hf",
        trust_remote_code=True,
        torch_dtype=torch.float16  # Use fp16 to save memory
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model loaded in FP16 precision for memory efficiency")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load fine-tuning data
    print("\nLoading fine-tuning datasets...")
    train_dataset = load_from_disk('data/finetuning/train')
    val_dataset = load_from_disk('data/finetuning/val')
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning Mamba-2.8B for up to {num_epochs} epochs")
    print(f"Batch size: {batch_size} with gradient accumulation: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Precision: FP16")
    print(f"Estimated time per epoch: ~60-80 minutes")
    print(f"Total estimated time: ~8-10 hours")
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
            loss = outputs.loss / accumulation_steps  # Scale loss
            
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                
                total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_loader)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} - Time: {epoch_time/60:.2f} min")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Perplexity: {val_perplexity:.2f}")
        print(f"{'='*60}\n")
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
        }
        
        model.save_pretrained(f'{save_dir}/checkpoint_epoch{epoch}')
        print(f"Checkpoint saved: epoch{epoch}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(f'{save_dir}/best_model')
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save history
        training_history.append(checkpoint_data)
        
        with open(f'{save_dir}/training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Early stopping if overfitting
        if epoch > 2 and val_loss > best_val_loss + 0.1:
            print(f"\nEarly stopping at epoch {epoch} - validation loss increasing")
            break
    
    print(f"\nFine-tuning complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {save_dir}/best_model")

if __name__ == "__main__":
    main()