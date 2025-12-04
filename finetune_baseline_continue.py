"""
Continue fine-tuning baseline from epoch 5
Run overnight for 20 more epochs
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
    batch_size = 4
    learning_rate = 5e-5  # Lower LR for continued training
    num_epochs = 20  # Train 20 more epochs
    max_length = 1024
    
    save_dir = 'checkpoints/finetuned_baseline'
    
    # Load model from epoch 5
    print("Loading model from epoch 5...")
    model = AutoModelForCausalLM.from_pretrained(
        f"{save_dir}/checkpoint_epoch5",
        trust_remote_code=True
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load fine-tuning data
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
    
    # Optimizer with lower learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\n{'='*60}")
    print(f"Continuing training from epoch 5 for {num_epochs} more epochs")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate} (reduced for stability)")
    print(f"Estimated time: ~{num_epochs * 4:.0f} minutes (~{num_epochs * 4 / 60:.1f} hours)")
    print(f"{'='*60}\n")
    
    # Load training history
    history_file = f'{save_dir}/training_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            training_history = json.load(f)
        best_val_loss = min(h['val_loss'] for h in training_history)
        start_epoch = training_history[-1]['epoch']
    else:
        training_history = []
        best_val_loss = float('inf')
        start_epoch = 5
    
    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
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
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            model.save_pretrained(f'{save_dir}/checkpoint_epoch{epoch}')
            print(f"Saved checkpoint at epoch {epoch}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(f'{save_dir}/best_model')
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save history
        checkpoint_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
        }
        training_history.append(checkpoint_data)
        
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    print(f"\nFine-tuning complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {save_dir}/best_model")

if __name__ == "__main__":
    main()