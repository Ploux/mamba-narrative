"""
Data preparation script for Project Gutenberg corpus
Subtask 2 from mamba_baseline.md
"""

import os
from datasets import load_dataset
from transformers import GPT2Tokenizer
import json

def download_gutenberg():
    """Download Project Gutenberg English books dataset"""
    print("Downloading Project Gutenberg English books...")
    print("This may take a few minutes...")
    
    # Load the dataset - this has ~48k English books with metadata removed
    dataset = load_dataset("sedthh/gutenberg_english")
    
    print(f"\nDataset loaded:")
    print(f"Train set: {len(dataset['train'])} books")
    
    # Show example
    example = dataset['train'][0]
    print(f"\nExample book preview (first 500 chars):")
    print(example['TEXT'][:500])
    
    print(f"\nBook metadata available:")
    print(f"Fields: {list(example.keys())}")
    
    return dataset

def initialize_tokenizer():
    """Initialize GPT-2 tokenizer"""
    print("\nInitializing GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Vocabulary size: {len(tokenizer)}")
    return tokenizer

def main():
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Step 1: Download dataset
    dataset = download_gutenberg()
    
    # Step 2: Initialize tokenizer
    tokenizer = initialize_tokenizer()
    
    # Save dataset info
    info = {
        'dataset': 'sedthh/gutenberg_english',
        'total_books': len(dataset['train']),
        'tokenizer': 'gpt2',
        'vocab_size': len(tokenizer),
        'notes': 'English books from Project Gutenberg with metadata removed'
    }
    
    with open('data/dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n✅ Data preparation initialized!")
    print(f"Total books available: {len(dataset['train']):,}")
    print("\nNext steps:")
    print("1. Examine the data quality")
    print("2. Filter for narrative fiction")
    print("3. Implement preprocessing pipeline")
    print("4. Tokenize and save processed data")

if __name__ == "__main__":
    main()
