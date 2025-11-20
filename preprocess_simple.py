"""
Simple preprocessing - accumulate all data then save once
With 5000 books this should fit in your 192GB RAM easily
"""

import os
import json
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import random

def main():
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Initialize tokenizer
    print("Initializing GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading Project Gutenberg dataset...")
    dataset = load_dataset("sedthh/gutenberg_english", split="train")
    
    # Filter for substantial books
    print(f"\nFiltering {len(dataset)} books...")
    books = []
    for i, example in enumerate(tqdm(dataset)):
        text = example['TEXT']
        if len(text) > 10000:
            books.append({'text': text, 'book_id': i})
    
    # Sample 5000 books
    random.seed(42)
    random.shuffle(books)
    books = books[:5000]
    print(f"Using {len(books)} books")
    
    # Create splits
    n_train = int(len(books) * 0.9)
    n_val = int(len(books) * 0.05)
    
    train_books = books[:n_train]
    val_books = books[n_train:n_train+n_val]
    test_books = books[n_train+n_val:]
    
    print(f"\nTrain: {len(train_books)} books")
    print(f"Val: {len(val_books)} books")
    print(f"Test: {len(test_books)} books")
    
    # Process train split
    print("\n=== Tokenizing train split ===")
    train_sequences = []
    for book in tqdm(train_books):
        tokens = tokenizer.encode(book['text'])
        for i in range(0, len(tokens) - 1024, 1024):
            sequence = tokens[i:i+1024]
            if len(sequence) == 1024:
                train_sequences.append({'input_ids': sequence})
    
    print(f"Train sequences: {len(train_sequences):,}")
    train_dataset = Dataset.from_list(train_sequences)
    train_dataset.save_to_disk('data/processed/train')
    
    # Process val split
    print("\n=== Tokenizing validation split ===")
    val_sequences = []
    for book in tqdm(val_books):
        tokens = tokenizer.encode(book['text'])
        for i in range(0, len(tokens) - 1024, 1024):
            sequence = tokens[i:i+1024]
            if len(sequence) == 1024:
                val_sequences.append({'input_ids': sequence})
    
    print(f"Val sequences: {len(val_sequences):,}")
    val_dataset = Dataset.from_list(val_sequences)
    val_dataset.save_to_disk('data/processed/val')
    
    # Process test split
    print("\n=== Tokenizing test split ===")
    test_sequences = []
    for book in tqdm(test_books):
        tokens = tokenizer.encode(book['text'])
        for i in range(0, len(tokens) - 1024, 1024):
            sequence = tokens[i:i+1024]
            if len(sequence) == 1024:
                test_sequences.append({'input_ids': sequence})
    
    print(f"Test sequences: {len(test_sequences):,}")
    test_dataset = Dataset.from_list(test_sequences)
    test_dataset.save_to_disk('data/processed/test')
    
    # Save info
    info = {
        'tokenizer': 'gpt2',
        'vocab_size': len(tokenizer),
        'max_length': 1024,
        'train_sequences': len(train_sequences),
        'val_sequences': len(val_sequences),
        'test_sequences': len(test_sequences),
        'total_train_tokens': len(train_sequences) * 1024,
    }
    
    with open('data/processed/info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n✅ Complete!")
    print(f"Train: {len(train_sequences):,} sequences")
    print(f"Val: {len(val_sequences):,} sequences") 
    print(f"Test: {len(test_sequences):,} sequences")
    print(f"Total tokens: {len(train_sequences) * 1024:,}")

if __name__ == "__main__":
    main()