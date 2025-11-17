"""
Preprocess and tokenize Project Gutenberg data
Creates training-ready datasets for Mamba baseline
"""

import os
import json
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import random

def load_and_filter_data():
    """Load dataset and filter for fiction"""
    print("Loading Project Gutenberg dataset...")
    dataset = load_dataset("sedthh/gutenberg_english", split="train")
    
    print(f"Total books: {len(dataset)}")
    
    # Filter for books that are likely fiction based on metadata
    # The dataset has 'METADATA' field with genre info
    print("\nFiltering for fiction books...")
    
    fiction_books = []
    for i, example in enumerate(tqdm(dataset)):
        # Keep books that are long enough (>10k chars) and look like narrative
        text = example['TEXT']
        if len(text) > 10000:  # Filter out very short books
            fiction_books.append({
                'text': text,
                'book_id': i
            })
    
    print(f"Filtered to {len(fiction_books)} substantial books (>10k chars)")
    
    return fiction_books

def create_splits(books, train_ratio=0.9, val_ratio=0.05):
    """Create train/validation/test splits preserving book boundaries"""
    random.seed(42)
    random.shuffle(books)
    
    n_books = len(books)
    n_train = int(n_books * train_ratio)
    n_val = int(n_books * val_ratio)
    
    train_books = books[:n_train]
    val_books = books[n_train:n_train+n_val]
    test_books = books[n_train+n_val:]
    
    print(f"\nSplit statistics:")
    print(f"Train: {len(train_books)} books")
    print(f"Validation: {len(val_books)} books")
    print(f"Test: {len(test_books)} books")
    
    return train_books, val_books, test_books

def tokenize_books(books, tokenizer, max_length=1024):
    """Tokenize books into fixed-length sequences"""
    print(f"\nTokenizing {len(books)} books into sequences of {max_length} tokens...")
    
    all_sequences = []
    
    for book in tqdm(books):
        # Tokenize the entire book
        tokens = tokenizer.encode(book['text'])
        
        # Split into chunks of max_length
        for i in range(0, len(tokens) - max_length, max_length):
            sequence = tokens[i:i+max_length]
            if len(sequence) == max_length:  # Only keep full sequences
                all_sequences.append({
                    'input_ids': sequence,
                    'book_id': book['book_id']
                })
    
    print(f"Created {len(all_sequences)} sequences of {max_length} tokens")
    
    return all_sequences

def main():
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Initialize tokenizer
    print("Initializing GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and filter data
    books = load_and_filter_data()
    
    # Create splits
    train_books, val_books, test_books = create_splits(books)
    
    # Tokenize each split
    print("\n=== Tokenizing Training Data ===")
    train_sequences = tokenize_books(train_books, tokenizer, max_length=1024)
    
    print("\n=== Tokenizing Validation Data ===")
    val_sequences = tokenize_books(val_books, tokenizer, max_length=1024)
    
    print("\n=== Tokenizing Test Data ===")
    test_sequences = tokenize_books(test_books, tokenizer, max_length=1024)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_sequences)
    val_dataset = Dataset.from_list(val_sequences)
    test_dataset = Dataset.from_list(test_sequences)
    
    # Save to disk
    print("\nSaving processed datasets...")
    train_dataset.save_to_disk('data/processed/train')
    val_dataset.save_to_disk('data/processed/val')
    test_dataset.save_to_disk('data/processed/test')
    
    # Save processing info
    info = {
        'tokenizer': 'gpt2',
        'vocab_size': len(tokenizer),
        'max_length': 1024,
        'train_books': len(train_books),
        'val_books': len(val_books),
        'test_books': len(test_books),
        'train_sequences': len(train_sequences),
        'val_sequences': len(val_sequences),
        'test_sequences': len(test_sequences),
        'total_train_tokens': len(train_sequences) * 1024,
    }
    
    with open('data/processed/info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n Data preprocessing complete!")
    print(f"\nDataset statistics:")
    print(f"  Training sequences: {len(train_sequences):,}")
    print(f"  Validation sequences: {len(val_sequences):,}")
    print(f"  Test sequences: {len(test_sequences):,}")
    print(f"  Total training tokens: {len(train_sequences) * 1024:,}")
    print(f"\nProcessed data saved to: data/processed/")

if __name__ == "__main__":
    main()
