"""
Memory-efficient preprocessing and tokenization
Processes and saves data in batches to avoid OOM
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
    
    # Sample a smaller subset for baseline experiments
    # We don't need all 47k books for initial baseline
    print("\nSampling subset for baseline training...")
    
    fiction_books = []
    for i, example in enumerate(tqdm(dataset)):
        text = example['TEXT']
        if len(text) > 10000:  # Filter out very short books
            fiction_books.append({
                'text': text,
                'book_id': i
            })
    
    # Sample 5000 books for baseline (still plenty of data)
    random.seed(42)
    random.shuffle(fiction_books)
    fiction_books = fiction_books[:5000]
    
    print(f"Using {len(fiction_books)} books for baseline training")
    
    return fiction_books

def create_splits(books, train_ratio=0.9, val_ratio=0.05):
    """Create train/validation/test splits"""
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

def tokenize_and_save_split(books, tokenizer, output_path, split_name, max_length=1024):
    """Tokenize books and save incrementally"""
    print(f"\n=== Processing {split_name} split ===")
    print(f"Tokenizing {len(books)} books...")
    
    os.makedirs(output_path, exist_ok=True)
    
    all_sequences = []
    total_sequences = 0
    batch_size = 100  # Process 100 books at a time
    
    for batch_start in tqdm(range(0, len(books), batch_size)):
        batch_books = books[batch_start:batch_start+batch_size]
        
        for book in batch_books:
            tokens = tokenizer.encode(book['text'])
            
            # Split into chunks
            for i in range(0, len(tokens) - max_length, max_length):
                sequence = tokens[i:i+max_length]
                if len(sequence) == max_length:
                    all_sequences.append({
                        'input_ids': sequence,
                        'book_id': book['book_id']
                    })
        
        # Save batch and clear memory
        if len(all_sequences) >= 10000:
            batch_dataset = Dataset.from_list(all_sequences)
            if total_sequences == 0:
                batch_dataset.save_to_disk(output_path)
            else:
                # Append to existing dataset
                existing = Dataset.load_from_disk(output_path)
                from datasets import concatenate_datasets
                combined = concatenate_datasets([existing, batch_dataset])
                combined.save_to_disk(output_path)
            
            total_sequences += len(all_sequences)
            all_sequences = []
    
    # Save remaining sequences
    if all_sequences:
        batch_dataset = Dataset.from_list(all_sequences)
        if total_sequences == 0:
            batch_dataset.save_to_disk(output_path)
        else:
            existing = Dataset.load_from_disk(output_path)
            from datasets import concatenate_datasets
            combined = concatenate_datasets([existing, batch_dataset])
            combined.save_to_disk(output_path)
        
        total_sequences += len(all_sequences)
    
    print(f"Created {total_sequences:,} sequences")
    return total_sequences

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
    
    # Process each split
    train_count = tokenize_and_save_split(
        train_books, tokenizer, 'data/processed/train', 'train'
    )
    val_count = tokenize_and_save_split(
        val_books, tokenizer, 'data/processed/val', 'validation'
    )
    test_count = tokenize_and_save_split(
        test_books, tokenizer, 'data/processed/test', 'test'
    )
    
    # Save processing info
    info = {
        'tokenizer': 'gpt2',
        'vocab_size': len(tokenizer),
        'max_length': 1024,
        'train_books': len(train_books),
        'val_books': len(val_books),
        'test_books': len(test_books),
        'train_sequences': train_count,
        'val_sequences': val_count,
        'test_sequences': test_count,
        'total_train_tokens': train_count * 1024,
    }
    
    with open('data/processed/info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n✅ Data preprocessing complete!")
    print(f"\nDataset statistics:")
    print(f"  Training sequences: {train_count:,}")
    print(f"  Validation sequences: {val_count:,}")
    print(f"  Test sequences: {test_count:,}")
    print(f"  Total training tokens: {train_count * 1024:,}")
    print(f"\nProcessed data saved to: data/processed/")
    print("\nReady for training!")

if __name__ == "__main__":
    main()
