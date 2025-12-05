"""
Create narrative fine-tuning dataset from Project Gutenberg
Focus on fiction only, character consistency and plot tracking
"""

from datasets import load_dataset, Dataset
import json
import random
from tqdm import tqdm

def create_narrative_dataset():
    """
    Create dataset with fiction narrative examples
    """
    
    print("Loading Project Gutenberg dataset...")
    dataset = load_dataset("sedthh/gutenberg_english", split="train")
    
    # Filter for fiction (LOCC starts with P) and substantial length
    print("Filtering for fiction books...")
    fiction_books = []
    
    for book in tqdm(dataset):
        try:
            metadata = json.loads(book['METADATA'])
            locc = metadata.get('locc', '')
            text = book['TEXT']
            
            # Fiction = LOCC starts with P, and substantial length
            if locc.startswith('P') and len(text) > 50000:
                fiction_books.append({
                    'text': text,
                    'title': metadata.get('title', 'unknown'),
                    'locc': locc
                })
        except:
            continue
    
    print(f"Found {len(fiction_books)} substantial fiction books")
    
    # Use subset for faster processing
    # fiction_books = fiction_books[:1000]
    
    examples = []
    
    print("Creating training examples...")
    for book in tqdm(fiction_books):
        text = book['text']
        
        # Split into chunks of ~2000 characters (roughly 500 tokens)
        chunk_size = 2000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text)-chunk_size*2, chunk_size)]
        
        # For each chunk, create continuation task
        for i in range(len(chunks)-1):
            prompt = chunks[i]
            continuation = chunks[i+1]
            
            # Only include if both parts are substantial
            if len(prompt) > 1500 and len(continuation) > 1500:
                examples.append({
                    'prompt': prompt,
                    'continuation': continuation,
                    'source': book['title']
                })
        
        # Limit total examples
        if len(examples) >= 50000:
            break
    
    print(f"\nCreated {len(examples)} training examples")
    
    # Split into train/val/test
    random.shuffle(examples)
    
    train_size = int(0.8 * len(examples))
    val_size = int(0.1 * len(examples))
    
    train_data = examples[:train_size]
    val_data = examples[train_size:train_size+val_size]
    test_data = examples[train_size+val_size:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Save datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    train_dataset.save_to_disk('data/finetuning/train')
    val_dataset.save_to_disk('data/finetuning/val')
    test_dataset.save_to_disk('data/finetuning/test')
    
    print("\nDatasets saved to data/finetuning/")
    
    # Show example
    print("\nExample training instance:")
    print("=" * 60)
    print("PROMPT:")
    print(train_data[0]['prompt'][:300] + "...")
    print("\nCONTINUATION:")
    print(train_data[0]['continuation'][:300] + "...")

if __name__ == "__main__":
    create_narrative_dataset()