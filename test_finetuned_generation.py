"""
Test generation from fine-tuned baseline
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading fine-tuned 2800M baseline...")
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/finetuned_2800m/best_model",
    trust_remote_code=True
)
model = model.cuda()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Test prompts
prompts = [
    "Sir Edmund was known throughout the kingdom as the bravest knight who ever lived. He had faced dragons, defeated armies, and never once showed fear. One morning,",
    "Detective Sarah Chen had solved every case in her twenty-year career through careful observation and brilliant deduction. When she arrived at the mansion,",
    "It was a dark and stormy night"
]

# Open file for writing
with open('finetuned_2800m_generations.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("Fine-tuned 2800M Generations\n")
    f.write("="*60 + "\n\n")
    
    for prompt in prompts:
        print(f"Generating for: {prompt[:50]}...")
        
        f.write(f"\nPrompt: {prompt}\n")
        f.write("-"*60 + "\n")
        
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        f.write(generated + "\n")
        f.write("="*60 + "\n")

print("\nGenerations saved")