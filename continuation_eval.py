"""
Continuation-based evaluation for narrative consistency
Tests if models maintain character traits and plot elements across long spans
"""

import torch
from transformers import GPT2Tokenizer
import json

# Evaluation prompts with clear character traits and plot setups
EVAL_PROMPTS = [
    {
        "name": "brave_knight",
        "prompt": "Sir Edmund was known throughout the kingdom as the bravest knight who ever lived. He had faced dragons, defeated armies, and never once showed fear. One morning, he received a mysterious letter that would change everything. The letter said:",
        "character_traits": ["brave", "knight", "fearless"],
        "expected_consistency": "Should maintain Edmund's bravery throughout"
    },
    {
        "name": "clever_detective",
        "prompt": "Detective Sarah Chen had solved every case in her twenty-year career through careful observation and brilliant deduction. She never relied on luck, only logic. When she arrived at the mansion where the diamond had been stolen, she immediately noticed something odd about the butler's shoes.",
        "character_traits": ["detective", "clever", "logical", "observant"],
        "expected_consistency": "Should show Sarah using logic and observation"
    },
    {
        "name": "kind_teacher",
        "prompt": "Mr. Rodriguez was beloved by all his students for his patience and kindness. He never raised his voice, always found time to help struggling students, and believed every child could succeed. When the school principal told him about the difficult new student arriving tomorrow, he smiled warmly and said,",
        "character_traits": ["teacher", "kind", "patient"],
        "expected_consistency": "Should maintain Rodriguez's kindness and patience"
    },
    {
        "name": "gun_on_wall",
        "prompt": "The old hunting rifle hung above the fireplace, its polished barrel gleaming in the afternoon light. Margaret had inherited it from her grandfather but had never fired it. As she poured tea for her unexpected guest, she couldn't help but glance at the rifle. The guest noticed her looking and asked about it. She explained its history, then they continued their conversation about the missing inheritance documents.",
        "plot_elements": ["rifle", "inheritance", "guest"],
        "expected_consistency": "Should return to rifle or inheritance by end (Chekhov's gun)"
    },
    {
        "name": "mysterious_box",
        "prompt": "The locked wooden box sat on the table between them. Neither James nor his sister knew what their late father had placed inside it, but the lawyer had been very specific: do not open it until exactly midnight on the anniversary of his death. That was three hours from now. James paced nervously while his sister watched the clock.",
        "plot_elements": ["locked box", "midnight", "father's death", "siblings"],
        "expected_consistency": "Should build toward midnight/opening the box"
    }
]

def generate_continuation(model, tokenizer, prompt, max_length=500, temperature=0.8, device='cuda'):
    """Generate continuation from prompt"""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits
            
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    continuation = generated_text[len(prompt):]
    
    return continuation

def check_trait_consistency(continuation, traits):
    """
    Simple check: do character traits remain referenced?
    Returns dict with trait presence info
    """
    continuation_lower = continuation.lower()
    
    results = {}
    for trait in traits:
        # Check for trait or related words
        present = trait.lower() in continuation_lower
        results[trait] = present
    
    return results

def check_plot_elements(continuation, elements):
    """
    Check if plot elements are referenced again (Chekhov's gun test)
    """
    continuation_lower = continuation.lower()
    
    results = {}
    for element in elements:
        present = element.lower() in continuation_lower
        results[element] = present
    
    return results

def evaluate_model(model, tokenizer, output_file, device='cuda'):
    """
    Run all evaluation prompts and save results
    """
    results = []
    
    print(f"\nRunning continuation evaluation...")
    print("="*60)
    
    for prompt_data in EVAL_PROMPTS:
        print(f"\nPrompt: {prompt_data['name']}")
        print("-"*60)
        
        # Generate continuation
        continuation = generate_continuation(
            model, tokenizer, prompt_data['prompt'],
            max_length=500, temperature=0.8, device=device
        )
        
        result = {
            'name': prompt_data['name'],
            'prompt': prompt_data['prompt'],
            'continuation': continuation,
        }
        
        # Check character traits if present
        if 'character_traits' in prompt_data:
            trait_results = check_trait_consistency(
                continuation, prompt_data['character_traits']
            )
            result['trait_consistency'] = trait_results
            print(f"Character traits maintained: {trait_results}")
        
        # Check plot elements if present
        if 'plot_elements' in prompt_data:
            element_results = check_plot_elements(
                continuation, prompt_data['plot_elements']
            )
            result['plot_consistency'] = element_results
            print(f"Plot elements referenced: {element_results}")
        
        results.append(result)
        
        print(f"\nContinuation preview (first 200 chars):")
        print(continuation[:200])
        print("...")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    
    return results

def compare_models(baseline_results_file, dual_track_results_file):
    """
    Compare baseline vs dual-track on consistency metrics
    """
    with open(baseline_results_file) as f:
        baseline_results = json.load(f)
    
    with open(dual_track_results_file) as f:
        dual_track_results = json.load(f)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for i, (base, dual) in enumerate(zip(baseline_results, dual_track_results)):
        print(f"\n{base['name']}:")
        
        if 'trait_consistency' in base:
            base_traits = sum(base['trait_consistency'].values())
            dual_traits = sum(dual['trait_consistency'].values())
            print(f"  Character traits maintained:")
            print(f"    Baseline: {base_traits}/{len(base['trait_consistency'])}")
            print(f"    Dual-track: {dual_traits}/{len(dual['trait_consistency'])}")
        
        if 'plot_consistency' in base:
            base_plot = sum(base['plot_consistency'].values())
            dual_plot = sum(dual['plot_consistency'].values())
            print(f"  Plot elements referenced:")
            print(f"    Baseline: {base_plot}/{len(base['plot_consistency'])}")
            print(f"    Dual-track: {dual_plot}/{len(dual['plot_consistency'])}")

if __name__ == "__main__":
    print("Continuation evaluation prompts loaded.")
    print(f"Number of prompts: {len(EVAL_PROMPTS)}")
    print("\nUse evaluate_model(model, tokenizer, output_file) to run evaluation")