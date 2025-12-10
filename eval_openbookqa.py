"""Evaluate trained model on OpenBookQA test set with accuracy metrics."""
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import json
import argparse
from tqdm import tqdm
import re

def extract_answer(text):
    """Extract answer from generated text (answer1, answer2, etc.)"""
    text_lower = text.lower()

    # IMPORTANT: Check specific patterns FIRST before generic ones
    # This prevents matching answer options in the question
    specific_patterns = [
        r'correct\s*answer\s*is\s*answer\s*(\d)',  # "correct answer is answer2"
        r'answer\s*is\s*answer\s*(\d)',             # "answer is answer2"
        r'\(answer\s*(\d)\)',                        # "(answer2)"
    ]

    for pattern in specific_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return f"answer{match.group(1)}"

    # Fall back to generic pattern, but take the LAST occurrence
    # (to avoid matching question options)
    generic_pattern = r'answer\s*(\d)'
    matches = list(re.finditer(generic_pattern, text_lower))
    if matches:
        # Take the last match (most likely the actual answer)
        return f"answer{matches[-1].group(1)}"

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter")
    parser.add_argument("--dataset_path", type=str, default="dataset/openbookqa/test.json")
    parser.add_argument("--output_file", type=str, default="eval_results_openbookqa.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation samples")
    args = parser.parse_args()

    print("="*80)
    print("OpenBookQA EVALUATION")
    print("="*80)

    # Load model
    print(f"\nLoading model from: {args.adapter_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_path,
        device_map="auto",
    )

    # Load tokenizer
    config = model.peft_config['default']
    base_model = config.base_model_name_or_path
    print(f"Base model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset_path}")
    with open(args.dataset_path, 'r') as f:
        data = json.load(f)

    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Limiting to {args.max_samples} samples")

    print(f"Total samples: {len(data)}")

    results = []
    correct = 0
    total = 0

    print("\nStarting evaluation...")
    for item in tqdm(data):
        instruction = item['instruction']
        ground_truth_answer = item['answer']

        # Format prompt (Phi-3 chat format)
        prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,  # Shorter for multiple choice
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        response = generated_text.split("<|assistant|>")[-1].strip()

        # Extract predicted answer
        predicted_answer = extract_answer(response)

        # Check correctness
        is_correct = (predicted_answer == ground_truth_answer)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "instruction": instruction,
            "ground_truth": ground_truth_answer,
            "predicted": predicted_answer,
            "response": response,
            "correct": is_correct
        })

    # Calculate final accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nTotal samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*80)

    # Save detailed results
    output_data = {
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        },
        "results": results
    }

    print(f"\nSaving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\nDone!")

    # Print a few examples
    print("\nSample Predictions:")
    print("-" * 80)
    for i, result in enumerate(results[:5]):
        status = "✓" if result['correct'] else "✗"
        print(f"\n{status} Example {i+1}:")
        print(f"  Ground truth: {result['ground_truth']}")
        print(f"  Predicted: {result['predicted']}")
        print(f"  Response: {result['response'][:100]}...")

if __name__ == "__main__":
    main()
