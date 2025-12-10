"""Re-compute accuracy with fixed extraction logic."""
import json
import re

def extract_answer_fixed(text):
    """Fixed extraction that prioritizes specific patterns."""
    text_lower = text.lower()

    # Check specific patterns FIRST
    specific_patterns = [
        r'correct\s*answer\s*is\s*answer\s*(\d)',
        r'answer\s*is\s*answer\s*(\d)',
        r'\(answer\s*(\d)\)',
    ]

    for pattern in specific_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return f"answer{match.group(1)}"

    # Fall back to last occurrence
    generic_pattern = r'answer\s*(\d)'
    matches = list(re.finditer(generic_pattern, text_lower))
    if matches:
        return f"answer{matches[-1].group(1)}"

    return None

# Load existing results
with open('eval_results_openbookqa_full.json', 'r') as f:
    data = json.load(f)

# Re-compute with fixed extraction
correct = 0
total = 0
fixed_count = 0

print("Re-evaluating with fixed extraction logic...")
print("=" * 80)

for item in data['results']:
    response = item['response']
    ground_truth = item['ground_truth']
    old_predicted = item['predicted']

    # Apply fixed extraction
    new_predicted = extract_answer_fixed(response)

    is_correct = (new_predicted == ground_truth)
    if is_correct:
        correct += 1
    total += 1

    # Track how many predictions changed
    if old_predicted != new_predicted:
        fixed_count += 1

new_accuracy = (correct / total) * 100
old_accuracy = data['summary']['accuracy']

print(f"\nResults:")
print(f"  Old accuracy: {old_accuracy:.2f}% ({data['summary']['correct']}/{total})")
print(f"  New accuracy: {new_accuracy:.2f}% ({correct}/{total})")
print(f"  Predictions fixed: {fixed_count}/{total}")
print(f"  Improvement: {new_accuracy - old_accuracy:+.2f}%")
print("=" * 80)

# Show some examples that were fixed
print("\nExamples of fixed predictions:")
print("-" * 80)
count = 0
for item in data['results']:
    old_predicted = item['predicted']
    new_predicted = extract_answer_fixed(item['response'])

    if old_predicted != new_predicted:
        count += 1
        if count <= 5:
            status = "✓" if new_predicted == item['ground_truth'] else "✗"
            print(f"\n{status} Example {count}:")
            print(f"  Ground truth: {item['ground_truth']}")
            print(f"  Old prediction: {old_predicted}")
            print(f"  New prediction: {new_predicted}")
            print(f"  Response snippet: ...{item['response'][-100:]}")
        if count >= 5:
            break
