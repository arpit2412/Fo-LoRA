"""Save corrected evaluation results."""
import json
import re

def extract_answer_fixed(text):
    """Fixed extraction that prioritizes specific patterns."""
    text_lower = text.lower()

    specific_patterns = [
        r'correct\s*answer\s*is\s*answer\s*(\d)',
        r'answer\s*is\s*answer\s*(\d)',
        r'\(answer\s*(\d)\)',
    ]

    for pattern in specific_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return f"answer{match.group(1)}"

    generic_pattern = r'answer\s*(\d)'
    matches = list(re.finditer(generic_pattern, text_lower))
    if matches:
        return f"answer{matches[-1].group(1)}"

    return None

# Load and fix
with open('eval_results_openbookqa_full.json', 'r') as f:
    data = json.load(f)

# Update all predictions
correct = 0
for item in data['results']:
    item['predicted'] = extract_answer_fixed(item['response'])
    item['correct'] = (item['predicted'] == item['ground_truth'])
    if item['correct']:
        correct += 1

# Update summary
data['summary']['correct'] = correct
data['summary']['accuracy'] = (correct / data['summary']['total']) * 100

# Save corrected results
with open('eval_results_openbookqa_CORRECTED.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Saved corrected results: {correct}/{data['summary']['total']} = {data['summary']['accuracy']:.2f}%")
