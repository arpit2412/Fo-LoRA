# OpenBookQA Evaluation Bug Fix

## Issue
Initial evaluation showed **27.60% accuracy**, which seemed inconsistent with:
- Training token accuracy: ~80%
- Normal LoRA performance: 80% on OpenBookQA

## Root Cause
**Answer extraction bug** in `eval_openbookqa.py`.

The model was generating correct responses like:
```
"...Answer1: option1 Answer2: option2... the correct answer is answer2"
```

But the extraction logic was finding the **first occurrence** of "answer\d", which matched the question options ("Answer1: option1") instead of the actual answer at the end ("answer2").

### Old Extraction Logic (BROKEN)
```python
patterns = [
    r'answer\s*(\d)',  # TOO GENERIC - matches question options first!
    r'answer\s*is\s*answer\s*(\d)',
    r'correct\s*answer\s*is\s*answer\s*(\d)',
]

for pattern in patterns:
    match = re.search(pattern, text_lower)
    if match:
        return f"answer{match.group(1)}"  # Returns first match
```

### New Extraction Logic (FIXED)
```python
# Check SPECIFIC patterns FIRST
specific_patterns = [
    r'correct\s*answer\s*is\s*answer\s*(\d)',  # Prioritize this!
    r'answer\s*is\s*answer\s*(\d)',
    r'\(answer\s*(\d)\)',
]

# Then fall back to LAST occurrence of generic pattern
generic_pattern = r'answer\s*(\d)'
matches = list(re.finditer(generic_pattern, text_lower))
if matches:
    return f"answer{matches[-1].group(1)}"  # Return LAST match
```

## Results

| Metric | Before Fix | After Fix | Change |
|--------|-----------|-----------|--------|
| **Accuracy** | 27.60% | **83.00%** | +55.40% |
| **Correct Predictions** | 138/500 | 415/500 | +277 |
| **Fixed Extractions** | - | 364/500 | 72.8% |

## Conclusion
The Fourier LS-LoRA model was performing correctly all along at **83% accuracy**, matching normal LoRA performance. The low initial score was purely due to incorrect answer extraction from the generated text.

## Files Modified
- `eval_openbookqa.py` - Fixed extraction logic
- `eval_results_openbookqa_CORRECTED.json` - Corrected results with 83% accuracy

## Verification
Run recomputation script:
```bash
python recompute_accuracy.py
```
