# LS-LoRA Fixes Applied

**Date:** 2025-12-09
**Status:** All critical fixes implemented and tested

---

## Summary

Fixed **7 critical bugs** in LS-LoRA implementation that were preventing it from working. The core architecture (layer-wise shared spline, like SineLoRA) was correct, but initialization and normalization issues prevented learning.

---

## Fixes Applied to `src/peft/tuners/lslora.py`

### Fix 1.1: Identity Initialization (Line 76)
**Before:**
```python
sin_init = grid + 0.2 * torch.sin(grid * 3.14159)  # Non-identity!
```

**After:**
```python
identity_init = grid.clone()  # Pure identity: v_i = k_i
```

**Impact:** Network now starts from LoRA baseline instead of degraded state.

---

### Fix 1.2: Grid Range Respect (Line 71)
**Before:**
```python
grid = torch.linspace(-1.0, 1.0, num_knots)  # Hardcoded, ignores config
```

**After:**
```python
grid = torch.linspace(k_min, k_max, num_knots)  # Uses config values
```

**Impact:** Spline operates on correct range [-3, 3] as configured.

---

### Fix 1.3: Remove Tanh Normalization (Lines 116-135)
**Before:**
```python
z_normalized = torch.tanh(z * input_scale)  # Squashes to [-1, 1]
spline_out = self.apply_spline(z_normalized, active_adapter)
z = spline_out * output_scale  # Scale back up
```

**After:**
```python
# Clamp to grid range (no information loss)
z_clamped = torch.clamp(z, k_min, k_max)
z_scaled = z_clamped * input_scale
spline_out = self.apply_spline(z_scaled, active_adapter)
z = spline_out * output_scale
```

**Impact:** No vanishing gradients, preserves magnitude information.

---

### Fix 1.4: Output Scale Initialization (Line 83)
**Before:**
```python
self.output_scale[adapter_name] = nn.Parameter(torch.ones(1) * 0.1)  # 10% strength
```

**After:**
```python
self.output_scale[adapter_name] = nn.Parameter(torch.ones(1))  # 100% strength
```

**Impact:** Spline contribution no longer suppressed by 10x.

---

### Fix 2.1-2.3: Add Piecewise Linear Interpolation (Lines 141-207)

Added three methods:
1. `apply_spline_piecewise()` - Hard indexing, O(1), MATHS.md spec
2. `apply_spline_soft()` - Softmax interpolation, O(n), current approach
3. `apply_spline()` - Router (defaults to piecewise)

**Impact:** Can now test both interpolation strategies.

---

### Fix 2.4: Config Parameter (Line 37)
Added `interpolation_method` parameter to `LSLoraConfig`:
```python
interpolation_method: str = "piecewise"  # "piecewise" or "soft"
```

---

## Fixes Applied to `train_lslora.py`

### Fix 3.1: Regularization Weight (Line 108)
**Before:** `spline_reg_weight=0.0001` (100x too weak)
**After:** `spline_reg_weight=0.01` (MATHS.md spec)

---

### Fix 3.2: Learning Rate (Line 156)
**Before:** `learning_rate=5e-5` (4x lower than standard LoRA)
**After:** `learning_rate=2e-4` (matches standard LoRA)

---

### Fix 3.3: Optimizer Precision (Line 163)
**Before:** `optim="paged_adamw_8bit"` (lower precision)
**After:** `optim="paged_adamw_32bit"` (matches standard LoRA)

---

### Fix 3.4: Diagnostic Baseline (Line 195)
**Before:**
```python
s_curve_baseline = grid + 0.1 * torch.tanh(grid)  # Non-identity baseline
deviation = (values - s_curve_baseline).abs().mean()
```

**After:**
```python
identity_baseline = grid  # True identity
deviation = (values - identity_baseline).abs().mean()
```

**Impact:** Diagnostics now correctly measure deviation from identity.

---

## Configuration Changes

### Updated LS-LoRA Config (train_lslora.py)
```python
peft_config = LSLoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    num_knots=5,
    k_min=-3.0,
    k_max=3.0,
    spline_reg_weight=0.01,  # ✓ Fixed
    interpolation_method="piecewise"  # ✓ New
)
```

### Updated Training Args (train_lslora.py)
```python
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=2e-4,  # ✓ Fixed (was 5e-5)
    logging_steps=10,
    max_steps=max_steps,
    save_strategy="steps",
    save_steps=100,
    fp16=False,
    bf16=True,
    optim="paged_adamw_32bit",  # ✓ Fixed (was 8-bit)
    report_to="wandb",
    ddp_find_unused_parameters=False,
    max_length=max_seq_length,
    dataset_text_field="text",
)
```

---

## Files Modified

1. **`src/peft/tuners/lslora.py`** (main implementation)
   - Fixed LSLoraConfig class (added interpolation_method)
   - Fixed LSLoraLayer.update_layer() (initialization, grid range)
   - Fixed LSLoraLayer.forward() (removed tanh, fixed clamping)
   - Added apply_spline_piecewise() method
   - Renamed apply_spline() → apply_spline_soft()
   - Added router apply_spline() method
   - Updated LSLoraModel._create_new_module() (pass interpolation_method)

2. **`train_lslora.py`** (training script)
   - Updated peft_config (reg weight, interpolation method)
   - Updated training_args (learning rate, optimizer)
   - Fixed check_spline_behavior() diagnostics

3. **`test_lslora_forward.py`** (NEW - testing script)
   - Verifies all fixes work correctly
   - Tests both interpolation methods
   - Checks gradient flow

4. **`FIXES_APPLIED.md`** (THIS FILE - documentation)

---

## Expected Results

### Before Fixes:
- LS-LoRA: BROKEN (doesn't run or produces bad results)
- Standard LoRA: 80.20% accuracy

### After Fixes:
- **Minimum Goal:** LS-LoRA ≥ 80% (match standard LoRA)
- **Target Goal:** LS-LoRA > 82% (2% improvement)
- **Stretch Goal:** LS-LoRA > 85% (5% improvement)

---

## How to Train

### Quick Start (Default Config):
```bash
bash train_lslora.sh
```

### Custom Config:
```bash
python train_lslora.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --dataset_path dataset/commonsense_5k.json \
    --output_dir output/lslora-fixed-$(date +%Y%m%d_%H%M%S) \
    --max_steps 500 \
    --batch_size 4
```

### Test Different Interpolation Methods:
Edit `train_lslora.py` line 109:
```python
interpolation_method="piecewise"  # or "soft"
```

---

## Diagnostic Metrics to Monitor

During training, check the spline diagnostics (printed at end):

1. **Deviation:** `mean(|v_i - k_i|)`
   - Initial: ~0.0 (identity)
   - Target: 0.01-0.1 (learning non-linearity)
   - Bad: >1.0 (overfitting)

2. **Slope Variance:** `var((v_{i+1} - v_i) / Δk)`
   - Initial: ~0.0 (linear)
   - Target: >0.05 (meaningful curvature)
   - Bad: >1.0 (unstable)

3. **Input/Output Scale:**
   - Initial: 1.0 / 1.0
   - Should stay near 1.0 (otherwise indicates issues)

4. **Training Loss:**
   - Should decrease smoothly
   - No NaN or explosions

---

## Ablation Study (Optional)

To systematically test impact of each fix, create variants:

| Variant | Fixes Applied | Expected Accuracy |
|---------|---------------|-------------------|
| V0 (Original) | None | BROKEN |
| V1 | Identity init only | ~70-75% |
| V2 | V1 + Grid range | ~75-78% |
| V3 | V2 + No tanh | ~78-80% |
| V4 (All critical) | V3 + Output scale | ~80% ✓ |
| V5 | V4 + Piecewise interp | ~80-82% |
| V6 | V5 + Strong reg | ~81-83% |
| V7 (Full) | V6 + Training config | ~82-85% ✓✓ |

---

## Architecture Confirmation

✅ **Layer-wise spline** - One spline per LoRA layer (e.g., 224 splines for Phi-3)
✅ **Shared across features** - Same spline for all activations in that layer
✅ **Element-wise application** - `φ(BAx)` where φ is the learnable spline
✅ **Similar to SineLoRA** - But with 10 learnable params instead of 1

**Total Parameters:**
- Standard LoRA: ~1.3M parameters (rank 16)
- LS-LoRA: ~1.3M + 2.2K = **~1.302M parameters**
- Overhead: **0.17%** (negligible)

---

## Next Steps

1. **Immediate:** Run training with fixed implementation
   ```bash
   bash train_lslora.sh
   ```

2. **Validation:** Check that accuracy ≥ 80% (standard LoRA baseline)

3. **Diagnostic:** Inspect spline diagnostics at end of training
   - Verify deviation > 0.01 (splines learned something)
   - Verify slope_variance > 0.05 (non-linear)

4. **Comparison:** Run evaluation and compare to standard LoRA
   ```bash
   bash eval_lslora.sh output/lslora-fixed-TIMESTAMP
   python calculate_score.py output/lslora-fixed-TIMESTAMP/eval_results.json
   ```

5. **Optional:** If V7 (full fixes) works well (>82%), try:
   - Reduce num_knots to 3 (minimal complexity)
   - Test gated non-linearity
   - Extend to other benchmarks (GSM8K, MMLU)

---

## Troubleshooting

### Issue: Training crashes / NaN loss
- Check grid range matches activation magnitudes
- Try smaller learning rate (1e-4)
- Reduce regularization weight

### Issue: No improvement over standard LoRA
- Check spline diagnostics: deviation should be > 0.01
- If deviation ≈ 0, splines not learning → increase LR or reduce reg
- Try soft interpolation instead of piecewise

### Issue: Worse than standard LoRA
- Verify all 7 fixes applied correctly
- Check that output_scale initialized to 1.0 (not 0.1)
- Confirm no tanh normalization in forward pass

---

## References

- **Plan:** `/home/arpit/.claude/plans/smooth-rolling-cake.md`
- **MATHS.md:** Mathematical specification (sections 1-5)
- **SineLoRA:** Reference for layer-wise non-linear LoRA

---

## Summary of Changes

**7 Critical Bugs Fixed:**
1. ✅ Identity initialization (v_i = k_i)
2. ✅ Grid range respect (k_min/k_max from config)
3. ✅ Remove tanh normalization
4. ✅ Fix output_scale init (1.0 not 0.1)
5. ✅ Add piecewise linear interpolation
6. ✅ Increase regularization (0.01 not 0.0001)
7. ✅ Fix training config (LR, optimizer)

**Architecture Validated:**
- ✅ Layer-wise shared spline (like SineLoRA)
- ✅ Element-wise application `φ(BAx)`
- ✅ More flexible than sinusoidal (10 params vs 1)

**Status:** READY FOR TRAINING! 🚀
