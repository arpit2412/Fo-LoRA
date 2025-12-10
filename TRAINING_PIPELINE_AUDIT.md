# Training Pipeline Optimization Audit

**Date**: 2025-12-10
**Current Status**: GPU utilization 90% (good), but pipeline can be optimized further
**Environment**: PyTorch 2.9.1, CUDA 12.8, Transformers 4.57.3

---

## Critical Findings

### 🔴 CRITICAL #1: Flash Attention NOT Installed (BIGGEST IMPACT)

**Current Status**: ❌ NOT INSTALLED
```
`flash-attention` package not found
Current `flash-attention` does not support `window_size`
```

**Impact**: Missing **2-3x speedup** on attention operations (50-60% of training time)

**Why It Matters**:
- Phi-3 uses multi-head attention in every layer (32 layers × 32 heads)
- Standard attention is O(n²) in memory and compute
- Flash Attention 2 is memory-efficient O(n) with kernel fusion
- **Estimated speedup: 1.5-2x overall training speed**

**Fix**:
```bash
pip install flash-attn --no-build-isolation
```

**Usage in training**:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # ADD THIS
)
```

**Note**: With quantization (4-bit), Flash Attention compatibility may vary. Test first!

---

### 🔴 CRITICAL #2: DataLoader Not Optimized

**Current Status**: No dataloader settings specified

**Missing Configuration**:
```python
training_args = SFTConfig(
    # ... existing args ...
    dataloader_num_workers=4,       # MISSING - parallel data loading
    dataloader_pin_memory=True,     # MISSING - faster CPU→GPU transfer
    dataloader_prefetch_factor=2,   # MISSING - prefetch batches
)
```

**Impact**:
- CPU may be bottleneck loading/tokenizing data while GPU waits
- **Estimated speedup: 10-20%** if CPU-bound

**Recommended Settings**:
```python
# For 2 GPUs, 4 workers per GPU = 8 total
dataloader_num_workers=4  # Per device workers
dataloader_pin_memory=True  # Faster CPU→GPU
dataloader_prefetch_factor=2  # Prefetch 2 batches per worker
```

**Test to verify CPU bottleneck**:
```bash
# Monitor during training
nvidia-smi dmon -s u -d 1
# If GPU util fluctuates (e.g., 50%→90%→50%), CPU is bottleneck
```

---

### 🟡 MODERATE #3: torch.compile NOT Used

**Current Status**: Not using PyTorch 2.x compilation

**Potential**: PyTorch 2.9 has excellent `torch.compile()` support
- Can provide 20-40% speedup with graph optimizations
- Fuses operations, reduces kernel launches

**Implementation**:
```python
# After creating peft model
model = get_peft_model(model, peft_config)

# Compile the model (PyTorch 2.0+)
model = torch.compile(
    model,
    mode="reduce-overhead",  # Best for training
    fullgraph=False,  # Allow dynamic shapes
)
```

**Caveat**:
- LoRA + quantization + compile can have compatibility issues
- Test with short run first (10 steps)
- May increase first step latency (compilation time)

**Estimated Impact**: 15-25% speedup if compatible

---

### 🟡 MODERATE #4: Optimizer Not Fused

**Current Setting**: `optim="adamw_torch"`

**Better Option**: Fused AdamW (CUDA kernel fusion)
```python
optim="adamw_torch_fused"  # or "adamw_8bit" for memory savings
```

**Impact**: 5-10% speedup on optimizer step
- Fused optimizer combines multiple operations into single CUDA kernel
- Less kernel launch overhead

---

### 🟢 LOW #5: Sequence Packing Could Help

**Current**: Each sample padded to max_length (2048)

**Optimization**: Pack multiple short sequences into single sample
- Reduces wasted computation on padding tokens
- Most complex to implement (requires dataset restructuring)

**Estimated Impact**: 10-30% speedup if average sequence << 2048

**Implementation**: Use `packing=True` in SFTConfig (if supported)

---

### 🟢 LOW #6: Gradient Accumulation Tuning

**Current**: `gradient_accumulation_steps=4` (single GPU) or `4 // world_size` (DDP)

**Analysis**:
- Batch size 16 with accumulation 4 = effective batch 64
- Good balance between memory and throughput

**Recommendation**: **Keep as is** - well-tuned for 90% GPU utilization

---

### ✅ GOOD #7: Already Optimized Settings

**bf16 training**: ✅ Using bfloat16 (better than fp16 for stability)
**Gradient checkpointing**: ✅ Enabled (saves memory)
**4-bit quantization**: ✅ Using NF4 (reduces model size 4x)
**DDP setup**: ✅ Proper multi-GPU configuration

---

## Priority Optimization Plan

### Phase 1: Quick Wins (30 min implementation)

1. **Add DataLoader settings** (HIGHEST PRIORITY for CPU-bound)
   ```python
   dataloader_num_workers=4,
   dataloader_pin_memory=True,
   dataloader_prefetch_factor=2,
   ```
   **Expected**: 10-20% speedup if CPU bottleneck

2. **Use fused optimizer** (1 line change)
   ```python
   optim="adamw_torch_fused",
   ```
   **Expected**: 5-10% speedup

### Phase 2: Flash Attention (1-2 hours including testing)

3. **Install and test Flash Attention**
   ```bash
   pip install flash-attn --no-build-isolation
   ```
   Add `attn_implementation="flash_attention_2"` to model loading

   **Expected**: 50-100% speedup (MASSIVE)
   **Risk**: May not work with 4-bit quantization - needs testing

### Phase 3: torch.compile (1 hour including testing)

4. **Add torch.compile**
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```
   **Expected**: 15-25% speedup
   **Risk**: May have issues with LoRA + quantization

---

## Expected Total Speedup

| Optimization | Speedup | Compatibility Risk | Implementation Time |
|--------------|---------|-------------------|---------------------|
| DataLoader settings | 10-20% | None | 5 min |
| Fused optimizer | 5-10% | None | 1 min |
| Flash Attention 2 | 50-100% | MODERATE (4-bit) | 1-2 hours |
| torch.compile | 15-25% | MODERATE (LoRA) | 1 hour |

**Conservative Estimate** (Phase 1 only): **15-30% faster**
**Aggressive Estimate** (All phases): **2-3x faster** if Flash Attention works

---

## Implementation Code

### train_lslora.py Changes

```python
# After line 56: model loading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # NEW - Phase 2
)

# After line 157: training config
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=max_steps,
    save_strategy="steps",
    save_steps=100,
    fp16=False,
    bf16=True,
    optim="adamw_torch_fused",  # NEW - Phase 1
    report_to="wandb",
    ddp_find_unused_parameters=False,
    max_length=max_seq_length,
    dataset_text_field="text",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # NEW - Phase 1: DataLoader optimization
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2,
)

# After line 165: before training
# NEW - Phase 3: torch.compile (OPTIONAL)
# model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
```

---

## Testing Strategy

### Phase 1 Test (DataLoader + Fused Optimizer)
```bash
python train_lslora.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --dataset_path dataset/commonsense_5k.json \
    --output_dir output/phase1_test \
    --max_steps 20 \
    --batch_size 16

# Compare steps/sec with baseline
```

### Phase 2 Test (Flash Attention)
```bash
# Install first
pip install flash-attn --no-build-isolation

# Test with short run
python train_lslora.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --dataset_path dataset/commonsense_5k.json \
    --output_dir output/flash_test \
    --max_steps 10 \
    --batch_size 16
```

**If fails**: May not support flash_attention_2 with 4-bit quantization. Try without quantization first to isolate issue.

### Phase 3 Test (torch.compile)
```bash
# Uncomment torch.compile line, then:
python train_lslora.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --dataset_path dataset/commonsense_5k.json \
    --output_dir output/compile_test \
    --max_steps 10 \
    --batch_size 16
```

**Note**: First step will be SLOW (compilation), subsequent steps should be faster.

---

## Monitoring During Training

```bash
# Terminal 1: Training
./scripts/train_phi3_170k.sh

# Terminal 2: GPU monitoring
watch -n 1 nvidia-smi

# Terminal 3: CPU/memory monitoring
htop

# Check for bottlenecks:
# - GPU util should stay >85%
# - If fluctuating (50%→90%→50%), CPU is bottleneck → increase num_workers
# - If GPU memory near 100%, reduce batch_size
```

---

## Risk Assessment

| Optimization | Breaking Risk | Accuracy Risk | Debug Difficulty |
|--------------|--------------|---------------|------------------|
| DataLoader | Very Low | None | Easy |
| Fused optimizer | Very Low | None | Easy |
| Flash Attention | MODERATE | None | Moderate |
| torch.compile | MODERATE | None | Hard |

**Recommendation**: Implement Phase 1 immediately (safe, quick wins). Test Phase 2-3 on short runs before full training.

---

## Conclusion

**Current state**: Well-optimized GPU memory and operations
**Bottleneck likely**: Data loading (CPU) or attention operations
**Biggest opportunity**: Flash Attention 2 (if compatible with quantization)
**Safe improvements**: DataLoader settings + fused optimizer (15-30% gain)

**Next Action**: Apply Phase 1 optimizations (5 minutes), test, then consider Flash Attention.
