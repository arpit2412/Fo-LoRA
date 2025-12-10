# LS-LoRA Fourier Parameter Save/Load Fix

## Problem
Fourier parameters in LS-LoRA were training correctly but not persisting to disk, making the implementation unusable for evaluation and visualization.

## Root Cause
PEFT's `get_peft_model_state_dict()` in `/home/arpit/peft/src/peft/utils/save_and_load.py` filters parameters by prefix matching ("lora_" prefix). Fourier parameters have names like:
- `base_model.model.model.layers.0.self_attn.qkv_proj.fourier_params.weight`
- `base_model.model.model.layers.0.self_attn.qkv_proj.fourier_params.sigma`

These contain "fourier_params" but NOT "lora_", so they were filtered out during save.

Similarly, the load mechanism (`set_peft_model_state_dict()`) only loaded parameters matching specific prefixes, missing Fourier parameters.

## Solution
Added special handling for `PeftType.LSLORA` in both save and load mechanisms, similar to how SHIRA handles custom parameters.

### Changes Made

#### 1. Save Mechanism (line 134-159)
```python
elif config.peft_type == PeftType.LSLORA:
    # Handle LS-LoRA: includes both standard LoRA parameters and Fourier parameters
    bias = config.bias
    if bias == "none":
        to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
    elif bias == "all":
        to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError

    # Filter by adapter name for LoRA parameters
    to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}

    # Add Fourier parameters (critical for LS-LoRA)
    # Note: Fourier params may not include adapter name in key depending on how ModuleDict is serialized
    for k, v in state_dict.items():
        if "fourier_params" in k:
            to_return[k] = v
```

#### 2. Load Mechanism (line 545-557)
```python
elif config.peft_type == PeftType.LSLORA:
    # Handle Fourier parameters for LS-LoRA
    for name, module in model.named_modules():
        if hasattr(module, "fourier_params") and adapter_name in module.fourier_params:
            # Check for both weight and sigma parameters
            # Note: adapter name is NOT included in the saved key format
            weight_key = f"{name}.fourier_params.weight"
            sigma_key = f"{name}.fourier_params.sigma"

            if weight_key in peft_model_state_dict:
                module.fourier_params[adapter_name].weight.data = peft_model_state_dict.pop(weight_key)
            if sigma_key in peft_model_state_dict:
                module.fourier_params[adapter_name].sigma.data = peft_model_state_dict.pop(sigma_key)
```

## Verification
Created test script `/home/arpit/peft/test_save_load_fourier.py` that:
1. Creates LS-LoRA model with Fourier parameters
2. Modifies parameters to non-identity values
3. Saves model
4. Verifies Fourier parameters are in saved file (64 parameters)
5. Loads model
6. Verifies loaded parameters match original modified values

**Result**: ✅ ALL TESTS PASSED

## Files Modified
- `/home/arpit/peft/src/peft/utils/save_and_load.py` (lines 134-159, 545-557)

## Files Created
- `/home/arpit/peft/test_save_load_fourier.py` (test script)
- `/home/arpit/peft/print_learned_fourier.py` (visualization script)

## Impact
- Fourier parameters now correctly save and load
- Trained models can be used for evaluation
- Learned non-linearities can be analyzed per layer
- Implementation is now production-ready

## Next Steps
1. ✅ Full 500-step training running (output/lslora-final)
2. Load trained model and print learned coefficients
3. Visualize learned Fourier curves per layer
4. Evaluate on OpenBookQA dataset
