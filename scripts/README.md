# Scripts Directory

This directory contains all training, evaluation, and utility scripts for Fourier LS-LoRA experiments.

## Training Scripts

### Shell Scripts (*.sh)
- `train_phi3_170k.sh` - Train Phi-3 on CommonsenseQA 170k dataset
- `train_lslora.sh` - General training script template
- `eval_lslora.sh` - Evaluation script for trained models
- `eval.sh` - Standard evaluation script

**Naming Convention**: `{action}_{model}_{dataset}.sh`

Example: `train_llama_commonsense.sh`

### Python Training Scripts
- `train.py` - Standard training script
- Main training script is in root: `/home/arpit/peft/train_lslora.py`

## Evaluation Scripts

### Python Scripts
- `eval.py` - Standard evaluation
- `eval_mean_token_length.py` - Token length analysis
- `calculate_score.py` - Score calculation utilities

### Usage Pattern
```bash
# Training
./scripts/train_phi3_170k.sh

# Evaluation
./scripts/eval_lslora.sh output/Fo-LoRA-phi3-commonsense170k-TIMESTAMP
```

## Naming Conventions

### Run Names
Format: `Fo-LoRA-{model}-{dataset}-{timestamp}`

Examples:
- `Fo-LoRA-phi3-commonsense170k-20251210-113956`
- `Fo-LoRA-llama-commonsense5k-20251210-120000`

### Output Directories
Located in: `output/Fo-LoRA-{model}-{dataset}-{timestamp}/`

Contains:
- `adapter_model.safetensors` - Trained LoRA weights
- `adapter_config.json` - Configuration
- `tokenizer files` - Tokenizer configuration
- `checkpoint-{step}/` - Training checkpoints (every 100 steps)

### Log Files
Located in: `logs/training-Fo-LoRA-{model}-{dataset}-{timestamp}.log`

Contains full training output with timestamps.

## Creating New Training Scripts

Template for new datasets/models:

```bash
#!/bin/bash

# Configuration
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME="Fo-LoRA-{MODEL_SHORT}-{DATASET_SHORT}-${TIMESTAMP}"
OUTPUT_DIR="output/${RUN_NAME}"
LOG_FILE="logs/training-${RUN_NAME}.log"

# Create directories
mkdir -p "${OUTPUT_DIR}" "logs"

# Print configuration
echo "=============================================================================="
echo "FOURIER LS-LORA TRAINING - {MODEL_NAME} ON {DATASET_NAME}"
echo "=============================================================================="
echo "Run name: ${RUN_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Model: {FULL_MODEL_PATH}"
echo "Dataset: {DATASET_FILE}"
echo "=============================================================================="

# Start training
python train_lslora.py \
    --model_name {FULL_MODEL_PATH} \
    --dataset_path dataset/{DATASET_FILE} \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --max_steps {STEPS} \
    --batch_size {BATCH_SIZE} \
    2>&1 | tee "${LOG_FILE}"
```

## File Organization Rules

1. **Keep scripts here** - All .sh and utility .py files
2. **Keep main training script in root** - `/home/arpit/peft/train_lslora.py`
3. **Outputs go to output/** - Never in root directory
4. **Logs go to logs/** - Never in root directory
5. **Use timestamps** - Always include `$(date +%Y%m%d-%H%M%S)` in run names
