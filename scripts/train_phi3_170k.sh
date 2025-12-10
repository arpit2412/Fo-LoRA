#!/bin/bash

# Change to project root directory
cd /home/arpit/peft

# Training configuration
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME="Fo-LoRA-phi3-commonsense170k-${TIMESTAMP}"
OUTPUT_DIR="output/${RUN_NAME}"
LOG_FILE="logs/training-${RUN_NAME}.log"

# Create directories
mkdir -p "${OUTPUT_DIR}" "logs"

# Print configuration
echo "=============================================================================="
echo "FOURIER LS-LORA TRAINING - PHI-3 ON COMMONSENSE 170K"
echo "=============================================================================="
echo "Run name: ${RUN_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Model: microsoft/Phi-3-mini-4k-instruct"
echo "Dataset: commonsense_170k.json (170,420 samples)"
echo "Steps: 3000"
echo "Batch size: 16"
echo "=============================================================================="
echo ""

# Start training
python train_lslora.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --dataset_path dataset/commonsense_170k.json \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --max_steps 3000 \
    --batch_size 16 \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=============================================================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "Log saved to: ${LOG_FILE}"
echo "=============================================================================="
