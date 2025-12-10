#!/bin/bash

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/phi3-lora-${TIMESTAMP}"

echo "Starting training..."
echo "Output directory: ${OUTPUT_DIR}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
# Using torchrun for multi-GPU support
# Assuming 2 GPUs available
torchrun --nproc_per_node=2 train.py \
    --dataset_path "/home/arpit/LS-LoRA/dataset/commonsense_5k.json" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 64

echo "Training finished. Model saved to ${OUTPUT_DIR}"
