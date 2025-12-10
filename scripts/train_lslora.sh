#!/bin/bash

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/lslora-phi3-$TIMESTAMP"

echo "Output Directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Run training with torchrun for multi-GPU
# Using 2 GPUs as per previous setup
torchrun --nproc_per_node=2 train_lslora.py \
    --dataset_path "/home/arpit/LS-LoRA/dataset/commonsense_5k.json" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --max_steps 500
