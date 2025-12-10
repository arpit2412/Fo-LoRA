#!/bin/bash

# Usage: ./eval.sh <path_to_adapter>
# Example: ./eval.sh output/phi3-lora-20231027_120000

if [ -z "$1" ]; then
    echo "Error: Please provide the path to the adapter directory."
    echo "Usage: $0 <path_to_adapter>"
    exit 1
fi

ADAPTER_PATH="$1"
DATASET_PATH="/home/arpit/LS-LoRA/dataset/openbookqa/test.json"
OUTPUT_FILE="${ADAPTER_PATH}/eval_results.json"

echo "Evaluating adapter: ${ADAPTER_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output file: ${OUTPUT_FILE}"

python eval.py \
    --adapter_path "$ADAPTER_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_file "$OUTPUT_FILE"

echo "Evaluation finished."
