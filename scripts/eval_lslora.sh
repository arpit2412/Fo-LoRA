#!/bin/bash

ADAPTER_PATH=$1

if [ -z "$ADAPTER_PATH" ]; then
    echo "Usage: bash eval_lslora.sh <adapter_path>"
    exit 1
fi

echo "Evaluating adapter: $ADAPTER_PATH"
OUTPUT_FILE="$ADAPTER_PATH/eval_results.json"

python eval.py \
    --base_model "microsoft/Phi-3-mini-4k-instruct" \
    --adapter_path "$ADAPTER_PATH" \
    --dataset_path "/home/arpit/LS-LoRA/dataset/openbookqa/test.json" \
    --output_file "$OUTPUT_FILE"

echo "Evaluation done. Results saved to $OUTPUT_FILE"
