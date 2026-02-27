#!/bin/bash

# Parallel Inference on dapo_5k dataset using vllm with 8 GPUs
# This script runs inference on the dapo_5k dataset using Qwen3-8b model

# Configuration
MODEL="Qwen/Qwen3-8B"
DATA_PATH="data/dapo_5k/data.jsonl"
OUTPUT_DIR="output/dapo_5k_inference"
NUM_GPUS=8
TEMPERATURE=0.7
TOP_P=0.8
MAX_TOKENS=32768

# Run inference
python parallel_inference_dapo.py \
    --model "$MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus $NUM_GPUS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_tokens $MAX_TOKENS

echo ""
echo "Inference complete! Results saved to: $OUTPUT_DIR"
echo "  - all_results.jsonl: All inference results"
echo "  - errors.jsonl: Incorrect predictions for further analysis"
echo "  - correct_results.jsonl: Correct predictions"
echo "  - summary.json: Summary statistics"
