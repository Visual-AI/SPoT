#!/bin/bash
# Script to prepare full DAPO dataset and process until 2k correct answers are obtained

set -e  # Exit on error

echo "================================================================"
echo "DAPO Full Dataset Processing - Target: 4000 Correct Answers"
echo "================================================================"
echo ""

# Step 1: Prepare the full DAPO dataset
echo "Step 1: Preparing full DAPO dataset..."
echo "================================================================"
cd SFT/data
if [ ! -d "dapo_full" ]; then
    echo "Dataset not found. Downloading and preparing full DAPO dataset..."
    python prepare_dapo_full.py
else
    echo "✓ Dataset already exists at dapo_full/"
fi

# Step 2: Run processing with target of 2000 correct answers
echo ""
echo "Step 2: Processing with Gemini API (target: 2000 correct answers)..."
echo "================================================================"
cd SFT
python scripts/correct_errors_parallel.py \
    --direct \
    --input data/dapo_full/data.jsonl \
    --output data/dapo_5k_inference/gemini_full_direct_prompt_remove_efficient.jsonl \
    --correct-output data/dapo_5k_inference/gemini_full_direct_prompt_remove_efficient_correct.jsonl \
    --target-correct 4000 \
    --workers 300

echo ""
echo "================================================================"
echo "Processing Complete!"
echo "================================================================"
echo ""
echo "Output files:"
echo "  All results: data/dapo_5k_inference/gemini_full_direct.jsonl"
echo "  Correct only: data/dapo_5k_inference/gemini_full_direct_correct.jsonl"
echo ""
echo "To check the results:"
echo "  wc -l data/dapo_5k_inference/gemini_full_direct_correct.jsonl"
