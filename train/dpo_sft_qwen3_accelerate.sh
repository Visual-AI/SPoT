#!/bin/bash
# DPO + SFT combined training script for Qwen3-8B (accelerate launch)
# Usage: bash train/dpo_sft_qwen3_accelerate.sh <output_ckpt_path> <data_file> [base_model_path]

# Parse arguments
ckpt_path=""
data_file=""
model_name_or_path=""

for arg in "$@"; do
    if [ -z "$ckpt_path" ]; then
        ckpt_path="$arg"
    elif [ -z "$data_file" ]; then
        data_file="$arg"
    elif [ -z "$model_name_or_path" ]; then
        model_name_or_path="$arg"
    fi
done

if [ -z "$ckpt_path" ] || [ -z "$data_file" ]; then
    echo "Usage: bash train/dpo_sft_qwen3_accelerate.sh <output_ckpt_path> <data_file> [base_model_path]"
    echo "Example:"
    echo "  bash train/dpo_sft_qwen3_accelerate.sh ckpts/qwen3_dpo_sft data/combined_filtered_max60change.jsonl ckpts/qwen3_bco_1e6"
    exit 1
fi

# ============================================
# DPO + SFT HYPERPARAMETERS (FULL FINETUNE)
# ============================================

# Model path (default to Qwen/Qwen3-8B if not specified)
if [ -z "$model_name_or_path" ]; then
    model_name_or_path="Qwen/Qwen3-8B"
fi

# DPO beta
beta=0.1

# Learning rate
lr=1e-6

# Combined loss configuration
# loss_type: comma-separated list (e.g., "sigmoid,sft" or "bco_pair")
loss_type="bco_pair"

# loss_weights_str: comma-separated weights matching loss_type entries
loss_weights="1"

# Training epochs
epochs=2

# Weight decay
weight_decay=0.01

# Batch size per device
micro_batch_size=2

# Gradient accumulation steps
gradient_accumulation_steps=2

# Warmup ratio
warmup_ratio=0.05

# LR scheduler
lr_scheduler_type="cosine"

# Optimizer betas
adam_beta1=0.9
adam_beta2=0.95

# Max sequence length
length_limit=8192

# Logging / saving
logging_steps=10
save_strategy="steps"
save_steps=50

# ============================================
# EXECUTION
# ============================================

# Detect GPUs
gpu_count=$(nvidia-smi -L | wc -l)

echo "=============================================="
echo "DPO + SFT Combined Training for Qwen3-8B"
echo "Model: $model_name_or_path"
echo "Output: $ckpt_path"
echo "Data: $data_file"
echo "GPUs: $gpu_count"
echo "Loss type: $loss_type"
echo "Loss weights: $loss_weights"
echo "Beta: $beta"
echo "LR: $lr"
echo "Epochs: $epochs"
echo "BS per GPU: $micro_batch_size"
echo "Grad Accum: $gradient_accumulation_steps"
echo "Effective BS: $((micro_batch_size * gradient_accumulation_steps * gpu_count))"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

config_file="${ACCELERATE_CONFIG_FILE:-train/accelerate_config.yaml}"

accelerate launch \
    --config_file "$config_file" \
    --num_processes $gpu_count \
    train/dpo_sft_qwen3_accelerate.py \
    --model_name_or_path "$model_name_or_path" \
    --data_file "$data_file" \
    --output_dir "$ckpt_path" \
    --beta $beta \
    --loss_type "$loss_type" \
    --loss_weights_str "$loss_weights" \
    --precompute_ref_log_probs True \
    --precompute_ref_batch_size 4 \
    --length_limit $length_limit \
    --per_device_train_batch_size $micro_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs $epochs \
    --warmup_ratio $warmup_ratio \
    --bf16 True \
    --logging_steps $logging_steps \
    --save_strategy $save_strategy \
    --save_steps $save_steps \
    --lr_scheduler_type $lr_scheduler_type \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --adam_beta1 $adam_beta1 \
    --adam_beta2 $adam_beta2 \
    --report_to "wandb"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "Training success! Model saved to $ckpt_path"
else
    echo "Training failed with error code $exit_code"
    exit $exit_code
fi
