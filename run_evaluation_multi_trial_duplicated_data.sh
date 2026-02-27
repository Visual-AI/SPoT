#!/bin/bash

# Multi-Trial Evaluation using Data Duplication
# Uses random sampling (no fixed seed) so each duplicate gets different model output
# This is more efficient: model loaded once, dataset processed once

set -e

# Default model path
DEFAULT_MODEL="custom_models/Qwen3-8B-no-thinking"

# VLLM environment variables
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MODELSCOPE=False
export VLLM_RPC_TIMEOUT=180000
export VLLM_ENGINE_ITERATION_TIMEOUT_S=30

# Hugging Face authentication
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

# Parse arguments
MODEL_PATH=""
THINKING_FLAG=""

for arg in "$@"; do
    case $arg in
        --disable-thinking)
            THINKING_FLAG="--disable_thinking"
            shift
            ;;
        --enable-thinking)
            THINKING_FLAG="--enable_thinking"
            shift
            ;;
        *)
            if [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$arg"
            fi
            shift
            ;;
    esac
done

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"

# Generation parameters
TEMPERATURE=0.7
TOP_P=0.8
MAX_NEW_TOKENS=32768

# ============================================================================
# TASK CONFIGURATION
# Format: task_name -> "num_duplicates:tensor_parallel_size"
# num_duplicates = how many times to duplicate each example for variance measurement
# ============================================================================
declare -A TASK_CONFIG=(
    ["custom|aime24|0|0"]="16:1"
    ["custom|aime25|0|0"]="16:1"
    ["custom|amc23|0|0"]="16:1"
    ["custom|gpqa:diamond|0|0"]="5:2"

    ["custom|math_500|0|0"]="5:2"
    ["custom|minerva|0|0"]="5:1"
    ["custom|olympiadbench|0|0"]="5:2"
    ["custom|ifeval_no_thinking|0|0"]="5:2"
    # Custom task
    ["connect4"]="5:2"
)

# ============================================================================

echo "=========================================="
echo "Multi-Trial Evaluation (Data Duplication Method)"
echo "Model: $MODEL_PATH"
echo "Temperature: $TEMPERATURE (random sampling per duplicate)"
echo "Method: Duplicate data N times, evaluate once"
echo "Parallel execution: Up to 8 GPUs"
echo "=========================================="
echo ""
echo "Task Configuration:"
for task in "${!TASK_CONFIG[@]}"; do
    task_name=$(echo "$task" | sed 's/custom|//g' | sed 's/|.*//g' | sed 's/:/_/g')
    config=${TASK_CONFIG[$task]}
    num_duplicates=$(echo "$config" | cut -d':' -f1)
    tp_size=$(echo "$config" | cut -d':' -f2)
    echo "  $task_name: ${num_duplicates}x duplicates, TP=$tp_size"
done
echo "=========================================="

MODEL_NAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEMP_STR=$(echo "$TEMPERATURE" | tr '.' '_')
TOPP_STR=$(echo "$TOP_P" | tr '.' '_')
RESULTS_DIR="./evaluation_results/${MODEL_NAME}_${TIMESTAMP}_t${TEMP_STR}_p${TOPP_STR}_mt${MAX_NEW_TOKENS}_duplicated_data"

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR/outputs"
mkdir -p "$RESULTS_DIR/duplicated_datasets"

MAIN_LOG="$RESULTS_DIR/evaluation_run.log"
LOCK_FILE="$RESULTS_DIR/queue.lock"
GPU_ALLOCATION="$RESULTS_DIR/gpu_allocation.txt"

echo "Evaluation started at $(date)" | tee "$MAIN_LOG"
echo "Model: $MODEL_PATH" | tee -a "$MAIN_LOG"
echo "Results directory: $RESULTS_DIR" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Save task configuration
TASK_CONFIG_FILE="$RESULTS_DIR/task_config.txt"
echo "Task Configuration:" > "$TASK_CONFIG_FILE"
for task in "${!TASK_CONFIG[@]}"; do
    task_name=$(echo "$task" | sed 's/custom|//g' | sed 's/|.*//g' | sed 's/:/_/g')
    config=${TASK_CONFIG[$task]}
    num_duplicates=$(echo "$config" | cut -d':' -f1)
    tp_size=$(echo "$config" | cut -d':' -f2)
    echo "  $task_name: num_duplicates=$num_duplicates, tensor_parallel=$tp_size" >> "$TASK_CONFIG_FILE"
done

# Initialize GPU allocation
> "$GPU_ALLOCATION"
for i in {0..7}; do
    echo "$i:0" >> "$GPU_ALLOCATION"
done

rm -f "$LOCK_FILE"

# GPU allocation functions
allocate_gpus() {
    local tp_size=$1
    local allocated_gpus=""

    (
        flock -x 200
        local free_gpus=()
        while IFS=: read -r gpu_id status; do
            if [ "$status" = "0" ]; then
                free_gpus+=($gpu_id)
            fi
        done < "$GPU_ALLOCATION"

        if [ ${#free_gpus[@]} -ge $tp_size ]; then
            allocated_gpus=$(IFS=,; echo "${free_gpus[*]:0:$tp_size}")
            for ((i=0; i<$tp_size; i++)); do
                gpu_id=${free_gpus[$i]}
                sed -i "s/^$gpu_id:0$/$gpu_id:1/" "$GPU_ALLOCATION"
            done
        fi
        echo "$allocated_gpus"
    ) 200>"$LOCK_FILE"
}

release_gpus() {
    local gpu_list=$1
    (
        flock -x 200
        IFS=',' read -ra GPUS <<< "$gpu_list"
        for gpu_id in "${GPUS[@]}"; do
            sed -i "s/^$gpu_id:1$/$gpu_id:0/" "$GPU_ALLOCATION"
        done
    ) 200>"$LOCK_FILE"
}

# Function to run evaluation on duplicated data
run_evaluation() {
    local task=$1
    local task_display_name=$2
    local num_duplicates=$3
    local tensor_parallel_size=$4
    local gpu_list=$5

    local task_dir="$RESULTS_DIR/outputs/${task_display_name}"
    mkdir -p "$task_dir"
    local log_file="$RESULTS_DIR/logs/${task_display_name}.log"

    echo "[GPUs $gpu_list] ${task_display_name}: Evaluating with ${num_duplicates}x duplicated data" | tee -a "$MAIN_LOG"

    export CUDA_VISIBLE_DEVICES=$gpu_list

    # Isolated cache
    local task_cache_dir="$task_dir/.cache"
    mkdir -p "$task_cache_dir"
    export XDG_CACHE_HOME="$task_cache_dir"

    # Generate random seed for this run (for GPQA shuffling)
    local base_seed=$((RANDOM * RANDOM + 10#$(date +%s%N | cut -b10-18)))

    if [ "$task" = "connect4" ]; then
        # Connect4 doesn't support duplication easily, run normally
        python eval/eval_connect4_standardized.py \
            --run_path "$MODEL_PATH" \
            --data_dir eval \
            --data_file eval/connect4_eval_500.jsonl \
            --output_dir "$task_dir" \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --max_new_tokens $MAX_NEW_TOKENS \
            --tensor_parallel_size $tensor_parallel_size \
            $THINKING_FLAG \
            2>&1 | tee "$log_file"
    else
        # For lighteval tasks, use the duplicated dataset helper
        # Pass num_duplicates and base_seed to the modified sober_eval
        OMP_NUM_THREADS=8 python sober_eval/main_duplicated.py \
            --task "$task" \
            --model "$MODEL_PATH" \
            --max_new_tokens $MAX_NEW_TOKENS \
            --tensor_parallel_size $tensor_parallel_size \
            --use_chat_template \
            --custom_tasks_directory SFT/sober_eval/lighteval_tasks.py \
            --top_p $TOP_P \
            --temperature $TEMPERATURE \
            --top_k 20 \
            --num_duplicates $num_duplicates \
            --base_seed $base_seed \
            --output_dir "$task_dir" \
            $THINKING_FLAG \
            2>&1 | tee "$log_file"
    fi

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[GPUs $gpu_list] ✓ ${task_display_name} completed" | tee -a "$MAIN_LOG"
    else
        echo "[GPUs $gpu_list] ✗ ${task_display_name} FAILED (exit code $exit_code)" | tee -a "$MAIN_LOG"
    fi

    return $exit_code
}

# Worker function
run_worker() {
    local worker_id=$1

    while true; do
        local job=""
        {
            flock -x 200
            if [ -f "$TASK_QUEUE" ]; then
                job=$(head -n 1 "$TASK_QUEUE" 2>/dev/null || true)
                if [ -n "$job" ]; then
                    tail -n +2 "$TASK_QUEUE" > "${TASK_QUEUE}.tmp"
                    mv "${TASK_QUEUE}.tmp" "$TASK_QUEUE"
                fi
            fi
        } 200>"$LOCK_FILE"

        if [ -z "$job" ]; then
            break
        fi

        IFS=$'\t' read -r task_full task_display num_duplicates tp_size <<< "$job"

        # Allocate GPUs
        local gpu_list=""
        while [ -z "$gpu_list" ]; do
            gpu_list=$(allocate_gpus $tp_size)
            if [ -z "$gpu_list" ]; then
                sleep 2
            fi
        done

        # Run evaluation
        run_evaluation "$task_full" "$task_display" "$num_duplicates" "$tp_size" "$gpu_list"

        # Release GPUs
        release_gpus "$gpu_list"
    done
}

# Create task queue
TASK_QUEUE="$RESULTS_DIR/task_queue.txt"
> "$TASK_QUEUE"

echo "Building task queue..." | tee -a "$MAIN_LOG"

for task in "${!TASK_CONFIG[@]}"; do
    config=${TASK_CONFIG[$task]}
    num_duplicates=$(echo "$config" | cut -d':' -f1)
    tp_size=$(echo "$config" | cut -d':' -f2)
    task_display_name=$(echo "$task" | sed 's/custom|//g' | sed 's/|.*//g' | sed 's/:/_/g')

    printf "%s\t%s\t%s\t%s\n" "$task" "$task_display_name" "$num_duplicates" "$tp_size" >> "$TASK_QUEUE"
    echo "  $task_display_name: ${num_duplicates}x duplicates" | tee -a "$MAIN_LOG"
done

echo "" | tee -a "$MAIN_LOG"
echo "Starting workers..." | tee -a "$MAIN_LOG"

# Launch workers
NUM_WORKERS=8
pids=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    run_worker $i &
    pids+=($!)
    echo "Started worker $i (PID: ${pids[$i]})" | tee -a "$MAIN_LOG"
done

# Wait for completion
echo "" | tee -a "$MAIN_LOG"
echo "Waiting for all tasks to complete..." | tee -a "$MAIN_LOG"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    wait ${pids[$i]}
    echo "Worker $i finished" | tee -a "$MAIN_LOG"
done

echo "" | tee -a "$MAIN_LOG"
echo "All tasks completed!" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Aggregate results
echo "Aggregating results..." | tee -a "$MAIN_LOG"

python - <<PYTHON_EOF
import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

results_dir = "$RESULTS_DIR"
outputs_dir = Path(results_dir) / "outputs"

# Load task configuration
config_file = Path(results_dir) / "task_config.txt"
expected_duplicates = {}
if config_file.exists():
    with open(config_file) as f:
        for line in f:
            if "num_duplicates=" in line:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    task_name = parts[0].strip()
                    num_dup_str = parts[1].split("num_duplicates=")[1].split(",")[0].strip()
                    expected_duplicates[task_name] = int(num_dup_str)

aggregated = {}
print(f"\n{'='*80}")
print("MULTI-TRIAL RESULTS (Data Duplication Method)")
print(f"{'='*80}\n")

for task_dir in sorted(outputs_dir.glob("*")):
    if not task_dir.is_dir():
        continue

    task_name = task_dir.name
    result_files = glob.glob(str(task_dir / "**" / "results_*.json"), recursive=True)

    if not result_files:
        continue

    # Load results (should be one file with duplicated data)
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            if "results" not in data:
                continue

            # Check if this is already aggregated or raw
            results = data["results"]

            # Get the actual task key (not "all")
            task_keys = [k for k in results.keys() if k != "all"]
            if not task_keys:
                continue

            task_key = task_keys[0]
            metrics = results[task_key]

            num_dup = expected_duplicates.get(task_name, 1)

            print(f"Task: {task_name}")
            print(f"  Expected duplicates per question: {num_dup}")

            # If num_dup == 1, no aggregation needed
            if num_dup == 1:
                aggregated[task_name] = {}
                for metric_name, value in metrics.items():
                    if not metric_name.endswith("_stderr") and isinstance(value, (int, float)):
                        aggregated[task_name][metric_name] = {
                            "mean": float(value),
                            "std": 0.0,
                            "min": float(value),
                            "max": float(value),
                            "n_trials": 1
                        }
                        print(f"  {metric_name}: {value:.4f} (single run)")
            else:
                # Results are already averaged by lighteval
                # The modified main_duplicated.py should handle grouping
                aggregated[task_name] = {}
                for metric_name, value in metrics.items():
                    if not metric_name.endswith("_stderr"):
                        if isinstance(value, dict) and "mean" in value:
                            # Already aggregated
                            aggregated[task_name][metric_name] = value
                            mean = value["mean"]
                            std = value.get("std", 0)
                            if std > 0:
                                print(f"  {metric_name}: {mean:.4f} ± {std:.4f}")
                            else:
                                print(f"  {metric_name}: {mean:.4f}")
                        elif isinstance(value, (int, float)):
                            # Single value (not aggregated yet)
                            aggregated[task_name][metric_name] = {
                                "mean": float(value),
                                "std": 0.0,
                                "n_trials": 1
                            }
                            print(f"  {metric_name}: {value:.4f}")

            print()

        except Exception as e:
            print(f"Error reading {result_file}: {e}")

# Save aggregated results
output_file = Path(results_dir) / "aggregated_results.json"
with open(output_file, 'w') as f:
    json.dump({
        "task_config": expected_duplicates,
        "results": aggregated
    }, f, indent=2)

print(f"{'='*80}")
print(f"Results saved to: {output_file}")
print(f"{'='*80}\n")

PYTHON_EOF

echo "" | tee -a "$MAIN_LOG"
echo "Evaluation complete!" | tee -a "$MAIN_LOG"
echo "Results: $RESULTS_DIR/aggregated_results.json" | tee -a "$MAIN_LOG"
