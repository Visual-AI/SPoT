#!/bin/bash

LOCAL_DIR="<PATH TO LOCAL REPO>"
OUTPUT_DIR="<PATH TO OUTPUT DIR>"
PARTITION=<PARTITION>
VENV=<PATH TO VENV>/bin/activate
mkdir -p $OUTPUT_DIR/logs

MODELS=(
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    knoveleng/Open-RS3
    knoveleng/Open-RS2
    knoveleng/Open-RS1
    agentica-org/DeepScaleR-1.5B-Preview
    simplescaling/s1.1-7B
    open-thoughts/OpenThinker-7B
    Intelligent-Internet/II-Thought-1.5B-Preview
    # sail/Qwen2.5-Math-1.5B-Oat-Zero
    # simplescaling/s1.1-32B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    # open-thoughts/OpenThinker-32B
    # GAIR/LIMO
    # bespokelabs/Bespoke-Stratos-32B
)

TOP_PS=(
    0.9
)


TEMPS=(
    0.8
)

MAX_MODEL_LENGTHS=(
    32768
)

MAX_TOKENS_LIST=(
    # 4096
    # 8192
    # 16384
    32768
)

for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
for MODEL in "${MODELS[@]}"; do
for TOP_P in "${TOP_PS[@]}"; do
for TEMP in "${TEMPS[@]}"; do
echo "Submitting $MODEL job for temperature $TEMP, top_p $TOP_P, MAX_MODEL_LENGTH $MAX_MODEL_LENGTH, MAX_TOKENS $MAX_TOKENS"
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=eval-$MODEL-$SEED-$TEMP-$TOP_P-$MAX_MODEL_LENGTH-$MAX_TOKENS
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=$OUTPUT_DIR/logs/%j.out
#SBATCH --error=$OUTPUT_DIR/logs/%j.err
#SBATCH --partition=$PARTITION

source $VENV
cd $LOCAL_DIR

set -x

SEEDS=(
    0
    1
    2
)

TASKS=(
    "custom|aime24|0|0"
    "custom|math_500|0|0"
    "custom|amc23|0|0"
    "custom|aime25|0|0"
    # "custom|gpqa:diamond|0|0"
    "custom|minerva|0|0"
    "custom|olympiadbench|0|0"
)
for SEED in "\${SEEDS[@]}"; do
for TASK in "\${TASKS[@]}"; do
    python main.py \
        --model $MODEL \
        --task \$TASK \
        --temperature $TEMP \
        --top_p $TOP_P \
        --seed \$SEED \
        --output_dir $OUTPUT_DIR \
        --max_new_tokens $MAX_TOKENS \
        --max_model_length $MAX_MODEL_LENGTH \
        --custom_tasks_directory lighteval_tasks.py \
        --use_chat_template
done
done
EOT

done
done
done
done
done
