#!/bin/bash
# =============================================================================
# GRPO training for Qwen3-8B (non-thinking mode) on DAPO math data
# Hardware: 8x H800 (80GB) GPUs, single node
# verl v0.6.0 + vLLM 0.9.0 + torch 2.7.0
#
# Key design choices:
#   1. Non-thinking mode: enable_thinking=False passed to Qwen3 chat template
#      so the model generates directly without <think>...</think> blocks.
#   2. Data: 4k randomly sampled DAPO math problems with rule-based reward.
#      data_source="math_dapo" → verl dispatches to math_dapo.compute_score
#      which uses Minerva-style answer extraction (+1 correct, -1 incorrect).
#   3. GRPO with n=8 group sampling, KL loss regularization (no KL in reward).
#   4. Rollout via vLLM with TP=1 (8B model fits on single H800 with room
#      for KV cache; TP=1 gives 8 parallel rollout workers for max throughput).
#   5. Longer response (4096 tokens) since non-thinking mode needs the model
#      to show all reasoning steps in the visible output.
#
# Prerequisites:
#   1. Install verl:  pip install --no-deps -e .  (from verl repo root)
#   2. Preprocess data:
#      python3 examples/data_preprocess/dapo_math.py \
#          --data_file /path/to/dapo_full/data.jsonl \
#          --local_save_dir ~/data/dapo_math_4k \
#          --num_samples 4000 --seed 42
#   3. Model weights: Qwen/Qwen3-8B (auto-downloaded from HuggingFace)
#   4. wandb login  (for experiment tracking)
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3-8b.sh
#   # or with overrides:
#   bash examples/grpo_trainer/run_qwen3-8b.sh trainer.total_epochs=20
# =============================================================================

set -x

# ------- Configurable paths (override via env vars) -------
DATA_DIR=${DATA_DIR:-"${HOME}/data/dapo_math_4k"}
TRAIN_DATA=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
VAL_DATA=${TEST_FILE:-"${DATA_DIR}/test.parquet"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}

# ------- Sanity check -------
if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${VAL_DATA}" ]; then
    echo "ERROR: Data files not found:"
    echo "  TRAIN: ${TRAIN_DATA}"
    echo "  VAL:   ${VAL_DATA}"
    echo ""
    echo "Run data preprocessing first:"
    echo "  python3 examples/data_preprocess/dapo_math.py \\"
    echo "      --data_file /path/to/dapo_full/data.jsonl \\"
    echo "      --local_save_dir ${DATA_DIR} \\"
    echo "      --num_samples 4000 --seed 42"
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_dapo_math' \
    trainer.experiment_name='qwen3_8b_no_thinking' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 $@
