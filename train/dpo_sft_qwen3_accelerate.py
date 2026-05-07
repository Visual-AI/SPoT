#!/usr/bin/env python3
"""
DPO + SFT combined training for Qwen3-8B (Full Finetune).
Compatible with accelerate launch (DeepSpeed/FSDP).

This script trains Qwen3 using a combination of:
- DPO (Direct Preference Optimization): Learns to prefer gemini_corrected_answer over original_answer
- SFT (Supervised Fine-Tuning): Learns to generate correct answers via cross-entropy loss

Usage example:
    accelerate launch --config_file train/accelerate_config.yaml train/dpo_sft_qwen3_accelerate.py \
        --model_name_or_path Qwen/Qwen3-8B \
        --loss_type "bco_pair" \
        --output_dir output/model \
        --learning_rate 1e-6 \
        --num_train_epochs 2

Note: Use --loss_weights_str (not --loss_weights) for comma-separated weights.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import patches  # noqa: F401  — free ref_model after precompute

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
)

logger = logging.get_logger(__name__)


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments for Qwen3 DPO+SFT training."""
    data_file: str = field(default="data/combined_filtered_max60change.jsonl")
    wandb_project: Optional[str] = field(default="dpo-sft-qwen3")
    wandb_entity: Optional[str] = field(default=None)
    length_limit: int = field(default=8192)
    loss_weights_str: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated loss weights (e.g., '1.0,0.1')"}
    )


def format_data_for_dpo(example):
    """Format the dataset for DPO training.

    DPO requires:
    - prompt: The user's question (as messages list)
    - chosen: The preferred response (gemini_corrected_answer)
    - rejected: The less preferred response (original_answer)
    """
    question = example.get("question", example.get("prompt", ""))
    chosen_answer = example.get("gemini_corrected_answer", "")
    rejected_answer = example.get("original_answer", "")

    if chosen_answer.startswith("\n\n"):
        chosen_answer = chosen_answer[2:]
    if rejected_answer.startswith("\n\n"):
        rejected_answer = rejected_answer[2:]

    return {
        "prompt": [{"role": "user", "content": question}],
        "chosen": [{"role": "assistant", "content": chosen_answer}],
        "rejected": [{"role": "assistant", "content": rejected_answer}],
    }


def parse_loss_config(loss_type_str: str, loss_weights_str: Optional[str] = None):
    """Parse loss configuration from command line strings."""
    if ',' in loss_type_str:
        loss_types = [lt.strip() for lt in loss_type_str.split(',')]
    else:
        loss_types = [loss_type_str]

    loss_weights = None
    if loss_weights_str:
        weight_strs = loss_weights_str.split(',')
        loss_weights = [float(w.strip()) for w in weight_strs]
        if len(loss_weights) != len(loss_types):
            raise ValueError(
                f"Number of loss weights ({len(loss_weights)}) must match "
                f"number of loss types ({len(loss_types)})"
            )

    return loss_types, loss_weights


def main(script_args, training_args, model_args):
    # 1. Set up WandB
    if script_args.wandb_project:
        os.environ['WANDB_PROJECT'] = script_args.wandb_project
    if script_args.wandb_entity:
        os.environ['WANDB_ENTITY'] = script_args.wandb_entity

    # 2. Set key training parameters
    training_args.gradient_checkpointing = True
    training_args.max_length = script_args.length_limit
    training_args.max_prompt_length = script_args.length_limit // 2

    # 3. Handle loss configuration
    if isinstance(training_args.loss_type, str):
        loss_types, loss_weights = parse_loss_config(
            training_args.loss_type, script_args.loss_weights_str
        )
        training_args.loss_type = loss_types
        if loss_weights is not None:
            training_args.loss_weights = loss_weights

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Load policy model
    # Do NOT use device_map="auto" — let accelerate handle device placement
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    logger.info("Loading Policy Model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        use_cache=False,  # Required when gradient_checkpointing=True
        trust_remote_code=model_args.trust_remote_code,
    )

    # 6. Load reference model explicitly for full finetune
    # Explicit load (rather than letting TRL deepcopy) is more stable with FSDP/DeepSpeed
    logger.info("Loading Reference Model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        use_cache=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 7. Load and format dataset
    with training_args.main_process_first(desc="dataset processing"):
        logger.info(f"Loading dataset from {script_args.data_file}...")
        dataset = load_dataset("json", data_files=script_args.data_file, split="train")
        dataset = dataset.map(
            format_data_for_dpo,
            remove_columns=dataset.column_names,
            num_proc=os.cpu_count() // 2,
        )
        logger.info(f"Dataset formatted. Total examples: {len(dataset)}")

    # 8. Initialize DPOTrainer (full finetune, no peft_config)
    logger.info("Initializing DPOTrainer (Full Finetune)...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 9. Train
    logger.info("Starting DPO + SFT combined training...")
    trainer.train()

    # 10. Save — trainer.save_model handles weight merging in distributed environments
    logger.info("Saving model...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Convert saved checkpoint from fp32 to bf16
    # DeepSpeed ZeRO-2 saves fp32 master weights; re-save as bf16 to halve disk usage
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main:
        logger.info("Converting checkpoint to bf16...")
        _m = AutoModelForCausalLM.from_pretrained(
            training_args.output_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        _m.save_pretrained(training_args.output_dir, max_shard_size="5GB")
        del _m
        logger.info("Saved bf16 checkpoint.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args)
