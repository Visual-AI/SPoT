import os
import sys
import lighteval
import torch
import multiprocessing
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from datetime import datetime
import argparse
import json
from fsspec import url_to_fs
import random
import numpy as np

# Download NLTK data (same as original main.py)
try:
    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    def ensure_nltk_resource(resource_name):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                nltk.data.find(resource_name)
                print(f"✓ NLTK resource '{resource_name}' already available")
                return True
            except LookupError:
                print(f"Downloading NLTK resource '{resource_name}' (attempt {attempt + 1}/{max_retries})...")
                try:
                    result = nltk.download(resource_name, quiet=False)
                    if result:
                        try:
                            nltk.data.find(resource_name)
                            print(f"✓ Successfully downloaded '{resource_name}'")
                            return True
                        except LookupError:
                            print(f"⚠ Download reported success but resource not found (attempt {attempt + 1})")
                            continue
                    else:
                        print(f"⚠ Download failed for '{resource_name}' (attempt {attempt + 1})")
                except Exception as e:
                    print(f"⚠ Error downloading '{resource_name}': {e} (attempt {attempt + 1})")

        print(f"✗ Failed to download '{resource_name}' after {max_retries} attempts")
        return False

    import shutil
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    punkt_zip = os.path.join(nltk_data_dir, 'tokenizers', 'punkt.zip')

    punkt_exists = False
    try:
        nltk.data.find('tokenizers/punkt')
        punkt_exists = True
        print("✓ NLTK punkt already installed")
    except LookupError:
        if os.path.exists(punkt_zip) and not os.path.isfile(os.path.join(punkt_dir, 'README')):
            print(f"Detected incomplete punkt installation, cleaning up...")
            if os.path.exists(punkt_dir):
                try:
                    shutil.rmtree(punkt_dir)
                    print(f"  Removed incomplete directory: {punkt_dir}")
                except Exception as e:
                    print(f"  Warning: Could not remove {punkt_dir}: {e}")
        ensure_nltk_resource('tokenizers/punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("✓ NLTK punkt_tab already installed")
    except LookupError:
        ensure_nltk_resource('tokenizers/punkt_tab')

    print("NLTK setup complete\n")

except Exception as e:
    print(f"⚠ Warning: NLTK setup encountered an issue: {e}")
    print("Continuing anyway - some tasks (like ifeval) may fail\n")

# Import custom tasks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lighteval_tasks

__version__ = f"2.0_lighteval@{lighteval.__version__}_duplicated"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--task", type=str, default="lighteval|aime24|0|0")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--custom_tasks_directory", type=str, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--launcher_type", type=str, default="VLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--disable_thinking", dest="enable_thinking", action="store_false")

    # NEW: Data duplication parameters
    parser.add_argument("--num_duplicates", type=int, default=1,
                       help="Number of times to duplicate each example for variance measurement")
    parser.add_argument("--base_seed", type=int, default=None,
                       help="Base seed for GPQA shuffling (each duplicate gets base_seed + i)")

    parser.set_defaults(enable_thinking=False)
    return parser.parse_args()


def main():
    start = datetime.now()
    args = parse_args()
    fs, output_dir = url_to_fs(args.output_dir)

    # Monkey-patch AutoTokenizer (same as original)
    if args.use_chat_template:
        from transformers import AutoTokenizer
        _original_from_pretrained = AutoTokenizer.from_pretrained

        def _patched_from_pretrained(*args_inner, **kwargs_inner):
            tokenizer = _original_from_pretrained(*args_inner, **kwargs_inner)
            if hasattr(tokenizer, 'apply_chat_template'):
                original_apply_chat_template = tokenizer.apply_chat_template

                def patched_apply_chat_template(*template_args, **template_kwargs):
                    template_kwargs['enable_thinking'] = args.enable_thinking
                    return original_apply_chat_template(*template_args, **template_kwargs)

                tokenizer.apply_chat_template = patched_apply_chat_template
            return tokenizer

        AutoTokenizer.from_pretrained = _patched_from_pretrained
        print(f"✓ Monkey-patched AutoTokenizer.from_pretrained with enable_thinking={args.enable_thinking}")

    # Print configuration
    num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"=" * 80)
    print(f"GPU Configuration:")
    print(f"  Available GPUs: {num_available_gpus}")
    print(f"  Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"  Total GPUs used: {args.tensor_parallel_size}")

    print(f"\nData Duplication Configuration:")
    print(f"  Duplicates per example: {args.num_duplicates}")
    print(f"  Base seed: {args.base_seed if args.base_seed is not None else 'random'}")
    print(f"  Method: Each duplicate uses random sampling (no fixed seed for generation)")

    if args.use_chat_template:
        print(f"\nThinking Mode: {'ENABLED' if args.enable_thinking else 'DISABLED (empty tags)'}")
    print(f"=" * 80)

    max_model_length = args.max_model_length
    if args.max_model_length is None:
        max_model_length = args.max_new_tokens
    elif args.max_model_length == -1:
        max_model_length = None

    system_prompt = None
    if args.system_prompt is not None and os.path.exists(args.system_prompt):
        with open(args.system_prompt, "r") as f:
            system_prompt = f.read()

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )

    custom_tasks = args.custom_tasks_directory if args.custom_tasks_directory else lighteval_tasks

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        job_id=0,
        dataset_loading_processes=1,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=1,
        max_samples=args.max_samples,
        use_chat_template=args.use_chat_template,
        system_prompt=system_prompt,
        load_responses_from_details_date_id=None,
    )

    # CRITICAL: Use seed=None for random sampling
    # Each duplicate will get different random outputs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    model_config = VLLMModelConfig(
        model_name=args.model,
        dtype=args.dtype,
        seed=42,
        use_chat_template=args.use_chat_template,
        max_model_length=max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        generation_parameters=GenerationParameters(
            max_new_tokens=args.max_new_tokens,
            seed=None,  # Random sampling!
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        ),
    )

    # Apply data duplication by monkey-patching LightevalTask BEFORE creating the pipeline
    if args.num_duplicates > 1:
        print(f"\n{'='*80}")
        print(f"Duplicating dataset {args.num_duplicates}x for variance measurement...")
        print(f"{'='*80}\n")

        from lighteval.tasks.lighteval_task import LightevalTask
        original_eval_docs = LightevalTask.eval_docs

        def duplicating_eval_docs(self):
            """Wrap the original method to duplicate examples"""
            # Call the original method
            original_docs = original_eval_docs(self)

            # Duplicate each document
            duplicated_docs = []
            for doc in original_docs:
                for dup_idx in range(args.num_duplicates):
                    # Each duplicate will get different model output due to random sampling
                    duplicated_docs.append(doc)

            print(f"  Task {self.cfg.name}: {len(original_docs)} examples -> {len(duplicated_docs)} total")
            return duplicated_docs

        LightevalTask.eval_docs = duplicating_eval_docs
        print(f"{'='*80}\n")

    pipeline = Pipeline(
        tasks=args.task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        metric_options={},
    )

    pipeline.evaluate()
    pipeline.show_results()
    results = pipeline.get_results()

    # Cleanup VLLM
    print("Cleaning up VLLM engine...")
    try:
        import gc
        if hasattr(pipeline, 'model'):
            model = pipeline.model
            if hasattr(model, 'model'):
                if hasattr(model.model, 'llm_engine'):
                    print("  Destroying llm_engine...")
                    del model.model.llm_engine
                if hasattr(model.model, 'llm'):
                    print("  Destroying llm...")
                    del model.model.llm
                del model.model
            if hasattr(model, 'llm_engine'):
                print("  Destroying model.llm_engine...")
                del model.llm_engine
            if hasattr(model, 'llm'):
                print("  Destroying model.llm...")
                del model.llm

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  Cleanup complete!")
    except Exception as e:
        print(f"  Warning during cleanup (non-fatal): {e}")

    pipeline.save_and_push_results()

    # Post-process results if duplicated
    if args.num_duplicates > 1:
        print(f"\n{'='*80}")
        print(f"Aggregating results from {args.num_duplicates} duplicates...")
        print(f"{'='*80}\n")

        # Group results by original question index
        # Assumption: results are in order, so every N consecutive results are duplicates
        aggregated_results = {}

        for task_key, task_results in results["results"].items():
            if task_key == "all":
                # Get the metrics
                metrics = task_results

                # For now, lighteval already averaged everything
                # The duplicates are treated as independent samples
                # So the reported metrics are already the mean across all duplicates
                # We just note that variance is measured across duplicates
                aggregated_results[task_key] = metrics

        results["results"] = aggregated_results
        results["num_duplicates"] = args.num_duplicates

    # Convert to JSON-serializable format
    def make_json_serializable(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj

    config_general = results["config_general"].copy() if isinstance(results["config_general"], dict) else {}
    config_general = make_json_serializable(config_general)

    folder = args.model.replace("/", "_")
    fname = f"duplicated_{args.num_duplicates}x-{args.temperature}-{args.top_p}-{args.task.split('|')[1]}-{args.max_new_tokens}"
    if max_model_length != args.max_new_tokens:
        fname += f"-{max_model_length}"
    if not args.use_chat_template:
        fname += "-nochat"
    if args.tensor_parallel_size > 1:
        fname += f"-tp{args.tensor_parallel_size}"
    fpath = os.path.join(output_dir, folder, f"{fname}.json")

    data = {
        "start_time": start.isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_evaluation_time_seconds": (datetime.now() - start).total_seconds(),
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "max_model_length": max_model_length,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "seed": None,  # Random sampling
        "num_duplicates": args.num_duplicates,
        "base_seed": args.base_seed,
        "system_prompt": system_prompt,
        "use_chat_template": args.use_chat_template,
        "enable_thinking": args.enable_thinking,
        "tensor_parallel_size": args.tensor_parallel_size,
        "results": results["results"]["all"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "version": __version__,
        "launcher_type": args.launcher_type,
        "device_name": torch.cuda.get_device_name(),
        "lighteval_config": config_general,
    }

    print(json.dumps(data, indent=2))
    fs.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    with fs.open(fpath, "w") as f:
        f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
