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

# Download NLTK data required for evaluation (especially for ifeval and gpqa)
# This must happen before importing lighteval to avoid race conditions
try:
    import nltk
    import ssl

    # Handle SSL certificate issues that may occur in some environments
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Function to robustly download NLTK resource
    def ensure_nltk_resource(resource_name):
        """Download NLTK resource if not present, with retry logic."""
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
                        # Verify the download by trying to find it again
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

    # Download required NLTK resources
    # punkt_tab is the newer version, but lighteval still uses punkt
    # Only download if not already present to avoid repeated downloads
    import shutil

    # Handle the common FileExistsError when punkt directory is incomplete
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    punkt_zip = os.path.join(nltk_data_dir, 'tokenizers', 'punkt.zip')

    # Check if punkt already exists and is complete
    punkt_exists = False
    try:
        nltk.data.find('tokenizers/punkt')
        punkt_exists = True
        print("✓ NLTK punkt already installed")
    except LookupError:
        # If punkt.zip exists but punkt directory is incomplete, clean up
        if os.path.exists(punkt_zip) and not os.path.isfile(os.path.join(punkt_dir, 'README')):
            print(f"Detected incomplete punkt installation, cleaning up...")
            if os.path.exists(punkt_dir):
                try:
                    shutil.rmtree(punkt_dir)
                    print(f"  Removed incomplete directory: {punkt_dir}")
                except Exception as e:
                    print(f"  Warning: Could not remove {punkt_dir}: {e}")

        # Download punkt if not present
        ensure_nltk_resource('tokenizers/punkt')

    # Also check punkt_tab
    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("✓ NLTK punkt_tab already installed")
    except LookupError:
        ensure_nltk_resource('tokenizers/punkt_tab')

    print("NLTK setup complete\n")

except Exception as e:
    print(f"⚠ Warning: NLTK setup encountered an issue: {e}")
    print("Continuing anyway - some tasks (like ifeval) may fail\n")

# Import custom tasks from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lighteval_tasks

__version__ = f"2.0_lighteval@{lighteval.__version__}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path (default: same as model). Use this to specify a custom tokenizer for controlling chat template behavior like enable_thinking.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--task", type=str, default="lighteval|aime24|0|0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--custom_tasks_directory", type=str, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--launcher_type", type=str, default="VLLM")

    # Auto-detect number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (splits model across GPUs). Use for large models that don't fit on 1 GPU. Default: 1",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (useful for debugging). Default: None (all samples)",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to be processed in a single batch. Should be >= max_new_tokens for long generations. Default: None (vLLM default)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode (allow model to generate reasoning in <think> tags). Default: False (adds empty <think></think> tags)",
    )
    parser.add_argument(
        "--disable_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking mode (adds empty <think></think> tags). This is the default.",
    )
    parser.set_defaults(enable_thinking=False)
    return parser.parse_args()


def main():
    start = datetime.now()
    args = parse_args()
    fs, output_dir = url_to_fs(args.output_dir)

    # Monkey-patch AutoTokenizer to inject enable_thinking parameter
    # This must happen before any model loading
    if args.use_chat_template:
        from transformers import AutoTokenizer
        _original_from_pretrained = AutoTokenizer.from_pretrained

        def _patched_from_pretrained(*args_inner, **kwargs_inner):
            tokenizer = _original_from_pretrained(*args_inner, **kwargs_inner)

            # Wrap the tokenizer's apply_chat_template method
            if hasattr(tokenizer, 'apply_chat_template'):
                original_apply_chat_template = tokenizer.apply_chat_template

                def patched_apply_chat_template(*template_args, **template_kwargs):
                    # Inject enable_thinking parameter
                    template_kwargs['enable_thinking'] = args.enable_thinking
                    return original_apply_chat_template(*template_args, **template_kwargs)

                tokenizer.apply_chat_template = patched_apply_chat_template

            return tokenizer

        AutoTokenizer.from_pretrained = _patched_from_pretrained
        print(f"✓ Monkey-patched AutoTokenizer.from_pretrained with enable_thinking={args.enable_thinking}")

    # Print GPU configuration
    num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"=" * 80)
    print(f"GPU Configuration:")
    print(f"  Available GPUs: {num_available_gpus}")
    print(f"  Tensor Parallel Size: {args.tensor_parallel_size}", end="")
    if args.tensor_parallel_size > 1:
        print(f" (model split across {args.tensor_parallel_size} GPUs)")
    else:
        print(f" (model on single GPU)")

    total_gpus_used = args.tensor_parallel_size
    print(f"  Total GPUs used: {total_gpus_used}")

    if total_gpus_used > num_available_gpus:
        print(f"  ⚠️  WARNING: Requested {total_gpus_used} GPUs but only {num_available_gpus} available!")

    if args.max_samples:
        print(f"\n⚠️  DEBUG MODE: Limiting to {args.max_samples} samples")
        print(f"  Results are NOT representative - for debugging only!")

    if args.use_chat_template:
        print(f"\nThinking Mode: {'ENABLED' if args.enable_thinking else 'DISABLED (empty tags)'}")
    print(f"=" * 80)

    max_model_length = args.max_model_length
    if args.max_model_length is None:
        print("max_model_length not set. Setting it to max_new_tokens.")
        max_model_length = args.max_new_tokens
    elif args.max_model_length == -1:
        print("max_model_length is -1. Setting it to None.")
        max_model_length = None

    folder = args.model.replace("/", "_")
    fname = f"{args.seed}-{args.temperature}-{args.top_p}-{args.task.split('|')[1]}-{args.max_new_tokens}"
    if max_model_length != args.max_new_tokens:
        fname += f"-{max_model_length}"
    if not args.use_chat_template:
        fname += "-nochat"
    if args.tensor_parallel_size > 1:
        fname += f"-tp{args.tensor_parallel_size}"
    fpath = os.path.join(output_dir, folder, f"{fname}.json")
    if fs.exists(fpath) and not args.overwrite:
        print(f"File {fpath} already exists. Skipping.")
        return

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
    assert args.launcher_type == "VLLM", "Only VLLM is supported for now"

    # Use custom tasks module if custom_tasks_directory is specified, otherwise use imported module
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

    model_config = VLLMModelConfig(
        model_name=args.model,
        dtype=args.dtype,
        seed=args.seed,
        use_chat_template=args.use_chat_template,
        max_model_length=max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        generation_parameters=GenerationParameters(
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        ),
    )

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

    # CRITICAL: Explicitly destroy VLLM engine to prevent hang on shutdown
    # Without this, VLLM worker threads can hang for ~1 hour waiting for process group cleanup
    print("Cleaning up VLLM engine...")
    try:
        import gc
        # Try to access and cleanup VLLM engine at various nesting levels
        if hasattr(pipeline, 'model'):
            model = pipeline.model
            # Check common VLLM model wrapper patterns
            if hasattr(model, 'model'):
                if hasattr(model.model, 'llm_engine'):
                    print("  Destroying llm_engine...")
                    del model.model.llm_engine
                if hasattr(model.model, 'llm'):
                    print("  Destroying llm...")
                    del model.model.llm
                del model.model
            # Also try direct cleanup
            if hasattr(model, 'llm_engine'):
                print("  Destroying model.llm_engine...")
                del model.llm_engine
            if hasattr(model, 'llm'):
                print("  Destroying model.llm...")
                del model.llm

        # Force garbage collection and CUDA cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  Cleanup complete!")
    except Exception as e:
        print(f"  Warning during cleanup (non-fatal): {e}")

    pipeline.save_and_push_results()

    # Helper function to recursively convert Pydantic objects to dicts
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

    # Convert config_general to JSON-serializable format
    config_general = results["config_general"].copy() if isinstance(results["config_general"], dict) else {}
    config_general = make_json_serializable(config_general)

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
        "seed": args.seed,
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
    # Set multiprocessing start method to 'spawn' to avoid CUDA re-initialization errors
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    main()
