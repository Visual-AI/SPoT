"""
Parallel Inference on dapo_5k dataset using vllm with 8 GPUs

This script performs parallel inference on the dapo_5k dataset using the Qwen3-8b model
with Ray for data parallelism across 8 GPUs.
"""

import os
import json
import re
import argparse
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import ray
from vllm import LLM, SamplingParams
from more_itertools import distribute
from tqdm import tqdm


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load the dapo_5k dataset from jsonl file."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_boxed_answer(text: str) -> str:
    """Extract the answer from $\\boxed{ANSWER}$ format."""
    # Try to find the boxed answer
    pattern = r'\$\\boxed\{([^}]+)\}\$'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Alternative pattern without dollar signs
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove whitespace
    answer = answer.strip()
    # Remove LaTeX formatting
    answer = answer.replace('\\', '').replace(' ', '')
    return answer.lower()


def verify_result(model_output: str, ground_truth: str) -> bool:
    """Verify if the model output contains the correct answer."""
    extracted_answer = extract_boxed_answer(model_output)
    if not extracted_answer:
        return False

    # Normalize both answers for comparison
    extracted_norm = normalize_answer(extracted_answer)
    ground_truth_norm = normalize_answer(ground_truth)

    return extracted_norm == ground_truth_norm


@ray.remote(num_gpus=1)
def run_inference_worker(
    model_name: str,
    data_chunk: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_thinking: bool,
    worker_id: int
) -> List[Dict[str, Any]]:
    """
    Worker function to run inference on a chunk of data.

    Args:
        model_name: Name or path of the model
        data_chunk: Chunk of data to process
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum number of tokens to generate
        enable_thinking: Whether to enable thinking mode
        worker_id: ID of the worker

    Returns:
        List of results with predictions and verification status
    """
    print(f"Worker {worker_id}: Initializing model on GPU...")

    # Initialize the model for this worker
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    # Get tokenizer to apply chat template
    tokenizer = llm.get_tokenizer()

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    print(f"Worker {worker_id}: Processing {len(data_chunk)} samples...")

    # Prepare prompts
    prompts = []
    for item in data_chunk:
        # Create chat message
        messages = [{"role": "user", "content": item["prompt"]}]

        # Apply chat template with or without thinking mode
        if enable_thinking:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Disable thinking mode by using enable_thinking=False
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

        prompts.append(prompt)

    # Run inference
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # Process results
    results = []
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        ground_truth = data_chunk[idx]["solution"]

        # Verify the result
        is_correct = verify_result(generated_text, ground_truth)

        result = {
            "index": data_chunk[idx].get("extra_info", {}).get("index", idx),
            "prompt": data_chunk[idx]["prompt"],
            "ground_truth": ground_truth,
            "model_output": generated_text,
            "is_correct": is_correct,
            "extracted_answer": extract_boxed_answer(generated_text),
            "data_source": data_chunk[idx].get("data_source", ""),
            "ability": data_chunk[idx].get("ability", ""),
        }

        results.append(result)

    print(f"Worker {worker_id}: Completed {len(results)} samples")

    return results


def main():
    parser = argparse.ArgumentParser(description="Parallel inference on dapo_5k dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/dapo_5k/data.jsonl",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/dapo_5k_inference",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for parallel inference"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode (default: disabled)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"Parallel Inference Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data Path: {args.data_path}")
    print(f"  Number of GPUs: {args.num_gpus}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Thinking Mode: {'Enabled' if args.enable_thinking else 'Disabled'}")
    print(f"  Output Directory: {args.output_dir}")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.data_path)

    if args.max_samples:
        dataset = dataset[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for debugging")

    print(f"Loaded {len(dataset)} samples")

    # Initialize Ray
    print("\nInitializing Ray...")
    ray.init(ignore_reinit_error=True)

    # Distribute data across workers
    print(f"\nDistributing data across {args.num_gpus} workers...")
    data_chunks = [list(chunk) for chunk in distribute(args.num_gpus, dataset)]

    for i, chunk in enumerate(data_chunks):
        print(f"  Worker {i}: {len(chunk)} samples")

    # Start inference
    print("\nStarting parallel inference...")
    start_time = datetime.now()

    # Create remote tasks
    tasks = [
        run_inference_worker.remote(
            model_name=args.model,
            data_chunk=chunk,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            worker_id=i
        )
        for i, chunk in enumerate(data_chunks) if len(chunk) > 0
    ]

    # Wait for all tasks to complete
    results_chunks = ray.get(tasks)

    # Flatten results
    all_results = []
    for chunk_results in results_chunks:
        all_results.extend(chunk_results)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Shutdown Ray
    ray.shutdown()

    # Calculate statistics
    correct_count = sum(1 for r in all_results if r["is_correct"])
    total_count = len(all_results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    # Separate errors and correct results
    errors = [r for r in all_results if not r["is_correct"]]
    correct_results = [r for r in all_results if r["is_correct"]]

    print("\n" + "="*80)
    print("Inference Complete!")
    print(f"  Total Samples: {total_count}")
    print(f"  Correct: {correct_count}")
    print(f"  Errors: {len(errors)}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Throughput: {total_count/duration:.2f} samples/second")
    print("="*80)

    # Save all results
    results_file = output_dir / "all_results.jsonl"
    print(f"\nSaving all results to {results_file}...")
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Save errors for further analysis
    errors_file = output_dir / "errors.jsonl"
    print(f"Saving {len(errors)} errors to {errors_file}...")
    with open(errors_file, 'w', encoding='utf-8') as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + '\n')

    # Save correct results
    correct_file = output_dir / "correct_results.jsonl"
    print(f"Saving {len(correct_results)} correct results to {correct_file}...")
    with open(correct_file, 'w', encoding='utf-8') as f:
        for result in correct_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Save summary statistics
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "model": args.model,
        "num_gpus": args.num_gpus,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "total_samples": total_count,
        "correct_count": correct_count,
        "error_count": len(errors),
        "accuracy": accuracy,
        "throughput_samples_per_second": total_count / duration,
    }

    summary_file = output_dir / "summary.json"
    print(f"Saving summary to {summary_file}...")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("All files saved successfully!")
    print(f"  Results: {results_file}")
    print(f"  Errors: {errors_file}")
    print(f"  Correct: {correct_file}")
    print(f"  Summary: {summary_file}")
    print("="*80)


if __name__ == "__main__":
    main()
