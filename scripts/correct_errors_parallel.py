import json
import re
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import threading



# Thread-safe file writing
file_lock = threading.Lock()

def create_correction_prompt(question, student_answer):
    """Create the prompt for Gemini to correct the student's answer."""
    prompt = f"""Act as a helpful teaching assistant. Your goal is to revise a student model's answer to make it correct, while maintaining the student model's original writing style, tone, and formatting. The final result should look as if the student model had solved the problem correctly on its first try.

Question: {question}
Student Model's Answer: {student_answer}

You should first solve the problem independently and do the following:
1. Identify the correct parts of the student model's answer and keep them.
2. Replace the incorrect parts with correct reasoning.
3. Carefully match the student model's original writing style, including their tone, vocabulary, formatting and sentence structure.

**IMPORTANT OUTPUT FORMAT:**
1. First output ``=== CORRECTED STARTED ==='' followed by the corrected answer
2. Ends with the corrected answer in the format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$.'
3. Then output ``=== CORRECTED ENDED ==='' at the end of the corrected trace
4. Do not output meta-phrases like "Here is the corrected version"

 """
    return prompt

def extract_thinking_and_response(stream_response):
    """Extract thinking content and final response from streamed Gemini output."""
    thinking_content = []
    response_content = []
    current_section = "response"

    for chunk in stream_response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content

            if "<think>" in content:
                current_section = "thinking"
                parts = content.split("<think>")
                if parts[0]:
                    response_content.append(parts[0])
                if len(parts) > 1 and parts[1]:
                    thinking_content.append(parts[1])
            elif "</think>" in content:
                parts = content.split("</think>")
                if parts[0]:
                    thinking_content.append(parts[0])
                current_section = "response"
                if len(parts) > 1 and parts[1]:
                    # note, remove the first two '\n\n'
                    if parts[1][0]=='\n' and parts[1][1]=='\n':
                        parts[1]=parts[1][2:]
                    response_content.append(parts[1])
            else:
                if current_section == "thinking":
                    thinking_content.append(content)
                else:
                    response_content.append(content)

    return "".join(thinking_content), "".join(response_content)

def extract_boxed_answer(text):
    """Extract the answer from \\boxed{} format."""
    pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    return None

def remove_embedded_system_prompt(prompt):
    """Remove embedded system instructions from the beginning of a prompt.

    Common patterns to remove:
    - "Solve the following math problem efficiently and clearly..."
    - Similar instruction prefixes

    Returns just the actual question/problem.
    """
    # Pattern to match common system prompt prefixes
    # This looks for text ending with "Think step by step before answering." followed by newlines
    pattern = r"^.*?Think step by step before answering\.\s*\n+(.*)$"
    match = re.search(pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no match, check for other common patterns
    # Pattern for boxed answer format instructions
    pattern2 = r"^.*?\\boxed\{ANSWER\}\$\..*?\n+(.*)$"
    match2 = re.search(pattern2, prompt, re.DOTALL)
    if match2:
        return match2.group(1).strip()

    # If no embedded system prompt found, return original
    return prompt

def call_gemini_streaming(prompt, system_prompt=None, max_retries=5, initial_delay=1):
    """Call Gemini 2.5 Pro with streaming and return thinking + response.

    Args:
        prompt: The prompt to send to the API
        system_prompt: Optional system prompt to guide the model's behavior
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1)
    """
    for attempt in range(max_retries + 1):
        try:
            # Build messages list
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })

            stream = client.chat.completions.create(
                model="gemini-2.5-pro",
                messages=messages,
                stream=True,
                extra_body={
                    'extra_body': {
                        "google": {
                            "thinking_config": {
                                "thinking_budget": 16384,
                                "include_thoughts": True
                            },
                            'thought_tag_marker': 'think'
                        }
                    }
                }
            )

            thinking, response = extract_thinking_and_response(stream)
            return thinking, response
        except Exception as e:
            if attempt < max_retries:
                # Calculate exponential backoff delay
                delay = initial_delay * (2 ** attempt)
                print(f"\nError calling Gemini API (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"\nError calling Gemini API after {max_retries + 1} attempts: {e}")
                return None, None

def load_processed_indices(output_file):
    """Load indices that have already been processed."""
    processed = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line)
                        # Handle both formats: direct index or nested in extra_info
                        if 'index' in entry and entry['index']:
                            processed.add(entry['index'])
                        elif 'extra_info' in entry and 'index' in entry['extra_info']:
                            processed.add(entry['extra_info']['index'])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON at line {line_num} in {output_file}: {e}")
                    except Exception as e:
                        print(f"Warning: Error reading line {line_num} in {output_file}: {e}")
            print(f"Loaded {len(processed)} already processed entries from {output_file}")
        except Exception as e:
            print(f"Warning: Could not read output file {output_file}: {e}")
    return processed

def process_single_error(error, output_file, correct_output_file, use_direct_prompt=False, correct_count_tracker=None):
    """Process a single error and save result immediately.

    Args:
        error: The error entry to process
        output_file: Path to output file
        correct_output_file: Path to save correct answers
        use_direct_prompt: If True, use direct mode with original_prompt and solution;
                          If False, use create_correction_prompt (default)
        correct_count_tracker: Dict with 'count' key to track correct responses
    """
    try:
        # Choose prompt and ground truth based on mode
        if use_direct_prompt:
            # Direct mode: use original_prompt from data.jsonl
            question = error.get('original_prompt', error['prompt'])
            ground_truth = error.get('solution', error.get('ground_truth', ''))
            student_answer = None  # No student answer in direct mode

            # Use the original prompt directly
            # prompt_to_use = question
            # system_prompt= "Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$.' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering."
            prompt_to_add ="""Solve the following math problem. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$.' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.
            For example, 
            **Input**:
            Blue rolls a fair $n$-sided die numbered with integers from $1$ to $n$, and then flips a coin. The coin is weighted to land heads either $\\frac{1}{3}$ or $\\frac{2}{3}$ of the time. Given that the probability of both rolling a $7$ and flipping heads is $\\frac{1}{15}$, find $n$.
            **Output**:
            Let $R$ be the event of rolling the $n$-sided die, and let $C$ be the event of flipping the coin.\nLet $P(R=k)$ be the probability of rolling the number $k$. Since the die is a fair $n$-sided die numbered from $1$ to $n$, the probability of rolling any specific number $k$ (where $1 \\le k \\le n$) is $\\frac{1}{n}$.\n\nFor the event of rolling a $7$ to be possible, the number of sides $n$ must be greater than or equal to $7$. If $n < 7$, it is impossible to roll a $7$, and the probability would be $0$.\nAssuming $n \\ge 7$, the probability of rolling a $7$ is $P(R=7) = \\frac{1}{n}$.\n\nLet $H$ be the event of the coin landing on heads. The problem states that the probability of flipping heads, $P(H)$, is either $\\frac{1}{3}$ or $\\frac{2}{3}$.\n\nThe event of rolling a $7$ and the event of flipping heads are independent. Therefore, the probability of both events occurring is the product of their individual probabilities:\n$P(R=7 \\text{ and } H) = P(R=7) \\times P(H)$.\n\nWe are given that this combined probability is $\\frac{1}{15}$. So, we have the equation:\n$\\frac{1}{n} \\times P(H) = \\frac{1}{15}$.\n\nWe need to consider two cases for the value of $P(H)$.\n\nCase 1: The probability of flipping heads is $\\frac{1}{3}$.\nIn this case, our equation becomes:\n$\\frac{1}{n} \\times \\frac{1}{3} = \\frac{1}{15}$\n$\\frac{1}{3n} = \\frac{1}{15}$\nBy cross-multiplication, we get:\n$3n = 15$\n$n = 5$\nHowever, we established that for it to be possible to roll a $7$, we must have $n \\ge 7$. Since $n=5$ contradicts this condition, this case is not possible.\n\nCase 2: The probability of flipping heads is $\\frac{2}{3}$.\nIn this case, our equation becomes:\n$\\frac{1}{n} \\times \\frac{2}{3} = \\frac{1}{15}$\n$\\frac{2}{3n} = \\frac{1}{15}$\nBy cross-multiplication, we get:\n$2 \\times 15 = 3n \\times 1$\n$30 = 3n$\n$n = 10$\nThis value, $n=10$, is consistent with our condition that $n \\ge 7$. Therefore, this must be the correct value for $n$.\n\nLet's check our answer. If $n=10$, the probability of rolling a $7$ is $\\frac{1}{10}$. If the probability of flipping heads is $\\frac{2}{3}$, the combined probability is $\\frac{1}{10} \\times \\frac{2}{3} = \\frac{2}{30} = \\frac{1}{15}$. This matches the information given in the problem.\n\nTherefore, the final answer is: $\\boxed{10}$.
            
            **Input**

            """
            prompt_to_use = prompt_to_add+question+'\n**Output**'
            system_prompt=None
            # system_prompt='Please solve the math problem clearly and provide your final answer in \\boxed{}. For example, if the problem is\n Solve for y: 2y + 5 = 15\n, then the final answer should include:\n The final answer: \\boxed{5}.'
            # system_prompt = "Please solve the math problem clearly and provide your final answer in \\boxed{}. Example of desired output: ##Problem: Solve for y: 2y + 5 = 15\n ##Solution:\n Subtract 5 from both sides: 2y = 10, Divide by 2: y = 5\n ##Answer: \\boxed{5}."
        else:
            # Correction mode: use prompt and model_output from errors.jsonl
            question = error['prompt']
            student_answer = error['model_output']
            ground_truth = error['ground_truth']

            # Create a correction prompt
            prompt_to_use = create_correction_prompt(question, student_answer)
            system_prompt = None

        # Call Gemini with streaming
        thinking, corrected_answer = call_gemini_streaming(prompt_to_use, system_prompt=system_prompt)

        if corrected_answer is None:
            return None

        # Extract the answer from the corrected output
        extracted_answer = extract_boxed_answer(corrected_answer)

        # Get index - handle both formats
        index = error.get('index', error.get('extra_info', {}).get('index', ''))

        # Check if answer is correct
        is_correct = extracted_answer == ground_truth if extracted_answer else False

        # Create corrected entry
        corrected_entry = {
            'index': index,
            'prompt': error.get('prompt', ''),
            'question': question,
            'original_answer': student_answer if not use_direct_prompt else '',
            'original_extracted': error.get('extracted_answer', ''),
            'ground_truth': ground_truth,
            'gemini_thinking': thinking,
            'gemini_corrected_answer': corrected_answer,
            'gemini_extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'data_source': error.get('data_source', ''),
            'ability': error.get('ability', ''),
            'prompt_mode': 'direct' if use_direct_prompt else 'correction'
        }

        # Write to file with thread safety
        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(corrected_entry, ensure_ascii=False) + '\n')
                f.flush()

            # If correct, also write to correct output file
            if is_correct and correct_output_file:
                # Create training entry with only necessary fields
                training_entry = {
                    'index': index,
                    'prompt': error.get('prompt', ''),
                    'question': question,
                    'gemini_thinking': thinking,
                    'gemini_corrected_answer': corrected_answer,
                    'gemini_extracted_answer': extracted_answer,
                    'ground_truth': ground_truth,
                    'data_source': error.get('data_source', ''),
                    'ability': error.get('ability', ''),
                    'prompt_mode': 'direct' if use_direct_prompt else 'correction'
                }

                with open(correct_output_file, 'a', encoding='utf-8') as f_correct:
                    f_correct.write(json.dumps(training_entry, ensure_ascii=False) + '\n')
                    f_correct.flush()

                # Update correct count tracker
                if correct_count_tracker is not None:
                    correct_count_tracker['count'] += 1

        return corrected_entry
    except Exception as e:
        print(f"\nError processing entry: {e}")
        return None

def filter_correct_answers(output_file, correct_output_file):
    """Filter and save only correct answers to a separate file for training.

    Args:
        output_file: Path to the full output JSONL file
        correct_output_file: Path to save only correct answers
    """
    print(f"\n" + "="*60)
    print("FILTERING CORRECT ANSWERS")
    print("="*60)

    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} does not exist!")
        return

    correct_count = 0
    total_count = 0

    with open(output_file, 'r', encoding='utf-8') as f_in, \
         open(correct_output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            try:
                entry = json.loads(line)
                total_count += 1

                # Verify that extracted answer matches ground truth
                gemini_answer = entry.get('gemini_extracted_answer', '')
                ground_truth = entry.get('ground_truth', '')

                if gemini_answer and gemini_answer == ground_truth:
                    # Create training entry with only necessary fields
                    training_entry = {
                        'index': entry['index'],
                        'prompt': entry['prompt'],
                        'question': entry['question'],
                        'gemini_thinking': entry['gemini_thinking'],
                        'gemini_corrected_answer': entry['gemini_corrected_answer'],
                        'gemini_extracted_answer': entry['gemini_extracted_answer'],
                        'ground_truth': entry['ground_truth'],
                        'data_source': entry.get('data_source', ''),
                        'ability': entry.get('ability', ''),
                        'prompt_mode': entry.get('prompt_mode', '')
                    }

                    f_out.write(json.dumps(training_entry, ensure_ascii=False) + '\n')
                    correct_count += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")

    print(f"\nFiltering Results:")
    print(f"Total entries processed: {total_count}")
    print(f"Correct answers (matched ground truth): {correct_count}")
    if total_count > 0:
        print(f"Accuracy: {correct_count}/{total_count} ({correct_count*100//total_count}%)")
    print(f"\nCorrect answers saved to: {correct_output_file}")
    print("="*60)

    return correct_count, total_count

def count_correct_in_file(filepath):
    """Count the number of correct entries in a file."""
    if not os.path.exists(filepath):
        return 0
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                count += 1
    except Exception as e:
        print(f"Warning: Error counting correct entries in {filepath}: {e}")
    return count

def process_errors_parallel(input_file, output_file, correct_output_file, max_workers=200, use_direct_prompt=False, target_correct=None):
    """Process all errors with parallel API calls.

    Args:
        input_file: Path to input JSONL file with errors
        output_file: Path to output JSONL file
        correct_output_file: Path to save only correct answers
        max_workers: Number of parallel workers (default: 200)
        use_direct_prompt: If True, use error['prompt'] directly;
                          If False, use create_correction_prompt (default)
        target_correct: Stop when this many correct answers are obtained (None = process all)
    """

    # Load already processed indices to support resume
    print(f"Checking for existing results in: {output_file}")
    processed_indices = load_processed_indices(output_file)

    # Count existing correct answers
    existing_correct = count_correct_in_file(correct_output_file)
    print(f"Existing correct answers: {existing_correct}")

    # Check if target already reached
    if target_correct and existing_correct >= target_correct:
        print(f"\n{'='*60}")
        print(f"TARGET ALREADY REACHED!")
        print(f"Already have {existing_correct} correct answers (target: {target_correct})")
        print(f"{'='*60}")
        return

    # Read all errors
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        errors = [json.loads(line) for line in f]
    print(f"Total entries in input file: {len(errors)}")

    # Filter out already processed errors
    # Handle both index formats: direct or nested in extra_info
    def get_entry_index(entry):
        """Extract index from entry, handling both formats."""
        if 'index' in entry and entry['index']:
            return entry['index']
        elif 'extra_info' in entry and 'index' in entry['extra_info']:
            return entry['extra_info']['index']
        return None

    errors_to_process = [e for e in errors if get_entry_index(e) not in processed_indices]

    print("\n" + "="*60)
    if processed_indices:
        print(f"RESUMING: {len(processed_indices)} already processed, {len(errors_to_process)} remaining")
        print(f"Progress: {len(processed_indices)}/{len(errors)} ({len(processed_indices)*100//len(errors)}%)")
    else:
        print(f"STARTING FRESH: Processing {len(errors_to_process)} entries")

    if len(errors_to_process) == 0:
        print("All entries have been processed! Nothing to do.")
        print("="*60)
        return

    print(f"Workers: {max_workers} parallel threads")
    print(f"Prompt mode: {'direct' if use_direct_prompt else 'correction'}")
    if target_correct:
        print(f"Target correct answers: {target_correct} (current: {existing_correct}, need: {target_correct - existing_correct} more)")
    print(f"Output file: {output_file}")
    print(f"Correct output file: {correct_output_file}")
    print("="*60 + "\n")

    # Create a shared counter for correct responses
    correct_count_tracker = {'count': existing_correct}

    # Track completion flag
    stop_processing = threading.Event()

    # Process errors in parallel using ThreadPoolExecutor
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_error, error, output_file, correct_output_file, use_direct_prompt, correct_count_tracker): error
                   for error in errors_to_process}

        # Process results as they complete with progress bar
        with tqdm(total=len(errors_to_process), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    success_count += 1

                    # Update progress bar with correct count
                    if target_correct:
                        pbar.set_postfix({'correct': f"{correct_count_tracker['count']}/{target_correct}"})

                    # Check if we've reached the target
                    if target_correct and correct_count_tracker['count'] >= target_correct:
                        print(f"\n\n{'='*60}")
                        print(f"TARGET REACHED!")
                        print(f"Obtained {correct_count_tracker['count']} correct answers (target: {target_correct})")
                        print(f"Stopping further processing...")
                        print(f"{'='*60}\n")

                        # Cancel remaining futures
                        for f in futures:
                            if not f.done():
                                f.cancel()

                        pbar.update(1)
                        break

                pbar.update(1)

    total_processed = len(processed_indices) + success_count
    final_correct = correct_count_tracker['count']

    print(f"\n" + "="*60)
    print(f"COMPLETED!")
    print(f"Successfully processed in this run: {success_count}/{len(errors_to_process)}")
    print(f"Total in output file: {total_processed}/{len(errors)} ({total_processed*100//len(errors) if len(errors) > 0 else 0}%)")
    print(f"Correct answers obtained: {final_correct}" + (f"/{target_correct}" if target_correct else ""))

    if target_correct and final_correct >= target_correct:
        print(f"\n✓ TARGET REACHED: {final_correct} correct answers!")
    elif target_correct:
        print(f"\n⚠ Still need {target_correct - final_correct} more correct answers to reach target")
        print(f"Run the script again to continue processing.")

    if total_processed < len(errors):
        remaining = len(errors) - total_processed
        print(f"\nRemaining unprocessed entries: {remaining}")
        if not target_correct or final_correct < target_correct:
            print(f"Run the script again to continue.")

    print(f"\nResults saved to: {output_file}")
    print(f"Correct answers saved to: {correct_output_file}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data with Gemini API in parallel")
    parser.add_argument("--input", default=None,
                        help="Input JSONL file (default: ../data/dapo_5k/data.jsonl for direct mode, ../data/dapo_5k_inference/errors.jsonl for correction mode)")
    parser.add_argument("--output", default=None,
                        help="Output JSONL file (default: gemini_direct.jsonl for direct mode, gemini_corrected.jsonl for correction mode)")
    parser.add_argument("--workers", type=int, default=200,
                        help="Number of parallel workers (default: 200)")
    parser.add_argument("--direct", action="store_true",
                        help="Use direct prompt mode (use original_prompt from data.jsonl instead of correcting errors)")
    parser.add_argument("--filter-only", action="store_true",
                        help="Only filter existing results (skip processing)")
    parser.add_argument("--correct-output", default=None,
                        help="Output file for correct answers only (default: adds '_correct' suffix to output file)")
    parser.add_argument("--target-correct", type=int, default=None,
                        help="Stop processing after obtaining this many correct answers (default: process all)")

    args = parser.parse_args()

    # Set default input file based on mode if not specified
    if args.input is None:
        if args.direct:
            args.input = "../data/dapo_5k/data.jsonl"
        else:
            args.input = "../data/dapo_5k_inference/errors.jsonl"

    # Set default output file based on mode if not specified
    if args.output is None:
        if args.direct:
            args.output = "../data/dapo_5k_inference/gemini_direct_prompt2.jsonl"
        else:
            args.output = "../data/dapo_5k_inference/gemini_corrected.jsonl"

    # Set default correct output file
    if args.correct_output is None:
        # Insert '_correct' before the file extension
        base, ext = os.path.splitext(args.output)
        args.correct_output = f"{base}_correct{ext}"

    # Process errors unless --filter-only is specified
    if not args.filter_only:
        process_errors_parallel(
            input_file=args.input,
            output_file=args.output,
            correct_output_file=args.correct_output,
            max_workers=args.workers,
            use_direct_prompt=args.direct,
            target_correct=args.target_correct
        )
    else:
        # Filter-only mode: re-filter existing results
        filter_correct_answers(args.output, args.correct_output)
