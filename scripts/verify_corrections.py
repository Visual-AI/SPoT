import json
import os

def verify_corrections(corrected_file, errors_file):
    """Verify and analyze the corrected results."""

    # Load corrected results
    corrected = []
    if os.path.exists(corrected_file):
        with open(corrected_file, 'r', encoding='utf-8') as f:
            for line in f:
                corrected.append(json.loads(line))

    # Load original errors
    with open(errors_file, 'r', encoding='utf-8') as f:
        errors = [json.loads(line) for line in f]

    total_errors = len(errors)
    total_corrected = len(corrected)

    print("=" * 80)
    print("GEMINI CORRECTION PROGRESS")
    print("=" * 80)
    print(f"Total errors to correct: {total_errors}")
    print(f"Total corrected so far: {total_corrected}")
    print(f"Progress: {total_corrected}/{total_errors} ({100*total_corrected/total_errors:.2f}%)")
    print(f"Remaining: {total_errors - total_corrected}")

    if corrected:
        # Analyze corrections
        correct_count = sum(1 for c in corrected if c.get('is_correct', False))
        has_thinking = sum(1 for c in corrected if c.get('gemini_thinking', '').strip())
        has_answer = sum(1 for c in corrected if c.get('gemini_corrected_answer', '').strip())
        extracted = sum(1 for c in corrected if c.get('gemini_extracted_answer'))

        print("\n" + "=" * 80)
        print("CORRECTION QUALITY ANALYSIS")
        print("=" * 80)
        print(f"Corrections matching ground truth: {correct_count}/{total_corrected} ({100*correct_count/total_corrected:.2f}%)")
        print(f"Results with thinking content: {has_thinking}/{total_corrected} ({100*has_thinking/total_corrected:.2f}%)")
        print(f"Results with corrected answer: {has_answer}/{total_corrected} ({100*has_answer/total_corrected:.2f}%)")
        print(f"Results with extracted answer: {extracted}/{total_corrected} ({100*extracted/total_corrected:.2f}%)")

        # Show a few examples
        print("\n" + "=" * 80)
        print("SAMPLE CORRECTIONS (First 3)")
        print("=" * 80)
        for i, c in enumerate(corrected[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Index: {c['index']}")
            print(f"Ground Truth: {c['ground_truth']}")
            print(f"Original Answer: {c['original_extracted']}")
            print(f"Gemini Answer: {c.get('gemini_extracted_answer', 'N/A')}")
            print(f"Is Correct: {c.get('is_correct', False)}")
            print(f"Question: {c['question'][:100]}...")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    corrected_file = "data/dapo_5k_inference/gemini_corrected.jsonl"
    errors_file = "data/dapo_5k_inference/errors.jsonl"
    verify_corrections(corrected_file, errors_file)
