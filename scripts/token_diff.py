"""
Token Difference Detection Algorithm

This module implements a text comparison function similar to diff tools or Beyond Compare.
Given two batches of token ID lists, it returns a mask indicating which positions differ.
"""

from typing import List, Tuple, Optional
import json
from difflib import SequenceMatcher


def compute_token_diff_mask(
    reference_ids: List[int],
    target_ids: List[int],
    algorithm: str = "simple"
) -> Tuple[List[bool], List[bool]]:
    """
    Compare two token ID sequences and return masks indicating differences.

    This function implements multiple comparison algorithms:
    - 'simple': Direct position-by-position comparison (no alignment)
    - 'lcs': Longest Common Subsequence based alignment (like diff tools)

    Args:
        reference_ids: Reference token ID list (original answer)
        target_ids: Target token ID list (corrected answer)
        algorithm: Comparison algorithm to use ('simple' or 'lcs')

    Returns:
        Tuple of (reference_mask, target_mask):
        - reference_mask: Boolean mask for reference, False = different/deleted
        - target_mask: Boolean mask for target, False = different/inserted

    Example:
        reference = [1, 2, 3, 4, 5]
        target = [1, 2, 6, 7, 5]
        ref_mask, tgt_mask = compute_token_diff_mask(reference, target, 'lcs')
        # ref_mask = [True, True, False, False, True]  # 3,4 changed
        # tgt_mask = [True, True, False, False, True]  # 6,7 are new
    """

    if algorithm == "simple":
        return _simple_diff(reference_ids, target_ids)
    elif algorithm == "lcs":
        return _lcs_diff(reference_ids, target_ids)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _simple_diff(
    reference_ids: List[int],
    target_ids: List[int]
) -> Tuple[List[bool], List[bool]]:
    """
    Simple position-by-position comparison.
    Masks are False for positions with differences.
    """
    max_len = max(len(reference_ids), len(target_ids))

    ref_mask = []
    tgt_mask = []

    for i in range(max_len):
        ref_val = reference_ids[i] if i < len(reference_ids) else None
        tgt_val = target_ids[i] if i < len(target_ids) else None

        if ref_val == tgt_val and ref_val is not None:
            ref_mask.append(True)
            tgt_mask.append(True)
        else:
            if i < len(reference_ids):
                ref_mask.append(False)
            if i < len(target_ids):
                tgt_mask.append(False)

    return ref_mask, tgt_mask


def _lcs_diff(
    reference_ids: List[int],
    target_ids: List[int]
) -> Tuple[List[bool], List[bool]]:
    """
    LCS-based diff algorithm (similar to Beyond Compare or Unix diff).
    Uses SequenceMatcher to find matching blocks and mark differences.
    """
    # Initialize masks - all False initially (all different)
    ref_mask = [False] * len(reference_ids)
    tgt_mask = [False] * len(target_ids)

    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, reference_ids, target_ids)

    # Mark matching regions as True
    for match in matcher.get_matching_blocks():
        ref_start, tgt_start, length = match

        # Mark the matching region in both masks
        for i in range(length):
            if ref_start + i < len(reference_ids):
                ref_mask[ref_start + i] = True
            if tgt_start + i < len(target_ids):
                tgt_mask[tgt_start + i] = True

    return ref_mask, tgt_mask


def get_diff_regions(
    reference_ids: List[int],
    target_ids: List[int],
    reference_mask: List[bool],
    target_mask: List[bool]
) -> List[dict]:
    """
    Extract regions with differences for detailed analysis.

    Returns:
        List of difference regions with details about what changed
    """
    diffs = []

    # Find continuous regions where mask is False
    i, j = 0, 0

    while i < len(reference_ids) or j < len(target_ids):
        # Find start of difference region
        while i < len(reference_ids) and reference_mask[i]:
            i += 1
        while j < len(target_ids) and target_mask[j]:
            j += 1

        if i >= len(reference_ids) and j >= len(target_ids):
            break

        # Found a difference region, find its end
        ref_start, tgt_start = i, j

        while i < len(reference_ids) and not reference_mask[i]:
            i += 1
        while j < len(target_ids) and not target_mask[j]:
            j += 1

        diffs.append({
            'ref_range': (ref_start, i),
            'tgt_range': (tgt_start, j),
            'ref_tokens': reference_ids[ref_start:i],
            'tgt_tokens': target_ids[tgt_start:j],
            'type': _classify_diff(ref_start, i, tgt_start, j)
        })

    return diffs


def _classify_diff(ref_start: int, ref_end: int, tgt_start: int, tgt_end: int) -> str:
    """Classify the type of difference."""
    ref_len = ref_end - ref_start
    tgt_len = tgt_end - tgt_start

    if ref_len == 0:
        return 'insertion'
    elif tgt_len == 0:
        return 'deletion'
    else:
        return 'modification'


def print_diff_summary(
    reference_ids: List[int],
    target_ids: List[int],
    reference_mask: List[bool],
    target_mask: List[bool],
    tokenizer=None
):
    """
    Print a human-readable summary of differences.

    Args:
        reference_ids: Reference token IDs
        target_ids: Target token IDs
        reference_mask: Reference mask (False = different)
        target_mask: Target mask (False = different)
        tokenizer: Optional tokenizer to decode token IDs to text
    """
    print("=" * 80)
    print("DIFF SUMMARY")
    print("=" * 80)

    # Overall statistics
    ref_unchanged = sum(reference_mask)
    ref_changed = len(reference_ids) - ref_unchanged
    tgt_unchanged = sum(target_mask)
    tgt_changed = len(target_ids) - tgt_unchanged

    print(f"\nReference: {len(reference_ids)} tokens ({ref_unchanged} unchanged, {ref_changed} changed)")
    print(f"Target:    {len(target_ids)} tokens ({tgt_unchanged} unchanged, {tgt_changed} changed)")

    # Get diff regions
    diffs = get_diff_regions(reference_ids, target_ids, reference_mask, target_mask)

    print(f"\nNumber of difference regions: {len(diffs)}")

    # Print each difference
    for idx, diff in enumerate(diffs, 1):
        print(f"\n--- Difference #{idx} ({diff['type']}) ---")
        print(f"Position: Reference[{diff['ref_range'][0]}:{diff['ref_range'][1]}] -> Target[{diff['tgt_range'][0]}:{diff['tgt_range'][1]}]")

        if tokenizer:
            ref_text = tokenizer.decode(diff['ref_tokens']) if diff['ref_tokens'] else "[DELETED]"
            tgt_text = tokenizer.decode(diff['tgt_tokens']) if diff['tgt_tokens'] else "[INSERTED]"
            print(f"Reference: '{ref_text}'")
            print(f"Target:    '{tgt_text}'")
        else:
            # Fallback to showing token IDs if no tokenizer
            print(f"Reference tokens: {diff['ref_tokens']}")
            print(f"Target tokens:    {diff['tgt_tokens']}")

    print("=" * 80)


if __name__ == "__main__":
    # Simple test example
    print("Running simple test example...")

    # Example: "The quick brown fox" -> "The fast brown dog"
    # Simulated as token IDs
    reference = [1, 2, 3, 4]  # The quick brown fox
    target = [1, 5, 3, 6]      # The fast brown dog

    print("\n### Test 1: Simple Algorithm ###")
    ref_mask, tgt_mask = compute_token_diff_mask(reference, target, algorithm='simple')
    print(f"Reference IDs: {reference}")
    print(f"Target IDs:    {target}")
    print(f"Reference mask: {ref_mask}")
    print(f"Target mask:    {tgt_mask}")

    print("\n### Test 2: LCS Algorithm ###")
    ref_mask, tgt_mask = compute_token_diff_mask(reference, target, algorithm='lcs')
    print(f"Reference IDs: {reference}")
    print(f"Target IDs:    {target}")
    print(f"Reference mask: {ref_mask}")
    print(f"Target mask:    {tgt_mask}")
    print_diff_summary(reference, target, ref_mask, tgt_mask)

    print("\n### Test 3: Insertion/Deletion ###")
    reference = [1, 2, 3, 4, 5]
    target = [1, 2, 6, 7, 8, 4, 5]  # Inserted 6,7,8 in middle

    ref_mask, tgt_mask = compute_token_diff_mask(reference, target, algorithm='lcs')
    print(f"Reference IDs: {reference}")
    print(f"Target IDs:    {target}")
    print(f"Reference mask: {ref_mask}")
    print(f"Target mask:    {tgt_mask}")
    print_diff_summary(reference, target, ref_mask, tgt_mask)
