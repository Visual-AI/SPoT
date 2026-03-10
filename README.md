<div align="center">

# Surgical Post-Training: Cutting Errors, Keeping Knowledge


[![arXiv](https://img.shields.io/badge/arXiv-2603.01683-b31b1b.svg)](https://arxiv.org/abs/2603.01683)
[![Model](https://img.shields.io/badge/🤗%20Model-Qwen3--8B--SPoT-blue)](https://huggingface.co/linius/Qwen3-8B-SPoT)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-connect4-blue)](https://huggingface.co/datasets/linius/connect4)

*SPoT successfully introduces new knowledge via an Oracle to boost LLM reasoning, while preventing catastrophic forgetting through a reward-based binary optimization objective. This work makes a deep investigation into the limitations of SFT and DPO.*

</div>


## News

- **[2026-03-05]** Our SPoT-tuned Qwen3-8B checkpoint is live on HuggingFace — try it yourself! [linius/Qwen3-8B-SPoT](https://huggingface.co/linius/Qwen3-8B-SPoT)
- **[2026-03-04]** The Connect4 OOD reasoning evaluation dataset is now publicly available: [linius/connect4](https://huggingface.co/datasets/linius/connect4).


## Main Results

SPoT consistently outperforms all baselines across in-domain reasoning, OOD reasoning, and general instruction following. Crucially, while every competing method trades off at least one capability, **SPoT is the only approach that improves all three simultaneously**.

### Qwen3-8B

| Method | In-domain Avg | OOD Avg | IFEval | Overall Avg |
|---|---|---|---|---|
| Qwen3-8B (base) | 46.8 | 29.9 | 83.0 | 47.1 |
| + SFT | 41.0 (-5.8) | 25.5 (-4.4) | 79.6 (-3.4) | 41.8 (-5.3) |
| + RFT | 47.3 (+0.5) | 26.1 (-3.8) | 81.5 (-1.5) | 46.4 (-0.7) |
| + SFT+ | 50.5 (+3.7) | 30.7 (+0.8) | 80.0 (-3.0) | 49.4 (+2.3) |
| **+ SPoT (ours)** | **52.1 (+5.3)** | **41.4 (+11.5)** | **84.8 (+1.8)** | **53.3 (+6.2)** |

### Llama3.1-8B-Instruct

| Method | In-domain Avg | OOD Avg | IFEval | Overall Avg |
|---|---|---|---|---|
| Llama3.1-8B-Instruct (base) | 18.6 | 16.8 | 73.6 | 24.3 |
| + SFT | 18.0 (-0.6) | 15.7 (-1.1) | 62.1 (-11.5) | 22.4 (-1.9) |
| + RFT | 18.0 (-0.6) | 17.2 (+0.4) | 71.2 (-2.4) | 23.7 (-0.6) |
| + SFT+ | 19.9 (+1.3) | 16.7 (-0.1) | 68.6 (-5.0) | 24.6 (+0.3) |
| **+ SPoT (ours)** | **20.7 (+2.1)** | **18.5 (+1.7)** | 73.2 (-0.4) | **26.0 (+1.7)** |

Benchmarks: AIME24/25, AMC23, MATH-500, Minerva, OlympiadBench (in-domain); GPQA-Diamond, Connect4 (OOD); IFEval (instruction following). The OOD gain on Connect4 alone is **+25.1 points** for Qwen3-8B (10.9 → 36.0).


## Data Pipeline

The pipeline generates contrastive pairs `(x, y⁻, y⁺)` where `y⁺` is a minimally-edited correction of the model's wrong response `y⁻`.

```
Raw Dataset  →  Error Elicitation  →  Oracle Rectification  →  Contrastive Pairs
  (DAPO)          (Model Inference)      (Gemini 2.5 Pro)        (x, y⁻, y⁺)
```

### Step 1: Error Elicitation

Run inference on the DAPO-Math dataset to collect model failures:

```bash
bash scripts/run_dapo_inference.sh
```

Or with custom options:

```bash
python scripts/parallel_inference_dapo.py \
    --model Qwen/Qwen3-8B \
    --data_path data/dapo_5k/data.jsonl \
    --output_dir output/dapo_5k_inference \
    --num_gpus 8 \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_tokens 32768
```

Outputs `errors.jsonl` (incorrect predictions) and `all_results.jsonl`.

### Step 2: Oracle Rectification

Use Gemini 2.5 Pro to surgically correct the errors (supports resuming):

```bash
# Correction mode: correct student errors while preserving style
python scripts/correct_errors_parallel.py \
    --input output/dapo_5k_inference/errors.jsonl \
    --output data/gemini_corrected.jsonl \
    --workers 200
```



## Evaluation

### Supported Benchmarks

| Task string | Benchmark | Type |
|---|---|---|
| `custom\|aime24\|0\|0` | AIME 2024 | Math |
| `custom\|aime25\|0\|0` | AIME 2025 | Math |
| `custom\|amc23\|0\|0` | AMC 2023 | Math |
| `custom\|math_500\|0\|0` | MATH-500 | Math |
| `custom\|minerva\|0\|0` | Minerva Math | Math |
| `custom\|olympiadbench\|0\|0` | OlympiadBench | Math |
| `custom\|gpqa:diamond\|0\|0` | GPQA-Diamond | Science |
| `custom\|ifeval_no_thinking\|0\|0` | IFEval | Instruction Following |
| `connect4` | Connect4 (OOD) | Game Reasoning |

All benchmarks except Connect4 use `sober_eval/main.py`. Connect4 uses `eval/eval_connect4.py`.

> **Connect4** serves as an OOD reasoning benchmark with verifiable intermediate steps. Game states are dynamically generated via [GAMEBoT](https://github.com/Visual-AI/GAMEBoT) to prevent data contamination.

### Multi-Trial Evaluation

Loads the model once and duplicates the dataset N times — more efficient than re-running per seed:

```bash
bash run_evaluation_multi_trial_duplicated_data.sh /path/to/model --disable-thinking
```

Results are saved to `evaluation_results/aggregated_results.json`.


## Citation

If you find this work useful, please cite:

```bibtex
@article{lin2026surgical,
      title={Surgical Post-Training: Cutting Errors, Keeping Knowledge},
      author={Wenye Lin and Kai Han},
      year={2026},
      journal={arXiv preprint arXiv:2603.01683}
}
```
