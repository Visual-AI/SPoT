# SPoT: Surgical Post-Training

Code for the paper **"Surgical Post-Training: Cutting Errors, Keeping Knowledge"**

> SPoT successfully introduces new knowledge via an Oracle to boost LLM reasoning, while preventing catastrophic forgetting through a reward-based binary optimization objective. This work makes a deep investigation into the limitations of SFT and DPO.



## Data Pipeline

The pipeline generates contrastive pairs `(x, y⁻, y⁺)` where `y⁺` is a minimally-edited correction of the model's wrong response `y⁻`.

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

---

## Evaluation

### Supported Benchmarks

| Task string | Benchmark | Script |
|---|---|---|
| `custom\|aime24\|0\|0` | AIME 2024 | `sober_eval/main.py` |
| `custom\|aime25\|0\|0` | AIME 2025 | `sober_eval/main.py` |
| `custom\|amc23\|0\|0` | AMC 2023 | `sober_eval/main.py` |
| `custom\|math_500\|0\|0` | MATH-500 | `sober_eval/main.py` |
| `custom\|minerva\|0\|0` | Minerva Math | `sober_eval/main.py` |
| `custom\|olympiadbench\|0\|0` | OlympiadBench | `sober_eval/main.py` |
| `custom\|gpqa:diamond\|0\|0` | GPQA-Diamond | `sober_eval/main.py` |
| `custom\|ifeval_no_thinking\|0\|0` | IFEval | `sober_eval/main.py` |
| `connect4` | Connect4 (OOD, dynamically generated) | `eval/eval_connect4.py` |

Connect4 serves as an OOD reasoning benchmark with verifiable intermediate steps. Game states are dynamically generated via [GAMEBoT](https://github.com/Visual-AI/GAMEBoT) to prevent data contamination, requiring a separate evaluation script.

### Multi-Trial Evaluation

Loads the model once and duplicates the dataset N times — more efficient than re-running per seed:

```bash
bash run_evaluation_multi_trial_duplicated_data.sh /path/to/model --disable-thinking
```

Results are saved to `evaluation_results/aggregated_results.json`.

