# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""

import random
import re
import numpy as np
from aenum import extend_enum

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# Prompt template adapted from
# - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
# - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
# Note that it is important to have the final answer in a box for math-verify to work correctly
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$.' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

MATH_QUERY_QWEN3 = "Solve the following math problem. Please reason step by step, and put your final answer within \\boxed{}."

# Prompt template from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
# GPQA_QUERY_TEMPLATE = """
# Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

# {Question}

# A) {A}
# B) {B}
# C) {C}
# D) {D}
# """.strip()

GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. Please show your choice in the answer field with only the choice letter, e.g., "Answer": "C". Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# Prompt template for MMLU-Pro (supports up to 10 options)
MMLU_PRO_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the option letters. Think step by step before answering.

{Question}

{Options}
""".strip()

latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)

# Metric for MMLU-Pro (uses letter options A-J)
# Note: "Letters" prefix supports all letters A-Z, so it works for 10-option questions
mmlu_pro_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="Letters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="Letters")],
    precision=5,
)


def math_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def amc_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def minerva_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def olympiadbench_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def gpqa_prompt_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )


def mmlu_pro_prompt_fn(line, task_name: str = None):
    # MMLU-Pro has variable number of options (up to 10)
    options = line["options"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    # Format options as "A) option1\nB) option2\n..."
    formatted_options = "\n".join([f"{letters[i]}) {opt}" for i, opt in enumerate(options)])

    query = MMLU_PRO_QUERY_TEMPLATE.format(
        Question=line["question"],
        Options=formatted_options
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=letters[:len(options)],
        gold_index=line["answer_index"],
        instruction=query,
    )


# ============================================================================
# IFEval with thinking token removal
# ============================================================================

def remove_thinking_tokens(text: str) -> str:
    """
    Remove thinking/reasoning tokens from model output.

    Handles various formats:
    - <think>...</think>
    - <thinking>...</thinking>
    - <reason>...</reason>
    - <reflection>...</reflection>
    - Other common thinking tag patterns

    Returns the text with thinking tokens removed.
    """
    # Common thinking tag patterns
    patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<reason>.*?</reason>',
        r'<reasoning>.*?</reasoning>',
        r'<reflection>.*?</reflection>',
        r'<thought>.*?</thought>',
        r'<thoughts>.*?</thoughts>',
        # DeepSeek-R1 style
        r'<｜begin▁of▁sentence｜>.*?<｜end▁of▁sentence｜>',
    ]

    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)

    # Clean up extra whitespace
    cleaned_text = re.sub(r'\n\n+', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def ifeval_prompt(line, task_name: str = None):
    """IFEval prompt function."""
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instructions_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )


def ifeval_metric_with_thinking_removal(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    """
    IFEval metric with thinking token removal.

    This removes thinking/reasoning tokens before evaluating instruction following.
    """
    # Import here to avoid circular dependencies
    import lighteval.tasks.extended.ifeval.instructions_registry as instructions_registry

    # Remove thinking tokens from the response
    raw_response = predictions[0]
    response = remove_thinking_tokens(raw_response)

    # Strict instructions
    instruction_list = formatted_doc.specific["instructions_id_list"]
    all_kwargs = formatted_doc.specific["kwargs"]
    prompt = formatted_doc.query

    # Loose instructions (same preprocessing as original IFEval)
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    is_following_list_strict = []
    is_following_list_loose = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs
        task_kwargs = {k: v for k, v in all_kwargs[index].items() if v}
        instruction.build_description(**task_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        # Strict
        if response.strip() and instruction.check_following(response):
            is_following_list_strict.append(True)
        else:
            is_following_list_strict.append(False)

        # Loose
        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list_loose.append(is_following)

    return {
        "prompt_level_strict_acc": int(all(is_following_list_strict)),
        "inst_level_strict_acc": is_following_list_strict,
        "prompt_level_loose_acc": int(all(is_following_list_loose)),
        "inst_level_loose_acc": is_following_list_loose,
    }


def agg_inst_level_acc(items):
    """Aggregate instruction-level accuracy."""
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc


# Metric names
ifeval_submetric_names = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]

# Create the custom metric
ifeval_metrics_no_thinking = SampleLevelMetricGrouping(
    metric_name=ifeval_submetric_names,
    higher_is_better=dict.fromkeys(ifeval_submetric_names, True),
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=ifeval_metric_with_thinking_removal,
    corpus_level_fn={
        "prompt_level_strict_acc": np.mean,
        "inst_level_strict_acc": agg_inst_level_acc,
        "prompt_level_loose_acc": np.mean,
        "inst_level_loose_acc": agg_inst_level_acc,
    },
)


# Define tasks
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=math_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
gpqa_diamond = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["custom"],
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metric=[gpqa_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=1,
)
minerva = LightevalTaskConfig(
    name="minerva",
    suite=["custom"],
    prompt_function=minerva_prompt_fn,
    hf_repo="knoveleng/Minerva-Math",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
amc23 = LightevalTaskConfig(
    name="amc23",
    suite=["custom"],
    prompt_function=amc_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
olympiadbench = LightevalTaskConfig(
    name="olympiadbench",
    suite=["custom"],
    prompt_function=olympiadbench_prompt_fn,
    hf_repo="knoveleng/OlympiadBench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
mmlu_pro = LightevalTaskConfig(
    name="mmlu_pro",
    suite=["custom"],
    prompt_function=mmlu_pro_prompt_fn,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="random_sampling",  # Enable few-shot example selection
    generation_size=32768,
    metric=[mmlu_pro_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=1,
)

# IFEval with thinking token removal
ifeval_no_thinking = LightevalTaskConfig(
    name="ifeval_no_thinking",
    suite=["custom"],
    prompt_function=ifeval_prompt,
    hf_repo="google/IFEval",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metric=[ifeval_metrics_no_thinking],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=1,
)

# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime25)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(gpqa_diamond)
TASKS_TABLE.append(minerva)
TASKS_TABLE.append(amc23)
TASKS_TABLE.append(olympiadbench)
TASKS_TABLE.append(mmlu_pro)
TASKS_TABLE.append(ifeval_no_thinking)

# Register the custom metric (only if not already registered)
if not hasattr(Metrics, "ifeval_metric_no_thinking"):
    extend_enum(Metrics, "ifeval_metric_no_thinking", ifeval_metrics_no_thinking)

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
