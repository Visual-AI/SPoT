from vllm import LLM, SamplingParams
import os
import re
import json
import jsonlines
import argparse
from tqdm import tqdm
import sys
import warnings

# import pdb
# import ray

warnings.filterwarnings("ignore")
import torch
from transformers import AutoTokenizer
# from math_verify import parse, verify
from collections import Counter


def extract_coordinates(coordinates_string):
    coordinates_list = re.findall(r"\((\d+),(\d+)\)", coordinates_string)
    # Convert to list of tuples of integers
    return [[int(x), int(y)] for x, y in coordinates_list]


def are_lists_equal(list1, list2):
    # 如果长度不同，直接返回 False
    if len(list1) != len(list2):
        return False

    # 将子列表转换为元组并计数
    counter1 = Counter(tuple(sublist) for sublist in list1)
    counter2 = Counter(tuple(sublist) for sublist in list2)

    # 比较计数器是否相同
    return counter1 == counter2


system_prompt_connect4 = '''
You are an expert player of the game Connect Four.

**Game Rules**
1. The game is played on a 6x7 grid by two players, X and O.
2. X typically plays first, then players alternate turns to drop their pieces.
3. The pieces can only be dropped at the lowest available space within the column.
4. The first player to connect four of their pieces in a row wins the game.
5. The connection can be horizontal, vertical, or diagonal.

**Input**
You will receive a state matrix representing the current game board:
* Empty space: _
* Player 1's piece: X
* Player 2's piece: O
The coordinates are zero-based indexing. For example, "(0,4):X" represents Player 1 has a piece on Row 0, Column 4. Row 0 is the lowest and Row 5 is the highest.

**Output**
Provide your chosen move. Your performance will be assessed on both the intermediate thinking results and the final decision. Follow the thinking process:

1. **Observations**
Based on the current game state, provide the following observations:
    * Where are your pieces located?
    * Where are your opponent's pieces located?
    * Check for all horizontal, vertical, or diagonal lines: are there any potential winning moves to form 4 in a row for you or your opponent? 
    Output all of the winning moves for you in the format "[Intermediate Thinking Results 1: (X,X), (X,X), ...]". If none, output "[Intermediate Thinking Results 1: None]". 
    Output all of the winning moves for your opponent in the format "[Intermediate Thinking Results 2: (X,X), (X,X), ...]". If none, output "[Intermediate Thinking Results 2: None]". 

2. **Strategic Analysis**
From your previous observations, if you have a winning move after checking, directly choose it. Otherwise if your opponent have a winning move, block it. If these are not the case, choose the best move based on the following strategy:
    * Look for opportunities to create multiple winning lines (for) simultaneously. If you have two discs in a row horizontally and two discs in a row diagonally, placing your next disc in the right position could lead to a win in multiple ways. For example, you have discs at [(0,1), (1,2), (2,2), (2,1)], then place your next disc at (2,3) would connect two lines: [(0,1), (1,2), (2,3)] and [(2,1), (2,2), (2,3)]
    * If your opponent has two consecutive discs in a row horizontally, block them from getting a third disc in that row. For example, if your opponent has discs at [(0,1), (0,2)], then place your next disc at (0,3) or (0,0) to block them.
    * Consider the center column as a strategic starting point. Placing your disc in the center column can give you more opportunities to create winning lines in different directions. Make the most of your opening moves by playing in the central columns.
    * Plan Ahead: Think one or two moves ahead. Try to anticipate where your opponent might be aiming to connect their discs and plan your strategy accordingly. For example, if your opponent has a winning move on (3,3), while (2,3) is not your winning move, you should not take (2,3) as your move, avoiding (3,3) to be a valid move for your opponent.
    * Try to get your 3 discs in a row with open spaces on either end.

3. **Conclusion**
In this section, based on your previous analysis, clearly state your decision for the position to place your next disc and give explanation.

4. **Chosen Move**
    * In this section, only output the chosen move. Do not include any other words.
    * The format is: "Chosen Move: (a,b)", where a is the row number (0-5), and b is the column number (0-6) where you want to place your disc.

'''


def eval_connect4(reasoning_trace, answer):
    response = reasoning_trace.lower()
    acc1, acc2 = 0, 0
    pattern_problem1 = re.compile(r"intermediate thinking results 1+: (.*?)]")
    match_problem1 = re.search(pattern_problem1, response)
    if match_problem1:
        intermediate_results1 = match_problem1.group(1).strip()
    else:
        intermediate_results1 = 'Format Error'

    # Problem 2
    pattern_problem2 = re.compile(r"intermediate thinking results 2+: (.*?)]")
    match_problem2 = re.search(pattern_problem2, response)
    if match_problem2:
        intermediate_results2 = match_problem2.group(1).strip()
    else:
        intermediate_results2 = 'Format Error'

    result1 = intermediate_results1.strip()
    result2 = intermediate_results2.strip()
    if result1 == 'Format Error':
        format_error = True
        print('------------------Format Error------------------')
        print(reasoning_trace)
    elif result1 == 'none':
        if answer[0] == []:
            acc1 += 1
    else:
        all_moves = extract_coordinates(result1)
        if are_lists_equal(all_moves, answer[0]):
            acc1 += 1
        else:
            print('------------------Wrong answer------------------')
            print('Expected:', answer[0])
            print('Got:', all_moves)
            print(reasoning_trace)
    if result2 == 'Format Error':
        format_error = True
    elif result2 == 'none':
        if answer[1] == []:
            acc2 += 1
    else:
        all_moves = extract_coordinates(result2)
        if are_lists_equal(all_moves, answer[1]):
            acc2 += 1

    return acc1, acc2


# print(f"Total examples: {total_examples}, Accuracy 1: {acc1 / total_examples}, Accuracy 2: {acc2 / total_examples}")
def get_prompts(dev_set, apply_chat_template):
    prompt2answer = {}
    processed_prompts = []
    with open(f"./{dev_set}.jsonl", 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            chat = [{"role": "system",
                     "content": system_prompt_connect4},
                    {"role": "user", "content": line['state']}]
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
            processed_prompts.append(prompt)
            prompt2answer[prompt] = line['answer']
    print(processed_prompts[-1])
    return processed_prompts, prompt2answer


def eval_ckpt(model_path):
    dev_set = 'eval_connect4'
    with open("./config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)

    num_gpus = torch.cuda.device_count()
    another_args = {'max_num_batched_tokens': 32768}
    apply_chat_template = AutoTokenizer.from_pretrained(model_path).apply_chat_template
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, **another_args, trust_remote_code=True,
              reasoning_parser="qwen3")

    processed_prompts, prompt2answer = get_prompts(dev_set, apply_chat_template)
    n, temperature = config[dev_set]['n'], config[dev_set]['temperature']
    sampling_params = SamplingParams(n=n, temperature=temperature, top_p=0.95,
                                     max_tokens=32768)
    outputs = llm.generate(processed_prompts, sampling_params, use_tqdm=True)
    eval_results = []
    eval_response = []
    for output in outputs:
        prompt = output.prompt
        responses = [output.outputs[i].text for i in range(n)]
        answer = prompt2answer[prompt]
        eval_result = [eval_connect4(response, answer) for response in responses]
        eval_results.extend(eval_result)
        eval_response.append({"question": prompt, "responses": responses, "results": eval_result, "answer": answer})

    acc1 = sum(result[0] for result in eval_results) / len(eval_results)
    acc2 = sum(result[1] for result in eval_results) / len(eval_results)
    eval_acc = acc1, acc2

    return eval_acc, eval_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, default='./connect4_output_qwen_base')
    parser.add_argument('--run_path', type=str, default= 'Qwen/Qwen3-8B')
    args = parser.parse_args()

    run_name = args.log_name
    model_path = args.run_path
    eval_acc, outputs = eval_ckpt(model_path)
    print("eval results:")
    print(eval_acc)
    print("eval done")

    print("example outputs:")
    print(outputs[0])
    print(outputs[1])
    print(outputs[2])
    
    from pathlib import Path

    output_path_str = "./connect4_output/eval_outputs/{}.json".format(run_name)
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    acc_path_str = "./connect4_output/results/{}.json".format(run_name)
    acc_path = Path(acc_path_str)
    acc_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    with open(acc_path, 'w', encoding='utf-8') as f:
        json.dump(eval_acc, f, ensure_ascii=False, indent=4)
