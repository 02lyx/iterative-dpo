#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json, os
from parser import extract_answer, strip_string

os.environ["HF_TOKEN"] = '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'

ds = load_dataset("RLHF4MATH/prompt_iter1")['train']

# instruct_prompt = r"Answer the question based on the following example:"
# example1 = r"""Question: Jack is stranded on a desert island. He wants some salt to season his fish. He collects 2 liters of seawater in an old bucket. If the water is 20% salt, how many ml of salt will Jack get when all the water evaporates? Answer: First find how many liters of the seawater are salt: 2 liters * 20% = 0.4 liters Then multiply that amount by 1000 ml/liter to find the number of ml of salt Jack gets: 0.4 liters * 1000 ml/liter = 400 ml."""
# example2 = r"""Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name."""
# few_shot_cot_prompt = instruct_prompt + '\n' + example2 + f'\nQuestion: '  #'\n' + example1

## for the dataset: RLHFlow/test_generation_2k
# def tokenize(sample):
#     if sample['']
#     answer_text = sample['solution'].split('####')[-1].strip()
#     sample["few_shot_cot_question"] = few_shot_cot_prompt + sample['question']
#     sample["answer_text"] = f"The answer is {answer_text}."
#     return sample

instruct_prompt = r"Please reason step by step, and put your final answer within \\boxed{{}}"
example = r"Question: What is the sum of the following infinite geometric series: $\frac{1}{3} - \frac{1}{9} + \frac{1}{27} - \frac{1}{81} + \frac{1}{243} - \frac{1}{729} + \frac{1}{2187} - \cdots$? Answer: To find the sum of an infinite geometric series, we need to determine if the series converges or diverges. In this case, we have a common ratio of $-\frac{1}{3}$. The series will converge if the absolute value of the common ratio is less than 1, and it will diverge otherwise. In this case, since $\left| -\frac{1}{3} \right| = \frac{1}{3} < 1$, the series converges. Next, we can use the formula for the sum of an infinite geometric series to find its value. The formula is given by: $$ S = \frac{a}{1-r} $$ where $S$ is the sum of the series, $a$ is the first term, and $r$ is the common ratio. In this case, the first term $a$ is $\frac{1}{3}$ and the common ratio $r$ is $-\frac{1}{3}$. Plugging these values into the formula, we have: $$ S = \frac{\frac{1}{3}}{1 - \left(-\frac{1}{3}\right)} $$ Simplifying the denominator, we get: $$ S = \frac{\frac{1}{3}}{\frac{4}{3}} $$ Dividing the numerator and denominator, we obtain: $$ S = \frac{1}{4} $$ Therefore, the sum of the given infinite geometric series is $\boxed{\frac{1}{4}}$. The answer is: \frac{1}{4}"
few_shot_cot_prompt = instruct_prompt + '\n' + example + f'\nQuestion: '  

def tokenize(sample):
    sample["prompt"] = few_shot_cot_prompt + sample['question']
    if sample['type'] in ['gsm8k']:
        answer_text = sample['solution'].split('####')[-1].strip()
    elif sample['type'] in ['gpt-3.5-turbo', 'math', 'MATH_Rephrased', 'MATH_FOBAR', 'MATH_SV', 'GSM_Rephrased', 'GSM_SV', 'GSM_FOBAR']:
        answer_text = extract_answer(sample['solution'])
    else:
        answer_text = ""
        print("error: unknown type")
    sample["answer"] =  strip_string(answer_text)
    return sample
# ds = ds.map(
#     lambda x: {
#         "prompt": tokenizer.apply_chat_template(x[script_args.dataset_key], tokenize=False, add_generation_prompt=True)
#     }
# )

ds = ds.map(tokenize, num_proc=16)
print(ds)
repo_id = "Yuanxin-Liu/Check"  
ds.push_to_hub(repo_id)