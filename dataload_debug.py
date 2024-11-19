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
import json, os, re
from parser import extract_answer
import tqdm

# Token name: For repo Iterative-DPO
os.environ["HF_TOKEN"] = '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'

model_path = "ZhangShenao/baseline-gemma-2-2b-it-sft"
dataset_name_or_path = "RLHF4MATH/prompt_iter1"
print("model_path", model_path)
seed = 42
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=2048,
    load_format="auto",
    seed=42,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=1,
    top_p=1.0,
    max_tokens=256,
    n=16,
    stop_token_ids=[tokenizer.eos_token_id],
    stop=['Question:'],
)

ds = load_dataset(dataset_name_or_path)['train']

# instruct_prompt = r"Answer the question based on the following example:"
# example1 = r"""Question: Jack is stranded on a desert island. He wants some salt to season his fish. He collects 2 liters of seawater in an old bucket. If the water is 20% salt, how many ml of salt will Jack get when all the water evaporates? Answer: First find how many liters of the seawater are salt: 2 liters * 20% = 0.4 liters Then multiply that amount by 1000 ml/liter to find the number of ml of salt Jack gets: 0.4 liters * 1000 ml/liter = 400 ml."""
# example2 = r"""Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name."""
# few_shot_cot_prompt = instruct_prompt + '\n' + example2 + f'\nQuestion: '  #'\n' + example1



# for the dataset: RLHF4MATH/prompt_iter1
def tokenize(sample):
    if sample['type'] in ['gsm8k', 'GSM_Rephrased', 'GSM_SV', 'GSM_FOBAR']:
        answer_text = sample['solution'].split('####')[-1].strip()
    elif sample['type'] in ['gpt-3.5-turbo', 'math', 'MATH_Rephrased', 'MATH_FOBAR', 'MATH_SV']:
        answer_text = extract_answer(sample['solution'])
    else:
        answer_text = ""
        print("error: unknown type")
    sample["answer"] = answer_text
    return sample

ds = ds.map(tokenize, num_proc=16)

data_size = len(ds)

print(ds[0])
print(ds[3])

# prompts = [ds[i]["few_shot_cot_question"] for i in range(len(ds))]
# outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)