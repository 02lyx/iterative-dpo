import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from alignment import H4ArgumentParser
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
)

from trl.commands.cli_utils import TrlParser
from dpo import MyDPOTrainer


def prepare_data(
    data_dir: str = "/home/swb9572/IDPO/Online-RLHF/data/data_with_rewards.json",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    margin_scale=1,
    choose_type="random",
    eot_token="",
    length_penalty=0,
) -> Dataset:
    """Prepare the dataset for DPO training by rejection sampling.
    We implement different strategies to select pairs, including
    max_min: best v.s. worst
    max_random: best v.s. random from the remaining;
    max_max: best v.s. second best
    max_min_p: best v.s. worst but we additionally add a length penalty in the reward value
    """
    ds = load_dataset("json", data_files=data_dir, split="train")
    # ds = load_dataset("json", data_files='/home/swb9572/IDPO/Online-RLHF/data/data_with_rewards.json', split="train")
    print(ds)

    pos = []
    neg = []
    prompts = []

    margin = []
    for sample in ds:
        # P = tokenizer.apply_chat_template(sample["prompt"], tokenize = False, add_generation_prompt= True)
        P = tokenizer.apply_chat_template([{"role": "user", "content": sample["prompt"]}], tokenize = False, add_generation_prompt= True)
        print(P)
        if choose_type == "random":
            idx0 = 0
            idx1 = 1
        elif choose_type == "max_random":
            idx0 = np.argmax(sample["rewards"])
            if idx0 == 0:
                idx1 = 1
            else:
                idx1 = 0
        elif choose_type == "max_min":
            idx0 = np.argmax(sample["rewards"])
            idx1 = np.argmin(sample["rewards"])
        elif choose_type == "max_max":
            sorted_indices = np.argsort(sample["rewards"])
            idx0 = sorted_indices[-1]
            idx1 = sorted_indices[-2]
        elif choose_type == "max_min_p":
            r = [
                sample["rewards"][i] - length_penalty * len(sample["responses"][i])
                for i in range(len(sample["rewards"]))
            ]
            idx0 = np.argmax(r)
            idx1 = np.argmin(r)
        else:
            raise NotImplementedError

        if type(idx0) == np.ndarray or type(idx0) == list:
            assert len(idx0) == len(idx1)
            for i in range(len(idx0)):
                prompts.append(P)
                pos.append(sample["responses"][idx0[i]] + eot_token)
                neg.append(sample["responses"][idx1[i]] + eot_token)
                margin.append((sample["rewards"][idx0[i]] - sample["rewards"][idx1[i]]) * margin_scale)
        else:
            if sample["rewards"][idx0] > sample["rewards"][idx1]:
                prompts.append(P)
                pos.append(sample["responses"][idx0] + eot_token)
                neg.append(sample["responses"][idx1] + eot_token)
                margin.append((sample["rewards"][idx0] - sample["rewards"][idx1]) * margin_scale)
            elif sample["rewards"][idx0] < sample["rewards"][idx1]:
                prompts.append(P)
                pos.append(sample["responses"][idx1] + eot_token)
                neg.append(sample["responses"][idx0] + eot_token)
                margin.append((-sample["rewards"][idx0] + sample["rewards"][idx1]) * margin_scale)
        print(f"pos is {pos[-1]}")
        print(f"neg is {neg[-1]}")
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin})

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


### Main

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    attn_implementation="flash_attention_2",
    # attn_implementation="sdpa",
    torch_dtype=torch.float16,
)
model.config.use_cache = False

# if script_args.ignore_bias_buffers:
#     # torch distributed hack
#     model._ddp_params_and_buffers_to_ignore = [
#         name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
#     ]


ref_name = "google/gemma-2-2b-it"

model_ref = AutoModelForCausalLM.from_pretrained(
    ref_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

tokenizer.pad_token = tokenizer.eos_token

train_dataset = prepare_data(
    data_dir='./data/data_with_rewards.json',
    margin_scale=1.0,
    sanity_check=False,
    choose_type='max_min',
    eot_token="",
    length_penalty=0,
)

train_dataset['prompt']
train_dataset['response']