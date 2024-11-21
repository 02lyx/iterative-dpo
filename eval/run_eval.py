"""
This script supports multi-rounds vllm inference.
The inference is formulated as a multi-turn chat and the model should be registered as a server by scripts/register_server.sh first.
"""

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from eval.evaluate import evaluate
from tqdm import tqdm
from utils.data_loader import load_data
from utils.parser import *
from utils.utils import construct_prompt, load_jsonl, save_jsonl, set_seed
from vllm import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tora", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--ports", action='append', default=[])
    parser.add_argument("--horizon", default=6, type=int)  # the maximal number of tool calls
    parser.add_argument("--eval", default=False, type=bool)

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(args):
    examples = load_data(args.data_name, args.split, args.data_dir)

    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)
    elif args.num_test_sample == -1:
        args.num_test_sample = len(examples)

    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    if args.end == -1:
        args.end = len(examples)
    examples = examples[args.start: args.end]

    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    out_file = f"{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{dt_string}.jsonl"
    os.makedirs(f"{args.output_dir}/{model_name}/{args.data_name}", exist_ok=True)

    processed_files = [
        f
        for f in os.listdir(f"{args.output_dir}/{model_name}/{args.data_name}/")
        if f.endswith(".jsonl") and f.startswith(out_file_prefix)
    ]
    processed_samples = []
    for f in processed_files:
        processed_samples.extend(list(load_jsonl(f"{args.output_dir}/{model_name}/{args.data_name}/{f}")))

    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    total_examples = len(examples)
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    print(f"Idx {args.start} - {args.end}: Remain {len(examples)}/{total_examples} samples.")
    return examples, processed_samples, out_file


def main(args):
    ports = args.ports
    examples, processed_samples, out_file = prepare_data(args)

    SamplingParams.seed = args.seed
    samples = []

    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]
        example["question"] = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        full_prompt = construct_prompt(args, example)

        sample = {"idx": idx, "question": example["question"], "gt_cot": gt_cot, "gt": gt_ans, "prompt": full_prompt}
        for key in [
            "level", "type", "unit", "solution_type", "choices", "solution",
            "ques_type", "ans_type", "answer_type", "dataset", "subfield", "filed", "theorem", "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    print("dataset:", args.data_name, "samples:", len(samples))

    remain_prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]
    all_gts = [sample["gt"] for sample in samples for _ in range(args.n_sampling)]
    tmp_idx = list(range(len(all_gts)))
    all_gts = dict(zip(tmp_idx, all_gts))

    end_prompts = remain_prompts
    codes = end_prompts  # No execution, use generated text directly
    results = [(code, None) for code in codes]

    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling: (i + 1) * args.n_sampling]
        preds = code
        reports = [None] * len(code)
        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    all_samples.extend(processed_samples)
    save_jsonl(all_samples, out_file)

    if args.eval:
        result_str = evaluate(samples=all_samples, data_name=args.data_name, prompt_type=args.prompt_type, execute=False)
        print(result_str)
        with open(out_file.replace(".jsonl", f"_{args.prompt_type}.metrics"), "w") as f:
            f.write(result_str)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
