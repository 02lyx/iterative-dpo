import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator
import re
import string
from datasets import Dataset
tqdm.pandas()
from grader import math_equal
from parser import extract_answer


# def math_equal(prediction: Union[bool, float, str],
#                 reference: Union[float, str],
#                 include_percentage: bool = True,
#                 is_close: bool = True,
#                 timeout: bool = False,
#                 ) -> bool:


### compare with true label and store the index of the correct/incorrect response
### then after rm assign a score, we let the correct response +100, and the incorrect response -100
# def exact_match_hf_evaluate(
#     predictions,
#     references,
#     regexes_to_ignore=None,
#     ignore_case=False,
#     ignore_punctuation=False,
#     ignore_numbers=False,
# ):
#     if regexes_to_ignore is not None:
#         for s in regexes_to_ignore:
#             predictions = np.array([re.sub(s, "", x) for x in predictions])
#             references = np.array([re.sub(s, "", x) for x in references])
#     else:
#         predictions = np.asarray(predictions)
#         references = np.asarray(references)

#     if ignore_case:
#         predictions = np.char.lower(predictions)
#         references = np.char.lower(references)

#     if ignore_punctuation:
#         repl_table = string.punctuation.maketrans("", "", string.punctuation)
#         predictions = np.char.translate(predictions, table=repl_table)
#         references = np.char.translate(references, table=repl_table)

#     if ignore_numbers:
#         repl_table = string.digits.maketrans("", "", string.digits)
#         predictions = np.char.translate(predictions, table=repl_table)
#         references = np.char.translate(references, table=repl_table)

#     score_list = predictions == references

#     return {"exact_match": list(score_list)}

    
# class RegexFilter():
#     def __init__(
#         self,
#         regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
#         group_select=0,
#         fallback: str = "[invalid]",
#     ) -> None:
#         """
#         pass a string `regex` to run `re.compile(r"regex")` on.
#         `fallback` defines the output returned if no matches for the regex are located.
#         """
#         self.regex_pattern = regex_pattern
#         self.regex = re.compile(regex_pattern)
#         self.group_select = group_select
#         self.fallback = fallback

#     def apply(self, resps):
#         # here, we assume we have a list, in which each element is
#         # a list of model responses for some particular input/target pair.
#         # so we process each of these (same input/target response sets)
#         # independently (and keep them a list.)
#         filtered = []
#         for resp in resps:
#             match = self.regex.findall(resp)
#             if match:
#                 match = match[self.group_select]
#                 if isinstance(match, tuple):
#                     match = [m for m in match if m][0]
#                 match = match.strip()
#             else:
#                 match = self.fallback
#             filtered.append(match)
#         return filtered



# ## Test single data

# test_sample = {"prompt": "Answer the question based on the following example:\nQuestion: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name.\nQuestion: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "responses": [" Answer: There is no information given about how many clips Natalia sold in May.  \n", "  \nAnswer: We can solve this problem by breaking it down into steps: \n* **Step 1:** Figure out how many clips Natalia sold in May. She sold half as many as in April, so she sold 48 / 2 = 24 clips in May.\n* **Step 2:** Calculate the total number of clips sold in April and May.  48 + 24 = 72 clips were sold altogether.** \n\n\n", " Answer: Here we know that Natalia sold a total of 48 clips in April. In May she sold half as much, so 48 / 2 = 24 clips. Altogether, she sold 48 + 24 = 72 clips. \n", " Answer:  While we don't have enough information to calculate how many clips Natalia sold in May specifically, we know that: * April sales equaled 48 clips. * May sales were half April sales, meaning May sales were 48 / 2 = 24 clips. * In total, she sold 48 + 24 = 72 clips.\n", "\nAnswer:  Here's how to solve the problem:\n\n* **May's Sales:** Natalia sold half as many clips in May as she did in April, so she sold 48 / 2 = 24 clips.\n* **Total Sales:**  To find the total number of clips sold, add the April sales and May sales: 48 + 24 = 72 clips.\n\n**Answer:** Natalia sold a total of 72 clips. \n\n\n", " Answer:  Here we need to solve for the number of clips sold in May. First, we know that Natalia sold half as many clips in May as April. That means she sold 48 * 0.5 = 24 clips in May. To find the total, we add the number of clips sold in April and May. So, the total clips sold was 48 + 24 = 72 clips.\n\n\n", " Answer: Here's how to solve the problem: * We need to figure out how many clips Natalia sold in May. Half of 48 is 48 / 2 = 24 clips. *  To find the total number of clips sold, we add the number sold in April and May: 48 + 24 = 72 clips.\n", " Answer:   We need to find out how many clips Natalia sold in May. Since she sold half as many in May as she did in April, she sold 48/2 = 24 clips in May.  To find the total number of clips sold in both months, we add them together: 48 + 24 = 72 clips sold altogether.  \n\n", " Answer: Here's how to solve the problem:  * Tapas sold in April: 48 clips *  **We know she sold half as many in May**, so she sold 48 / 2 = 24 clips in May. * **Total clips sold:** 48 clips (April) + 24 clips (May) ** = 72 clips altogether.  \n\n\n", "  Answer: ... \n", " \nAnswer: Here's how to solve the problem:  * **May sales:** Natalia sold half as many clips in May, so she sold 48 / 2 = 24 clips. * **Total sales:** She sold 48 clips in April and 24 clips in May for a total of 48 + 24 = 72 clips. \n\n\n", " Answer: We need to figure out how many clips Natalia sold in May. Since she sold half as many in May as April, she sold 48 / 2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips.\n", "\n\nHere's how to solve it:\n\n*   **Step 1:**  Figure out how many clips Natalia sold in May. Half of 48 is 48/2 = 24 clips.\n*   **Step 2:**  Add the clips sold in April and May to find the total. 48 + 24 = 72 clips. \n\n\n\n**", "  \nAnswer: Here's how to solve the problem:\n\n* **May's Sales:** Natalia sold half the number of clips in May as in April, so she sold 48 / 2 = 24 clips in May.\n* **Total Sales:**  To find the total number of clips sold, add the sales from April and May: 48 + 24 = 72 clips.\n* **Answer:** Natalia sold a total of 72 clips in April and May. \n\n\n**", "\nAnswer:  We need to figure out how many clips Natalia sold in May. Half of 48 is 48 / 2 = 24 clips. Then, we need to add up the clips from April and May: 48 + 24 = 72 clips. \n", " **Solution:**  This is a word problem that we can solve by breaking down the steps. We need to do:  *Step 1: Find the number of clips sold in May:* Half of 48 clips is 48/2 = 24 clips\n*Step 2: Find the total number of clips sold:* Total clips sold = April clips + May clips = 48 + 24 = 72 clips**Answer:** Natalia sold a total of 72 clips in April and May.   \n\n"], "answer": "72"}

# print(test_sample["responses"])

# extracted_answer = [extract_answer(resp.lstrip()) for resp in test_sample["responses"]]

# print([resp.lstrip() for resp in test_sample["responses"]])
# print(extracted_answer)
# print(test_sample['answer'])

# print(math_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}", timeout=True))
# result = [math_equal(extracted_answer[i], test_sample["answer"], include_percentage=True, is_close=True, timeout=True) for i in range(len(extracted_answer))]

# print(result)
## Test batch data

## data = [{"prompt": "prompt", "responses": ["response1", "response2", ...], "answer": "answer"}, ...]
## data every entry {"prompt": "prompt", "responses": ["response1", "response2", ...], "answer": "answer"}

# ds_dir = "/home/swb9572/iterative-dpo/test_data_with_rewards.json"
# ds = load_dataset("json", data_files=ds_dir, split="train")

# filtered_dataset = []

# with torch.no_grad():
#     for sample in tqdm(ds):
#         if len(sample["responses"]) < 16:
#             continue
#         preprocessed_resps = [resp.lstrip() for resp in sample["responses"]]
#         print(preprocessed_resps)
#         ex_answer = [extract_answer(resp) for resp in preprocessed_resps]
#         print(ex_answer)
#         print(sample["answer"])
#         result = [math_equal(ex_answer[i], sample["answer"], include_percentage=True, is_close=True, timeout=True) for i in range(len(preprocessed_resps))]
#         print(result)
#         if (all(result)) or (not any(result)):
#             pass
#         else:
#             new_rewards = [x + 100 if y else x - 100 for x, y in zip(sample["rewards"], result)]
#             dict = {"prompt": sample["prompt"], "responses": sample["responses"], "rewards": new_rewards, "answer": sample["answer"]}
#             filtered_dataset.append(dict)

# new_dataset = Dataset.from_list(filtered_dataset)
# new_dataset.to_json("filtered_dataset.json")
# print(new_dataset)

ds_dir = "/home/swb9572/iterative-dpo/test_data_with_rewards.json"
ds = load_dataset("json", data_files=ds_dir, split="train")

filtered_dataset = []

with torch.no_grad():
    for sample in tqdm(ds):
        if len(sample["responses"]) < 16:
            continue
        preprocessed_resps = [resp.lstrip() for resp in sample["responses"]]
        print(preprocessed_resps)
        ex_answer = [extract_answer(resp) for resp in preprocessed_resps]
        print(ex_answer)
        print(sample["answer"])
        result = [math_equal(ex_answer[i], sample["answer"], include_percentage=True, is_close=True, timeout=True) for i in range(len(preprocessed_resps))]
        print(result)
        if (all(result)) or (not any(result)):
            pass
        else:
            new_rewards = [x + 100 if y else x - 100 for x, y in zip(sample["rewards"], result)]
            dict = {"prompt": sample["prompt"], "responses": sample["responses"], "rewards": new_rewards, "answer": sample["answer"]}
            filtered_dataset.append(dict)

new_dataset = Dataset.from_list(filtered_dataset)
new_dataset.to_json("filtered_dataset.json")
print(new_dataset)
