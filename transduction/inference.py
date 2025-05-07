from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
import re
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
import json
import pickle
from small_tokenizer import shrink_embeddings


from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import Dict, List, Tuple


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 1) tokenizer and base model share the same shrunk vocab
tokenizer = AutoTokenizer.from_pretrained(
    "/scratch/yb2510/ARC/transduction/shrunk_model_dir",  trust_remote_code=True, use_fast=False
)
model = AutoModelForCausalLM.from_pretrained(
    "/scratch/yb2510/ARC/transduction/shrunk_model_dir",
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# 2) overlay the LoRA weights
model = PeftModel.from_pretrained(
    model,
    "/scratch/yb2510/ARC/transduction/saved_model_augmeted/checkpoint-6678",
    is_train=False,         # we’re in inference mode
)

# 3) prep for generation
model.eval()
model.to(DEVICE)

train_data_path = "/scratch/yb2510/ARC/transduction/data/training/"
list1 = os.listdir(train_data_path)
complete_data = []

for x in list1:
    data = json.load(open(train_data_path+x))
    complete_data.append(data)

color_map = {
    0: "Black",  1: "Blue",    2: "Red",    3: "Green", 4: "Yellow",
    5: "Grey",   6: "Fuchsia", 7: "Orange", 8: "Teal",  9: "Brown"
}

def grid_to_color_str(grid: List[List[int]]) -> str:
    """
    Convert a 2D list of ints into a string with color names,
    rows separated by newlines, cells by spaces.
    """
    lines = []
    for row in grid:
        lines.append(" ".join(color_map[val] for val in row))
    return "\n".join(lines)

def make_prompt_and_answer(task: Dict) -> Tuple[str, str]:
    """
    Given a dict with 'train' (list of {'input', 'output'}) and a single
    'test' example, build the full prompt+answer strings.
    """
    # 1) System role
    system_msg = """----- Role: system --------------------
You are a world-class puzzle solver with exceptional pattern recognition
skills. Your task is to analyze puzzles, spot patterns, and provide direct
solutions. You are kind of a local solver - your strength is in finding local patterns and local subproblems to solve the puzzels.
"""

    # 2) User role header
    user_msg = """----- Role: user --------------------
Given input-output grid pairs as reference examples, carefully observe the
patterns to predict the output grid for new test input. Each pair follows
the same transformation rule. Grids are 2D arrays represented as strings,
with cells (colors) separated by spaces and rows by newlines.
There must be local subproblems in input, which might have one-to-one relation in output.
Solve these local problems one by one and then predict the output grid of test input.
Here are the input and output grids for the reference examples:
"""

    # 3) Append each train example
    for i, ex in enumerate(task["train"], start=1):
        inp_str  = grid_to_color_str(ex["input"])
        out_str  = grid_to_color_str(ex["output"])
        user_msg += f"\nExample {i}\nInput:\n{inp_str}\nOutput:\n{out_str}\n"

    # 4) Add the test prompt
    test_in = task["test"][0]["input"]
    test_in_str = grid_to_color_str(test_in)
    user_msg += f"""Here is the input grid for the test example:
Input:
{test_in_str}
Directly provide the output grid(s) corresponding to the given test input
grids, based on the patterns observed in the reference examples.
"""

    prompt = system_msg + "\n" + user_msg

    # 5) Build the "assistant" answer by rendering the expected test output
    test_out = task["test"][0]["output"]
    answer = "----- Role: assistant --------------------\nThe output grid for the test input grid is:\n"
    answer += "‘‘‘\n" + grid_to_color_str(test_out) + "\n‘‘‘"

    return prompt, answer


raw_examples=complete_data


for i, x in enumerate(raw_examples):
    if i>5:
        break
    prompt, answer = make_prompt_and_answer(raw_examples[i])

    print(prompt)
    print("Correct Answer:", answer)

# prompt = (
#     f"2+2=?"
# )

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# out    = model.generate(**inputs, max_new_tokens=7)


# 1) generate as before
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# gen_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False, num_beams=1)
# raw = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

# # now you can do:
# input_ids = tokenizer("Your prompt here", return_tensors="pt").to(DEVICE)
    generated = model.generate(
        **inputs,
        do_sample=True,           # or unset temperature / set do_sample=True
        temperature=0.2,
        max_new_tokens=1000
    )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))
