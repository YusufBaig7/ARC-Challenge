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

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from small_tokenizer import shrink_embeddings
from typing import Dict, List, Tuple

train_data_path = "/scratch/yb2510/ARC/transduction/data/training/"
augmented_data = "/scratch/yb2510/ARC/Aug_data/"
list1 = os.listdir(train_data_path)
list2 = os.listdir(augmented_data)
complete_data = []

for x in list1:
    data = json.load(open(train_data_path+x))
    complete_data.append(data)

for fname in os.listdir(augmented_data):
    # 1) Only process .json files
    if not fname.lower().endswith(".json"):
        print(f"Skipping non-JSON file: {fname}")
        continue

    full_path = os.path.join(augmented_data, fname)

    # 2) Skip zero-byte files
    if os.path.getsize(full_path) == 0:
        print(f"Skipping empty file: {fname}")
        continue

    # 3) Try to load, catch bad JSON
    try:
        with open(full_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Skipping invalid JSON in {fname}: {e}")
        continue

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


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    offload_to_cpu=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "/scratch/yb2510/ARC/deepseek_first_finetune_3_epoch/merged",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code = True,
)


model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained("/scratch/yb2510/ARC/deepseek_first_finetune_3_epoch/merged", trust_remote_code=True, use_fast=False)

raw_examples=complete_data

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

from datasets import Dataset
ds = Dataset.from_list(raw_examples)

def tokenize_fn(ex):
    prompt, answer = make_prompt_and_answer(ex)
    full_input = prompt + " " + answer
    enc = tokenizer(
        full_input,
        truncation=True,
        max_length=8192,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    labels     = enc["input_ids"].copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
    enc["labels"] = labels
    return enc

def is_under_800(example):
    prompt, answer = make_prompt_and_answer(example)
    length = len(tokenizer(prompt + " " + answer,
                          truncation=False,
                          add_special_tokens=False)["input_ids"])
    return length <= 8192

corpus = []
for ex in ds:
  prompt, answer = make_prompt_and_answer(ex)
  corpus.append(prompt + " " + answer)

print(corpus[0])
  
big_corpus = "\n".join(corpus)
mapping = shrink_embeddings(
    model,
    tokenizer,
    corpus=big_corpus,
    keep_special_tokens=True,   # keep <pad>, <eos>, etc.
    keep_normalizer=False,      # drop any Unicode normalizer
    keep_token_order=True,      # preserve ordering of ids
)


# peft_model = PeftModel.from_pretrained(
#     model,
#     "/scratch/yb2510/deepseek_r1_lora_14B_3_epoch/checkpoint-15000",  # the adapter‐only folder
#     torch_dtype=torch.float16,                      # match your fine-tune dtype
#     low_cpu_mem_usage=True,
#     trust_remote_code = True,
#     # is_trainable=False, 
# ).to(DEVICE)

model.save_pretrained("/scratch/yb2510/ARC/transduction/shrunk_model_dir")
tokenizer.save_pretrained("/scratch/yb2510/ARC/transduction/shrunk_model_dir")

print(f"✅ Shrunk vocab to {len(tokenizer)} tokens")


# 1) Define a length‐computing function
def compute_length(example):
    prompt, answer = make_prompt_and_answer(example)
    # tokenize without special tokens so you get the *raw* length
    input_ids = tokenizer(
        prompt + " " + answer,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    return {"length": len(input_ids)}

# 2) Run it over the whole dataset
ds_with_lengths = ds.map(compute_length)

# 3) Pull out the lengths and take max
max_len = max(ds_with_lengths["length"])
print(f"Max tokenized length: {max_len}")


print("Length before Filter:\n",len(ds))
ds = ds.filter(is_under_800)
print("Length after Filter:\n",len(ds))

ds = ds.map(tokenize_fn, remove_columns=["train", "test"])

train_ds, eval_ds = ds, ds

lora_cfg = LoraConfig(
    r              = 256,
    lora_alpha     = 128,
        target_modules = [
            # per-layer projections
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # embeddings and final head
            "embed_tokens", "lm_head"
        ],
    lora_dropout   = 0.05,
    bias           = "none",
)

model = get_peft_model(model, lora_cfg)

training_args = TrainingArguments(
    per_device_train_batch_size = 1,
    learning_rate               = 1e-5,
    num_train_epochs            = 10,
    gradient_checkpointing=True,
    output_dir                  = "/scratch/yb2510/ARC/transduction/saved_model_augmeted",
    fp16=True,
    max_grad_norm=1.0,
    save_strategy      = "epoch",
    logging_steps      = 1,
    overwrite_output_dir=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm       = False,
    pad_to_multiple_of=None,
)

trainer = Trainer(
    model         = model,       
    args          = training_args,
    train_dataset = train_ds,
    data_collator = data_collator,
)

model.print_trainable_parameters()

trainer.train()