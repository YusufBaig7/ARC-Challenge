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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 1) tokenizer and base model share the same shrunk vocab
tokenizer = AutoTokenizer.from_pretrained(
    "/scratch/yb2510/ARC/deepseek_first_finetune/shrunk_model_dir",  trust_remote_code=True, use_fast=False
)
model = AutoModelForCausalLM.from_pretrained(
    "/scratch/yb2510/ARC/deepseek_first_finetune/shrunk_model_dir",
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# 2) overlay the LoRA weights
model = PeftModel.from_pretrained(
    model,
    "/scratch/yb2510/deepseek_r1_lora_14B/checkpoint-5000",
    is_train=False,         # we’re in inference mode
)

# 3) prep for generation
model.eval()
model.to(DEVICE)

def make_prompt_and_answer(ex):
    # 1) Render the grid as rows of space-separated values
    grid_str = "\n".join(" ".join(map(str, row)) for row in ex["grid"])

    # 2) Render the numbered questions
    qs_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(ex["questions"]))

    # 3) Build the ROLE: system section
    system_msg = """----- Role: system --------------------
You are a world-class puzzle solver with exceptional pattern-recognition skills.
Your task is to analyze grids, spot patterns and shapes, and provide direct answers
for the questions.
"""

    # 4) Build the ROLE: user section
    user_msg = f"""----- Role: user --------------------
Given an input grid, carefully recognize the shapes and their behavior with each other.
Grids are 2D arrays represented as strings, with cells (colors) separated by spaces
and rows by newlines. Consider black color as backgorund.

Here is the grid:
{grid_str}

Questions:
{qs_str}

Please output only a single list of size 7, each value is a non-negative integer separted by comma.


Your answer:"""

    # 5) Combine and return
    prompt = f"{system_msg}\n{user_msg}"
    answer = ",".join(map(str, ex["labels"]))
    answer = "["+ answer+"]"
    return prompt, answer


with open("/scratch/yb2510/ARC/deepseek_first_finetune/qa_dataset.pkl" ,"rb") as f:
    seq_lengths = pickle.load(f)

raw_examples=seq_lengths

prompt, answer = make_prompt_and_answer(raw_examples[5])

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
    max_new_tokens=50
)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
