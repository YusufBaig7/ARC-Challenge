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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # <-- turn on 8-bit
    bnb_4bit_compute_dtype=torch.float16 # do matmuls in fp16 for stability
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,  
    trust_remote_code = True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)

with open("/scratch/yb2510/ARC/deepseek_first_finetune/qa_dataset.pkl" ,"rb") as f:
        seq_lengths = pickle.load(f)

raw_examples=seq_lengths
print(raw_examples[0]['questions'])

for x in raw_examples[0]['questions']:
    print(x)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

from datasets import Dataset
ds = Dataset.from_list(raw_examples)

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
    
    prompt = f"{system_msg}\n{user_msg}"
    answer = ",".join(map(str, ex["labels"]))
    answer = "["+ answer+"]"
    return prompt, answer

def tokenize_fn(ex):
    prompt, answer = make_prompt_and_answer(ex)
    full_input = prompt + " " + answer
    enc = tokenizer(
        full_input,
        padding="max_length",
        truncation=True,
        max_length=1024,
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
    return length <= 1024


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

model.save_pretrained("/scratch/yb2510/ARC/deepseek_first_finetune/shrunk_model_dir")
tokenizer.save_pretrained("/scratch/yb2510/ARC/deepseek_first_finetune/shrunk_model_dir")

print(f"✅ Shrunk vocab to {len(tokenizer)} tokens")


print("Length before Filter:\n",len(ds))
ds = ds.filter(is_under_800)
print("Length after Filter:\n",len(ds))

ds = ds.map(tokenize_fn, remove_columns=["grid", "questions", "labels"])

train_ds, eval_ds = ds, ds

lora_cfg = LoraConfig(
    r              = 256,
    lora_alpha     = 32,
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
    num_train_epochs            = 1,
    output_dir                  = "/scratch/yb2510/deepseek_r1_lora_14B",
    max_grad_norm=1.0,
    save_strategy      = "epoch",
    logging_steps      = 1,
    overwrite_output_dir=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm       = False,  
)

trainer = Trainer(
    model         = model,       
    args          = training_args,
    train_dataset = train_ds,
    data_collator = data_collator,
)

dl = trainer.get_train_dataloader()
batch = next(iter(dl))
print({k: v.shape for k, v in batch.items()})
print("non-ignored tokens:", (batch["labels"] != -100).sum().item())

trainer.train()