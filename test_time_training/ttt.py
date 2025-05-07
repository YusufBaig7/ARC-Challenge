#!/usr/bin/env python3
import argparse
import json
import random
import torch
import numpy as np
from copy import deepcopy
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from small_tokenizer import shrink_embeddings
from functools import partial


def rotate_grid(grid, k):
    return np.rot90(np.array(grid), k).tolist()

def flip_grid(grid, axis):
    return np.flip(np.array(grid), axis=axis).tolist()

def permute_colors(grid):
    arr = np.array(grid)
    colors = np.unique(arr)
    # only permute non-black colors
    non_black = [c for c in colors if c != 0]
    permuted = np.random.permutation(non_black)
    # build mapping: black → black, others → permuted
    perm = {0: 0}
    perm.update({orig: new for orig, new in zip(non_black, permuted)})
    # apply map
    return np.vectorize(lambda v: perm[v])(arr).tolist()


def sample_augment_params():

    # 1) rotation
    k = random.choice([0, 1, 2, 3])
    # 2) flips
    flip_h = random.random() < 0.5
    flip_v = random.random() < 0.5
    # 3) color permutation (on all non‐zero colors 1–9)
    non_black = list(range(1, 10))
    permuted = np.random.permutation(non_black)
    color_mapping = {0: 0}
    color_mapping.update({orig: int(new) for orig, new in zip(non_black, permuted)})

    return k, flip_h, flip_v, color_mapping

def apply_augment(grid, params):
    """
    Apply the sampled params to one grid:
      1) rotate by k*90° CCW
      2) optionally flip horizontally/vertically
      3) remap colors using the same color_mapping
    Returns a NEW list‐of‐lists.
    """
    k, flip_h, flip_v, color_mapping = params

    # to numpy
    arr = np.array(grid)

    # rotate
    arr = np.rot90(arr, k)

    # flips
    if flip_h:
        arr = np.flip(arr, axis=0)
    if flip_v:
        arr = np.flip(arr, axis=1)

    # color remapping
    vec_map = np.vectorize(lambda v: color_mapping.get(int(v), int(v)))
    arr = vec_map(arr)

    # back to Python lists
    return arr.tolist()

def arc_augment(x, y):

    k = random.choice([0,1,2,3])
    x2, y2 = rotate_grid(x, k), rotate_grid(y, k)

    if random.random() < 0.5:
        x2, y2 = flip_grid(x2, 0), flip_grid(y2, 0)
    if random.random() < 0.5:
        x2, y2 = flip_grid(x2, 1), flip_grid(y2, 1)

    if random.random() < 0.5:
        x2, y2 = permute_colors(x2), permute_colors(y2)
    return x2, y2

from itertools import combinations

def build_pseudo_tasks(problem, m=10):
    x_tr = [ex['input']  for ex in problem['train']]
    y_tr = [ex['output'] for ex in problem['train']]
    N    = len(x_tr)
    tasks = []


    for i in range(N):
        # hold out exactly one example
        x_test_f, y_test_f = x_tr[i], y_tr[i]
        # the rest are your training pool
        x_tr_p = x_tr[:i] + x_tr[i+1:]
        y_tr_p = y_tr[:i] + y_tr[i+1:]

        for _ in range(m):
          params = sample_augment_params()
          aug_x_tr = [apply_augment(x, params) for x in x_tr_p]
          aug_y_tr = [apply_augment(y, params) for y in y_tr_p]
          ax_test  = apply_augment(x_test_f, params)
          ay_test  = apply_augment(y_test_f, params)

          tasks.append({
              'train': [{'input': xx, 'output': yy}
                        for xx, yy in zip(aug_x_tr, aug_y_tr)],
              'test':  [{'input': ax_test, 'output': ay_test}]
          })


    return tasks



color_map = {
    0:"Black",1:"Blue",2:"Red",3:"Green",4:"Yellow",
    5:"Grey",6:"Fuchsia",7:"Orange",8:"Teal",9:"Brown"
}

def grid_to_str(grid):
    return "\n".join(" ".join(color_map[c] for c in row)
                     for row in grid)

def make_prompt_and_answer(task):

    sys_msg = """----- Role: system --------------------
You are a world-class puzzle solver with exceptional pattern recognition
skills. Your task is to analyze puzzles, spot patterns, and provide direct
solutions. You are kind of a local solver - your strength is in finding local patterns and local subproblems to solve the puzzels.
"""
    user_msg = """----- Role: user --------------------
Given input-output grid pairs as reference examples, carefully observe the
patterns to predict the output grid for new test input. Each pair follows
the same transformation rule. Grids are 2D arrays represented as strings,
with cells (colors) separated by spaces and rows by newlines.
There must be local subproblems in input, which might have one-to-one relation in output.
Solve these local problems one by one and then predict the output grid of test input.
Here are the input and output grids for the reference examples:
"""

    for i, ex in enumerate(task['train'], 1):
        user_msg += f"\nExample {i}\nInput:\n{grid_to_str(ex['input'])}\n"
        user_msg += f"Output:\n{grid_to_str(ex['output'])}\n"
    ti = task['test'][0]['input']
    user_msg += f"""Here is the input grid for the test example:
Input:
{grid_to_str(ti)}
Directly provide the output grid(s) corresponding to the given test input
grids, based on the patterns observed in the reference examples.
"""
    # user_msg += f"\nTest Input:\n{grid_to_str(ti)}\n"
    prompt = sys_msg + "\n" + user_msg

    to = task['test'][0]['output']
    answer = "----- Role: assistant ----\nThe output is:\n```\n"
    answer += grid_to_str(to) + "\n```"
    return prompt, answer

def tokenize_example(ex, tokenizer, max_len=8192):
    prompt, answer = make_prompt_and_answer(ex)
    enc = tokenizer(
        prompt + answer,
        truncation=True,
        padding=False,    # ← pad every example to `max_len`
        max_length=max_len
    )
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    labels = enc["input_ids"].copy()
    labels[:len(p_ids)] = [-100] * len(p_ids)
    enc["labels"] = labels
    return enc

def main():

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # <-- turn on 8-bit
    bnb_4bit_compute_dtype=torch.float16 # do matmuls in fp16 for stability
    )

    base_model_dir = '/scratch/yb2510/ARC/transduction/merged_aug'

    # input_json = '/scratch/yb2510/ARC/test_time_training/data/evaluation/ff72ca3e.json'
    # base_model_dir = '/scratch/yb2510/ARC/transduction/merged_aug'
    m = 200

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_dir, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=64, lora_alpha=64,
        target_modules = [

          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj",
          "embed_tokens", "lm_head"
      ],
        lora_dropout=0.05, bias="none"
    )
    
    ttt_args = TrainingArguments(
        num_train_epochs=1,
        logging_steps=10,
        per_device_train_batch_size=1,        # 2 examples per step
        gradient_accumulation_steps=1,        # effective batch size 4
        learning_rate=1e-4,                   # LoRA adapters
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        save_strategy="no",
        output_dir="/scratch/yb2510/ARC/test_time_training",
        overwrite_output_dir=False,
        fp16=True,
        optim="adamw_torch",

    )

    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    list1 = os.listdir("/scratch/yb2510/ARC/test_time_training/data/evaluation")
    complete_data = []
    train_data_path = "/scratch/yb2510/ARC/test_time_training/data/evaluation/"

    for x in list1:
        data = json.load(open(train_data_path+x))
        complete_data.append(data)

    corpus2 = []
    ds2 = Dataset.from_list(complete_data)

    for ex in ds2:
      prompt, answer = make_prompt_and_answer(ex)
      corpus2.append(prompt + " " + answer)
      
    big_corpus = "\n".join(corpus2)

    mapping = shrink_embeddings(
    model_base,
    tokenizer,
    corpus=big_corpus,
    keep_special_tokens=True,   # keep <pad>, <eos>, etc.
    keep_normalizer=False,      # drop any Unicode normalizer
    keep_token_order=True,      # preserve ordering of ids
    )

    print(f"✅ Shrunk vocab to {len(tokenizer)} tokens")
    evaluation_dir = "/scratch/yb2510/ARC/test_time_training/data/evaluation"
    all_files = sorted(os.listdir(evaluation_dir))
    for fname in all_files[:50]:
      path = os.path.join(evaluation_dir, fname)
      problem = json.load(open(path))

      pseudo = build_pseudo_tasks(problem, m)
      hf_ds = Dataset.from_list(pseudo)

      corpus = []
      for ex in hf_ds:
        prompt, answer = make_prompt_and_answer(ex)
        corpus.append(prompt + " " + answer)

      tokenized = hf_ds.map(
          lambda ex: tokenize_example(ex, tokenizer),
          remove_columns=["train","test"]
      )

      # model = get_peft_model(model_base, lora_cfg).to(DEVICE)
      model = get_peft_model(deepcopy(model_base), lora_cfg).to(DEVICE)
      model.gradient_checkpointing_enable()
      trainer = Trainer(
          model=model,
          args=ttt_args,
          train_dataset=tokenized,
          data_collator=data_collator,
      )
      trainer.train()

      real_task = {
          "train": problem["train"],
          "test":  [{"input": problem["train"][0]["input"],
                    "output": problem["train"][0]["output"]}]
      }
      prompt, _ = make_prompt_and_answer(real_task)
      inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

      out = model.generate(
          **inputs,
          num_beams=3,
          max_new_tokens=512,
          early_stopping=True
      )
      pred = tokenizer.decode(
          out[0][ inputs["input_ids"].size(-1): ],
          skip_special_tokens=True
      )
      del model, trainer, tokenized, hf_ds, pseudo
      torch.cuda.empty_cache()
      
      with open(fname, "w") as f:
        true_str = grid_to_str(problem["test"][0]["output"])
        f.write(pred + '\n\n\n\n Correct Output' + true_str)

if __name__ == "__main__":
  main()
