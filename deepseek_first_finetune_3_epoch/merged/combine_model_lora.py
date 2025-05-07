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


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    offload_to_cpu=False,
)

# 1) Load your base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "/scratch/yb2510/ARC/deepseek_first_finetune/shrunk_model_dir",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code = True,
)

tokenizer = AutoTokenizer.from_pretrained("/scratch/yb2510/ARC/deepseek_first_finetune/shrunk_model_dir", trust_remote_code=True, use_fast=False)

# 2) Load in your LoRA adapter
peft_model = PeftModel.from_pretrained(
    model,
    "/scratch/yb2510/deepseek_r1_lora_14B_3_epoch/checkpoint-15000",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 3) Merge LoRA weights into the base model
merged_model = peft_model.merge_and_unload()  # returns a standard Transformers model with LoRA applied :contentReference[oaicite:0]{index=0}

# 4) Save the merged model & tokenizer as a single package
merged_model.save_pretrained("/scratch/yb2510/ARC/deepseek_first_finetune_3_epoch/merged/", safe_serialization=True)
tokenizer.save_pretrained("/scratch/yb2510/ARC/deepseek_first_finetune_3_epoch/merged/")
