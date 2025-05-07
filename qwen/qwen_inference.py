# ─── 0) load tokenizer and base model on GPU ───────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, os, json

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments

base_ckpt = "Qwen/Qwen2.5-Math-7B"
lora_dir  = "/scratch/yb2510/ARC/qwen-grid-lora"          # <- folder you saved above

tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)

base = AutoModelForCausalLM.from_pretrained(
    base_ckpt,
    torch_dtype="auto",                    # fp16 because Colab GPU
    device_map={"": 0},                    # everything on cuda:0
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base, lora_dir).eval()  # attaches LoRA
model.to("cuda")

# ─── 1) craft a chat prompt for inference ─────────────────────────────────────
new_sample = {
    "grid": [[0,2,0],[0,0,2],[2,0,0]],
    "questions": [
        "How many shapes are there?",
        "How many contacts?",
        "What is the color of the biggest shape?"
    ]
}

messages = [
    {"role":"user", "content": f"Grid = {new_sample['grid']}\n{new_sample['questions'][0]}"},
    {"role":"assistant","content":""},        # we expect the model to answer
    {"role":"user", "content": new_sample["questions"][1]},
    {"role":"assistant","content":""},
    {"role":"user", "content": new_sample["questions"][2]},
    {"role":"assistant","content":""},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False          # greedy is fine for numeric answers
    )

output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
print("\nRaw model output:\n", output)