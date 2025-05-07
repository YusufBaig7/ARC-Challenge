import torch, os, json

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Math-7B", trust_remote_code=True, cache_dir = "/scratch/yb2510/hf_models",
)

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-7B",
    torch_dtype=torch.float8,          # Colab T4/P100 → fp16 is safe
    device_map={"": 0},                 # put every layer on cuda:0
    trust_remote_code=True,
    cache_dir = "/scratch/yb2510/hf_models",
).cuda()                                # explicit move (harmless if already there)


sample = {
    "grid": [[0,0,0],[0,0,2],[2,0,2]],
    "questions": [
        "How many shapes are there?",
        "How many contacts?",
        "What is the color of the biggest shape?"
    ],
    "answers": [2, 0, 2]
}

messages = [
    {"role":"system", "content": " You are a puzzle solving assistant, a prompt is given please output the answers to all three questions as a combined list in a single line\n"},
    {"role":"user",      "content": f"Grid = {sample['grid']}\n{sample['questions'][0]}"},
    {"role":"assistant", "content": str(sample["answers"][0])},
    {"role":"user",      "content": sample["questions"][1]},
    {"role":"assistant", "content": str(sample["answers"][1])},
    {"role":"user",      "content": sample["questions"][2]},
    {"role":"assistant", "content": str(sample["answers"][2])},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
ds   = Dataset.from_list([{"text": text}])


peft_cfg = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"]
)
model = get_peft_model(base, peft_cfg)
model.print_trainable_parameters()

trainer = SFTTrainer(
    model           = model,
    train_dataset   = ds,
    peft_config     = peft_cfg,
    formatting_func = None,
    args = TrainingArguments(
        output_dir          = "qwen-grid-lora",
        num_train_epochs    = 4,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        learning_rate       = 2e-4,
        fp16                = True,     # matches torch_dtype above
        logging_steps       = 1,
        logging_strategy    = "steps",
        report_to           = "none",
        disable_tqdm        = False,
        save_strategy       = "no",
    ),
)

trainer.train()

trainer.model.save_pretrained("qwen-grid-lora")
tokenizer.save_pretrained("qwen-grid-lora")
