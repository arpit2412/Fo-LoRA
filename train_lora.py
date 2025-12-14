#!/usr/bin/env python3
"""
Training Script for Standard LoRA Baseline
"""

import argparse
import os
from datetime import datetime

import torch
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# =====================================================================
# Argument Parsing
# =====================================================================

parser = argparse.ArgumentParser(description="Train a language model with Standard LoRA")
parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--max_steps", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--save_steps", type=int, default=50)

args = parser.parse_args()

# =====================================================================
# Configuration
# =====================================================================

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"LoRA-Baseline-{timestamp}"
os.environ["WANDB_RUN_NAME"] = run_name

# =====================================================================
# Model Loading
# =====================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if "llama" in args.model_name.lower() else "eager"
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# =====================================================================
# Dataset
# =====================================================================

dataset = load_dataset("json", data_files=args.dataset_path, split="train")

def formatting_prompts_func(examples):
    if isinstance(examples['instruction'], list):
        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']
    else:
        instructions = [examples['instruction']]
        inputs = [examples['input']]
        outputs = [examples['output']]

    output_texts = []
    for i in range(len(instructions)):
        text = f"<|user|>\n{instructions[i]}\n{inputs[i]}<|end|>\n<|assistant|>\n{outputs[i]}<|end|>"
        output_texts.append(text)
    return output_texts

dataset = dataset.map(lambda x: {"text": formatting_prompts_func(x)}, batched=True)

# =====================================================================
# LoRA Config
# =====================================================================

peft_config = LoraConfig(
    r=args.r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# =====================================================================
# Training
# =====================================================================

training_args = SFTConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    logging_steps=10,
    max_steps=args.max_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    fp16=False,
    bf16=True,
    optim="adamw_torch_fused",
    report_to="wandb",
    max_length=args.max_seq_length,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

print("Starting LoRA Baseline Training...")
trainer.train()
trainer.save_model(args.output_dir)
print(f"Model saved to {args.output_dir}")
