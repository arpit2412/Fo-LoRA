#!/usr/bin/env python3
"""
Training Script for SincLoRA: Learnable Rank Adaptation via Sinc Basis Functions

This script trains a language model using SincLoRA, which learns optimal rank
allocation per layer through sinc basis functions.

Key features:
- Multi-GPU DDP training support
- 4-bit quantization with NF4
- Sinc regularization loss (encourages sparsity in alpha coefficients)
- Flash Attention 2 support (for compatible models)
- WandB logging

Usage:
    # Single GPU
    python train_sinclora.py --model_name microsoft/Phi-3-mini-4k-instruct \\
        --dataset_path dataset/commonsense_5k.json \\
        --output_dir output/sinclora_test \\
        --max_steps 500

    # Multi-GPU with DDP
    torchrun --nproc_per_node=2 train_sinclora.py \\
        --model_name microsoft/Phi-3-mini-4k-instruct \\
        --dataset_path dataset/commonsense_5k.json \\
        --output_dir output/sinclora_test \\
        --max_steps 500
"""

import argparse
import os
from datetime import datetime

import torch
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# Import SincLoRA (will be available after registration)
from peft.tuners.sinclora import SincLoraConfig


# =====================================================================
# Argument Parsing
# =====================================================================

parser = argparse.ArgumentParser(
    description="Train a language model with SincLoRA",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct",
                    help="HuggingFace model name")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="Path to training dataset (JSON format)")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory to save model checkpoints")
parser.add_argument("--run_name", type=str, default=None,
                    help="WandB run name (optional, auto-generated if not provided)")

# SincLoRA hyperparameters
parser.add_argument("--r", type=int, default=16,
                    help="Total rank budget (split across K basis functions)")
parser.add_argument("--K", type=int, default=8,
                    help="Number of sinc basis functions")
parser.add_argument("--lora_alpha", type=int, default=32,
                    help="LoRA scaling parameter")
parser.add_argument("--lora_dropout", type=float, default=0.05,
                    help="LoRA dropout probability")
parser.add_argument("--init_sigma", type=float, default=1.0,
                    help="Initial temperature for tanh normalization")
parser.add_argument("--omega_init", type=float, default=1.0,
                    help="Initial frequency for sinc functions")
parser.add_argument("--anchor_spacing", type=str, default="uniform",
                    choices=["uniform", "random"],
                    help="How to initialize anchor points")
parser.add_argument("--alpha_init", type=str, default="uniform",
                    choices=["uniform", "gaussian", "dirichlet"],
                    help="How to initialize alpha coefficients")
parser.add_argument("--sinc_reg_weight", type=float, default=0.01,
                    help="Regularization weight for sinc parameters")

# Training hyperparameters
parser.add_argument("--max_seq_length", type=int, default=2048,
                    help="Maximum sequence length")
parser.add_argument("--max_steps", type=int, default=500,
                    help="Number of training steps")
parser.add_argument("--batch_size", type=int, default=2,
                    help="Per-device training batch size")
parser.add_argument("--learning_rate", type=float, default=2e-4,
                    help="Learning rate")
parser.add_argument("--save_steps", type=int, default=100,
                    help="Save checkpoint every N steps")

args = parser.parse_args()


# =====================================================================
# Configuration
# =====================================================================

model_name = args.model_name
dataset_path = args.dataset_path
output_dir = args.output_dir
max_seq_length = args.max_seq_length
max_steps = args.max_steps
batch_size = args.batch_size

# Auto-generate run name if not provided
if args.run_name:
    run_name = args.run_name
else:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_short = model_name.split("/")[-1]
    dataset_short = os.path.basename(dataset_path).replace(".json", "")
    run_name = f"SincLoRA-{model_short}-{dataset_short}-{timestamp}"

os.environ["WANDB_RUN_NAME"] = run_name
print(f"\nWandB run name: {run_name}")
print(f"Output directory: {output_dir}\n")


# =====================================================================
# Model Loading with Quantization
# =====================================================================

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading model: {model_name}")

# DDP device_map logic
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = 4 // world_size
    print(f"DDP enabled: world_size={world_size}, local_rank={device_map['']}")
else:
    device_map = "auto"
    gradient_accumulation_steps = 4
    print("Single GPU mode")

# Try to use Flash Attention 2 (prioritize for Llama models)
is_llama = "llama" in model_name.lower()
use_flash_attention = is_llama

if use_flash_attention:
    try:
        print(f"Attempting to use Flash Attention 2 {'(Llama model detected)' if is_llama else ''}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        print("✓ Successfully loaded with Flash Attention 2")
    except Exception as e:
        print(f"⚠ Flash Attention 2 not available (falling back to standard attention)")
        if is_llama:
            print(f"  Note: Llama models benefit most from Flash Attention")
            print(f"  To install: pip install flash-attn --no-build-isolation")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
else:
    print("Loading with standard attention")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )

model = prepare_model_for_kbit_training(model)


# =====================================================================
# Tokenizer
# =====================================================================

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# =====================================================================
# Dataset Loading and Formatting
# =====================================================================

print(f"\nLoading dataset from {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")


def formatting_prompts_func(examples):
    """Format examples into chat template."""
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
        instruction = instructions[i]
        input_text = inputs[i]
        response = outputs[i]

        if input_text:
            text = f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n{response}<|end|>"
        else:
            text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
        output_texts.append(text)
    return output_texts


print("Formatting dataset...")
dataset = dataset.map(lambda x: {"text": formatting_prompts_func(x)}, batched=True)
print(f"Dataset size: {len(dataset)} examples\n")


# =====================================================================
# SincLoRA Configuration
# =====================================================================

print("Creating SincLoRA Config:")
print(f"  Total rank: {args.r}")
print(f"  K basis functions: {args.K}")
print(f"  Rank per basis: {args.r // args.K}")
print(f"  Alpha: {args.lora_alpha}")
print(f"  Dropout: {args.lora_dropout}")
print(f"  Sinc reg weight: {args.sinc_reg_weight}")
print(f"  Init sigma: {args.init_sigma}")
print(f"  Omega init: {args.omega_init}")
print(f"  Anchor spacing: {args.anchor_spacing}")
print(f"  Alpha init: {args.alpha_init}\n")

peft_config = SincLoraConfig(
    r=args.r,
    K=args.K,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    init_sigma=args.init_sigma,
    omega_init=args.omega_init,
    anchor_spacing=args.anchor_spacing,
    alpha_init=args.alpha_init,
    sinc_reg_weight=args.sinc_reg_weight,
)


# =====================================================================
# Custom Trainer with Sinc Regularization
# =====================================================================

class SincLoraTrainer(SFTTrainer):
    """Custom trainer that adds sinc regularization loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        # Unwrap DDP if needed
        peft_model = model.module if hasattr(model, "module") else model

        # Add sinc regularization
        if hasattr(peft_model, "get_sinc_loss"):
            reg_loss = peft_model.get_sinc_loss()
            active_adapter = peft_model.active_adapters[0]
            reg_weight = peft_model.peft_config[active_adapter].sinc_reg_weight

            total_loss = loss + reg_weight * reg_loss

            # Log regularization loss (optional)
            if self.state.global_step % self.args.logging_steps == 0:
                if hasattr(self, "accelerator"):
                    if self.accelerator.is_main_process:
                        print(f"  [Step {self.state.global_step}] Reg loss: {reg_loss.item():.4f}")

            return (total_loss, outputs) if return_outputs else total_loss

        return (loss, outputs) if return_outputs else loss


# =====================================================================
# Training Configuration
# =====================================================================

training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    logging_steps=10,
    max_steps=max_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    fp16=False,
    bf16=True,
    optim="adamw_torch_fused",  # Fused optimizer for speedup
    report_to="wandb",
    ddp_find_unused_parameters=False,
    max_length=max_seq_length,
    dataset_text_field="text",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # DataLoader optimizations
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2,
)


# =====================================================================
# Apply PEFT and Train
# =====================================================================

print("Applying SincLoRA to model...")
model = get_peft_model(model, peft_config)

print("\nTrainable parameters:")
model.print_trainable_parameters()

# Initialize trainer
print("\nInitializing SincLoRA trainer...")
trainer = SincLoraTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,  # SFTTrainer uses processing_class, not tokenizer
)

# Train!
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
trainer.train()

# Save final model
print("\n" + "=" * 70)
print("Training complete! Saving final model...")
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")


# =====================================================================
# Post-Training Analysis
# =====================================================================

print("\n" + "=" * 70)
print("ANALYZING LEARNED SINC RANKS")
print("=" * 70)

try:
    # Import analysis script
    import sys
    sys.path.append("scripts")
    from compute_stable_rank import analyze_sinc_ranks

    # Run analysis
    analyze_sinc_ranks(
        model_path=output_dir,
        output_file=f"{output_dir}/sinc_ranks.json"
    )
except Exception as e:
    print(f"Could not run automatic analysis: {e}")
    print(f"Run manually: python scripts/compute_stable_rank.py --model_path {output_dir}")

print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)
print(f"\nNext steps:")
print(f"  1. Visualize sinc basis: python visualization/plot_sinc_learned.py --model_path {output_dir}")
print(f"  2. Evaluate model: python eval.py --model_path {output_dir}")
print(f"  3. Check WandB for training metrics: https://wandb.ai")
