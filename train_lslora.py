import torch
from datasets import load_dataset
from peft import LSLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import os
import argparse
from datetime import datetime

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--run_name", type=str, default=None, help="WandB run name (optional)")
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=2)
args = parser.parse_args()

# Configuration
model_name = args.model_name
dataset_path = args.dataset_path
output_dir = args.output_dir
run_name = args.run_name
max_seq_length = args.max_seq_length
max_steps = args.max_steps
batch_size = args.batch_size

# Set WandB run name if provided
if run_name:
    os.environ["WANDB_RUN_NAME"] = run_name
    print(f"WandB run name: {run_name}")

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
print(f"Loading model: {model_name}")

# DDP device_map logic
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = 4 // world_size # Adjust gradient accumulation for DDP
else:
    gradient_accumulation_steps = 4

# OPTIMIZATION: Try to use Flash Attention 2 (50-100% speedup on attention)
# Prioritized for Llama models, optional for others due to quantization compatibility
is_llama = "llama" in model_name.lower()
use_flash_attention = is_llama  # Always try for Llama, optional for others

if use_flash_attention:
    try:
        print(f"Attempting to use Flash Attention 2 {'(Llama model detected)' if is_llama else ''}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # OPTIMIZATION: 2-3x faster attention
        )
        print("✓ Successfully loaded with Flash Attention 2")
    except Exception as e:
        print(f"⚠ Flash Attention 2 not available (falling back to standard attention)")
        if is_llama:
            print(f"  Note: Llama models benefit most from Flash Attention")
            print(f"  To install: pip install flash-attn --no-build-isolation")
        print(f"  Error: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
else:
    # Phi-3 and other models - standard attention with quantization
    print("Loading with standard attention (Flash Attention may have compatibility issues with 4-bit)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )

model = prepare_model_for_kbit_training(model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
print(f"Loading dataset from {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Format function
def formatting_prompts_func(examples):
    if isinstance(examples['instruction'], list):
        # Batch
        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']
    else:
        # Single example
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

# Apply formatting
print("Formatting dataset...")
dataset = dataset.map(lambda x: {"text": formatting_prompts_func(x)}, batched=True)

# Fourier LS-LoRA Config
print("Using Fourier LS-LoRA Config")
peft_config = LSLoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    fourier_reg_weight=0.01,  # L2 penalty for deviation from identity
    init_sigma=1.0,           # Initial tanh normalization scale
)

# Custom Trainer to handle Fourier Regularization
class FourierLoraTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        # Unwrap DDP if needed
        peft_model = model.module if hasattr(model, "module") else model

        # Add Fourier regularization
        if hasattr(peft_model, "get_fourier_loss"):
            reg_loss = peft_model.get_fourier_loss()
            active_adapter = peft_model.active_adapters[0]
            reg_weight = peft_model.peft_config[active_adapter].fourier_reg_weight

            total_loss = loss + reg_weight * reg_loss
            return (total_loss, outputs) if return_outputs else total_loss

        return (loss, outputs) if return_outputs else loss

# Training Arguments
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=2e-4,  # FIX: Increase from 5e-5 to 2e-4 (match standard LoRA)
    logging_steps=10,
    max_steps=max_steps,
    save_strategy="steps", # Save every 100 steps
    save_steps=100,
    fp16=False,
    bf16=True,
    optim="adamw_torch_fused",  # OPTIMIZATION: Fused optimizer for 5-10% speedup
    report_to="wandb",
    ddp_find_unused_parameters=False,
    max_length=max_seq_length,
    dataset_text_field="text",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Fix PyTorch 2.9 deprecation warning

    # OPTIMIZATION: DataLoader settings for 10-20% speedup (reduces CPU bottleneck)
    dataloader_num_workers=4,  # Parallel data loading (4 workers per device)
    dataloader_pin_memory=True,  # Faster CPU→GPU transfer
    dataloader_prefetch_factor=2,  # Prefetch 2 batches per worker
)

# Apply PEFT
model = get_peft_model(model, peft_config)

# OPTIMIZATION: torch.compile for 15-25% speedup (PyTorch 2.0+)
# NOTE: NOT COMPATIBLE with 4-bit quantization in PEFT (confirmed error)
# First step will be slower (compilation time), but subsequent steps faster
USE_TORCH_COMPILE = False  # DISABLED - incompatible with quantization
if USE_TORCH_COMPILE:
    try:
        print("Compiling model with torch.compile...")
        model = torch.compile(
            model,
            mode="reduce-overhead",  # Best for training workloads
            fullgraph=False,  # Allow dynamic shapes (needed for variable sequence lengths)
        )
        print("✓ Model compiled successfully")
    except Exception as e:
        print(f"⚠ torch.compile failed (continuing without compilation): {e}")

# Trainer
trainer = FourierLoraTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=None,  # Already applied above
    processing_class=tokenizer,
    args=training_args,
)

# Train
print("Starting LS-LoRA training...")
trainer.train()

# Free GPU memory after training
torch.cuda.empty_cache()

# Diagnostic: Check Fourier behavior
@torch.no_grad()
def check_fourier_behavior(model):
    """Check if Fourier coefficients are learning."""
    print("\n" + "="*60)
    print("FOURIER DIAGNOSTICS")
    print("="*60)

    identity = torch.tensor([0.0, 1.0, 0.0, 0.0])

    for name, module in model.named_modules():
        if hasattr(module, 'fourier_params'):
            for adapter_name, params in module.fourier_params.items():
                coeffs = params.weight
                # Deviation from identity [0, 1, 0, 0]
                dev = (coeffs - identity.to(coeffs.device)).abs().mean().item()

                # Non-linearity strength (sine + cosine coefficients)
                nonlin = (coeffs[2].abs() + coeffs[3].abs()).item()

                # Sigma value
                sigma = params.sigma.item()

                # Individual coefficients
                a0, a1, a2, a3 = coeffs.tolist()

                print(f"{name}/{adapter_name}:")
                print(f"  Coeffs: a₀={a0:.4f}, a₁={a1:.4f}, a₂={a2:.4f}, a₃={a3:.4f}")
                print(f"  σ={sigma:.4f}, deviation={dev:.4f}, non-linearity={nonlin:.4f}")

    print("="*60 + "\n")

# Run diagnostics
check_fourier_behavior(trainer.model)

# Save model
print(f"Saving model to {output_dir}")
torch.cuda.empty_cache()  # Free memory before save
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")
