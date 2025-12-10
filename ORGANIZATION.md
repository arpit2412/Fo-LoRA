# Project Organization Guide

This document describes the file organization structure for Fourier LS-LoRA experiments.

## Directory Structure

```
/home/arpit/peft/
├── src/peft/                      # PEFT library source code
│   └── tuners/lslora.py          # Fourier LS-LoRA implementation
├── scripts/                       # Training and evaluation scripts
│   ├── train_phi3_170k.sh        # Training script examples
│   ├── eval_lslora.sh            # Evaluation scripts
│   └── README.md                 # Script documentation
├── visualization/                 # Plotting and visualization tools
│   ├── print_learned_fourier.py  # Print coefficients
│   ├── plot_splines.py           # Generate plots
│   └── README.md                 # Visualization documentation
├── eval_results/                  # Evaluation JSON outputs
│   ├── eval_results_*.json       # Benchmark results
│   └── README.md                 # Results documentation
├── dev/                          # Development and testing
│   ├── test_*.py                 # Test scripts
│   ├── check_*.py                # Diagnostic tools
│   └── README.md                 # Development documentation
├── output/                       # Training outputs
│   └── Fo-LoRA-{model}-{dataset}-{timestamp}/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── checkpoint-*/
├── logs/                         # Training logs
│   └── training-Fo-LoRA-{model}-{dataset}-{timestamp}.log
├── plots/                        # Visualization outputs
│   └── Fo-LoRA-{model}-{dataset}-{timestamp}/
│       └── *.png
├── dataset/                      # Training datasets
│   ├── commonsense_5k.json
│   └── commonsense_170k.json
├── train_lslora.py              # Main training script (root)
├── ORGANIZATION.md              # This file
├── FIXES_APPLIED.md             # Bug fix history
└── MATHS.md                     # Mathematical documentation
```

## Naming Conventions

### Run Names
**Format**: `Fo-LoRA-{model}-{dataset}-{timestamp}`

**Components**:
- `Fo-LoRA`: Fixed prefix for Fourier LoRA
- `{model}`: Short model name (phi3, llama, mistral)
- `{dataset}`: Dataset identifier (commonsense5k, commonsense170k)
- `{timestamp}`: `$(date +%Y%m%d-%H%M%S)` - ensures uniqueness

**Examples**:
```
Fo-LoRA-phi3-commonsense170k-20251210-113956
Fo-LoRA-llama-commonsense5k-20251210-120000
Fo-LoRA-mistral-wikitext-20251210-140000
```

### Model Short Names
```
microsoft/Phi-3-mini-4k-instruct    → phi3
meta-llama/Llama-3.1-8B             → llama
mistralai/Mistral-7B-v0.1           → mistral
```

### Dataset Short Names
```
commonsense_5k.json       → commonsense5k
commonsense_170k.json     → commonsense170k
wikitext_train.json       → wikitext
```

### Output Directories
All outputs follow the run name pattern:

```
output/Fo-LoRA-{model}-{dataset}-{timestamp}/
logs/training-Fo-LoRA-{model}-{dataset}-{timestamp}.log
plots/Fo-LoRA-{model}-{dataset}-{timestamp}/
eval_results/eval_results_{model}_{dataset}_{benchmark}_{timestamp}.json
```

## File Organization Rules

### 1. Root Directory
**Keep minimal** - only essential files:
- `train_lslora.py` - Main training script
- `*.md` - Documentation files (ORGANIZATION, FIXES_APPLIED, MATHS)
- `*.py` - Core implementation files (if not in src/)

**Do NOT keep in root**:
- Training scripts (→ `scripts/`)
- Test files (→ `dev/`)
- Plot scripts (→ `visualization/`)
- Evaluation results (→ `eval_results/`)
- Outputs (→ `output/`)
- Logs (→ `logs/`)

### 2. Scripts Directory
**Purpose**: Training, evaluation, and utility scripts

**Contains**:
- `*.sh` - Shell scripts for training/evaluation
- `*.py` - Utility scripts (calculate_score.py, eval.py)

**Naming**: `{action}_{model}_{dataset}.sh`

### 3. Visualization Directory
**Purpose**: All plotting and visualization tools

**Contains**:
- `print_learned_fourier.py` - Print coefficients
- `plot_*.py` - Generate plots
- Visualization utilities

**Outputs go to**: `plots/{run_name}/`

### 4. Evaluation Results Directory
**Purpose**: JSON outputs from benchmarks

**Contains**: `eval_results_{model}_{dataset}_{benchmark}_{timestamp}.json`

**Format**: Consistent JSON structure for easy comparison

### 5. Development Directory
**Purpose**: Test scripts and debugging tools

**Contains**:
- `test_*.py` - Test scripts
- `check_*.py` - Diagnostic tools
- `debug_*.py` - Debugging utilities

**Keep temporary** - Clean up old tests regularly

### 6. Output Directory
**Purpose**: Trained model adapters and checkpoints

**Auto-created** by training scripts

**Contains**:
```
output/Fo-LoRA-{model}-{dataset}-{timestamp}/
├── adapter_model.safetensors    # Trained weights
├── adapter_config.json          # Config
├── tokenizer files              # Tokenizer
└── checkpoint-{step}/           # Checkpoints (every 100 steps)
```

### 7. Logs Directory
**Purpose**: Full training logs with timestamps

**Auto-created** by training scripts

**Format**: `training-Fo-LoRA-{model}-{dataset}-{timestamp}.log`

### 8. Plots Directory
**Purpose**: Visualization outputs

**Created by** visualization scripts

**Structure**:
```
plots/Fo-LoRA-{model}-{dataset}-{timestamp}/
├── layer_00_qkv_proj.png
├── layer_01_qkv_proj.png
└── summary.png
```

## Workflow Best Practices

### Starting a New Training Run

1. **Create training script** in `scripts/`:
```bash
nano scripts/train_{model}_{dataset}.sh
```

2. **Use template**:
```bash
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME="Fo-LoRA-{model}-{dataset}-${TIMESTAMP}"
OUTPUT_DIR="output/${RUN_NAME}"
LOG_FILE="logs/training-${RUN_NAME}.log"

mkdir -p "${OUTPUT_DIR}" "logs"

python train_lslora.py \
    --model_name {full_model_path} \
    --dataset_path dataset/{dataset_file} \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --max_steps {steps} \
    --batch_size {batch_size} \
    2>&1 | tee "${LOG_FILE}"
```

3. **Run training**:
```bash
chmod +x scripts/train_{model}_{dataset}.sh
./scripts/train_{model}_{dataset}.sh
```

### After Training Completes

1. **Print coefficients**:
```bash
python visualization/print_learned_fourier.py output/{run_name}
```

2. **Generate plots**:
```bash
python visualization/plot_splines.py \
    --model_path output/{run_name} \
    --output_dir plots/{run_name}
```

3. **Run evaluation**:
```bash
./scripts/eval_lslora.sh output/{run_name}
```

4. **Check results**:
```bash
cat eval_results/eval_results_{model}_{dataset}_{benchmark}_{timestamp}.json | jq
```

### Comparing Multiple Runs

```bash
# List all runs
ls -lh output/

# Compare accuracies
for file in eval_results/*.json; do
    echo "$file: $(jq '.accuracy' $file)"
done

# View training logs
tail -f logs/training-Fo-LoRA-*.log
```

## Migration Checklist

When reorganizing or starting fresh:

- [ ] All training scripts in `scripts/`
- [ ] All plotting tools in `visualization/`
- [ ] All eval results in `eval_results/`
- [ ] All test files in `dev/`
- [ ] Output directories follow naming convention
- [ ] Log files follow naming convention
- [ ] WandB run names match directory names
- [ ] No clutter in root directory (max 15 files)
- [ ] README.md exists in each subdirectory
- [ ] All timestamps use `$(date +%Y%m%d-%H%M%S)` format

## Questions?

See directory-specific README.md files:
- `scripts/README.md` - Training and evaluation
- `visualization/README.md` - Plotting tools
- `eval_results/README.md` - Benchmark results
- `dev/README.md` - Testing and development

Or check the main documentation:
- `FIXES_APPLIED.md` - Bug fix history
- `MATHS.md` - Mathematical documentation
- Plan file: `~/.claude/plans/floating-squishing-book.md` - Fourier parameter save/load fix
