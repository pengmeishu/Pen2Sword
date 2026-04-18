# Pen2Sword

**Pen2Sword**: Accelerating LLM Fine-Tuning via Embedding Knowledge Transfer

## Overview

This repository contains the implementation of Pen2Sword and baseline methods for efficient fine-tuning research.

**Key Idea**: Transfer embedding representations from a small model to a large model, enabling faster and better fine-tuning with reduced computational cost.

![Framework](framework.pdf)

## Environment Setup

```bash
# Create and activate environment
conda create -n pen2sword python=3.10
conda activate pen2sword

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Configure environment variables
export WANDB_API_KEY="your_wandb_key_here"
```

## Quick Start

### 1. Train Model

**Pen2Sword (Our Method):**
```bash
python scripts/train.py \
    --METHOD pen2sword \
    --MODEL_PATH /path/to/Llama-2-13b \
    --EMBMODEL_PATH /path/to/Llama-2-7b-chat-hf \
    --ALPHA 0.5 \
    --DATASET_NAME "theblackcat102/evol-codealpaca-v1" \
    --SAVE_CHECKPOINT
```

**Baseline Methods:**
```bash
# Vanilla LoRA
python scripts/train_baseline.py --METHOD vanilla --MODEL_PATH /path/to/13b --DATASET_NAME "theblackcat102/evol-codealpaca-v1"

# Weak-to-Strong
python scripts/train_baseline.py --METHOD weak2strong --MODEL_PATH /path/to/13b --WEAK_MODEL_PATH /path/to/7b --DATASET_NAME "theblackcat102/evol-codealpaca-v1"

# Proxy Tuning
python scripts/train_baseline.py --METHOD proxy_tuning --MODEL_PATH /path/to/13b --WEAK_MODEL_PATH /path/to/7b --TUNED_WEAK_MODEL_PATH /path/to/7b-finetuned
```

### 2. Merge and Evaluate

After training, merge LoRA weights with the base model:

```bash
python scripts/merge_and_eval.py \
    --CHECKPOINT_PATH ./model_save/YOUR_PROJECT/checkpoint.safetensors \
    --MODEL_PATH /path/to/your/base-model \
    --OUTPUT_DIR ./model_save/merged/YOUR_MODEL
```

## Project Structure

```
pen2sword/
├── src/
│   ├── methods/
│   │   ├── pen2sword/          # Our method
│   │   ├── baseline/           # Baseline methods (vanilla, weak2strong, proxy_tuning)
│   │   └── dpf/                # Domain-Preserving Function for expert selection
│   ├── util/
│   └── dataset/
├── scripts/
│   ├── train.py                # Pen2Sword training
│   ├── train_baseline.py       # Baseline training
│   ├── dpf_evaluate.py         # DPF expert model selection
│   └── merge_and_eval.py       # Merge LoRA weights
├── framework.pdf                # Framework diagram
└── README.md
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--METHOD` | Method: `pen2sword`, `vanilla`, `weak2strong`, `proxy_tuning` | Required |
| `--MODEL_PATH` | Path to large model | Required |
| `--EMBMODEL_PATH` | Path to small model (for pen2sword) | Required |
| `--ALPHA` | Fusion ratio (for pen2sword) | 0.5 |
| `--DATASET_NAME` | HuggingFace dataset | theblackcat102/evol-codealpaca-v1 or meta-math/MetaMathQA |
| `--SAVE_CHECKPOINT` | Save checkpoints | Off |
| `--DEBUG` | Debug mode (1000 samples) | Off |

## Domain-Preserving Function (DPF)

DPF selects the best expert model for a target domain using GPT-4o evaluation.

```bash
export OPENAI_API_KEY="your-openai-api-key"
python scripts/dpf_evaluate.py --DOMAIN code --CANDIDATE_MODELS "Llama-2-7B-Chat,CodeLlama-7B" --TOP_K 1
```

