# Dataset Configuration

This folder contains dataset configurations for Pen2Sword experiments.

## Overview

When using a dataset, the system needs to know:
1. **HuggingFace dataset name** - where to download from
2. **Input field name** - what column contains the input (e.g., `query`, `instruction`)
3. **Output field name** - what column contains the expected output
4. **Prompt template** - how to format the input/output for the model

## Supported Datasets

| Dataset | HuggingFace Name | Input Field | Output Field |
|---------|-----------------|-------------|--------------|
| **MetaMathQA** | `meta-math/MetaMathQA` | `query` | `response` |
| **Evol-CodeAlpaca** | `theblackcat102/evol-codealpaca-v1` | `instruction` | `output` |

## Adding a New Dataset

### Step 1: Find the dataset on HuggingFace

Search for your dataset at https://huggingface.co/datasets

### Step 2: Check the dataset structure

```python
from datasets import load_dataset

dataset = load_dataset("your/dataset-name")
print(dataset['train'][0])
# Look at the keys - these are your field names
```

### Step 3: Add to `datasets.py`

Edit `datasets/datasets.py` and add an entry:

```python
DATASET_CONFIGS = {
    # ... existing datasets ...

    "my_dataset": {
        "huggingface_name": "owner/dataset-name",
        "input_field": "your_input_field",
        "output_field": "your_output_field",
        "prompt_template": """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{input}

### Response:""",
        "description": "My custom dataset"
    },
}
```

### Step 4: Use it

```bash
python scripts/train.py --METHOD pen2sword \
    --DATASET_NAME "my_dataset" \
    --MODEL_PATH /path/to/model
```

## Prompt Template

The `{input}` placeholder will be replaced with the input field value.

Example template:
```python
"""Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{input}

### Response:"""
```

## Examples

### Math Dataset
```bash
python scripts/train.py --METHOD pen2sword \
    --DATASET_NAME "meta-math/MetaMathQA" \
    --MODEL_PATH /path/to/13b \
    --EMBMODEL_PATH /path/to/7b
```

### Code Dataset
```bash
python scripts/train.py --METHOD pen2sword \
    --DATASET_NAME "theblackcat102/evol-codealpaca-v1" \
    --MODEL_PATH /path/to/13b \
    --EMBMODEL_PATH /path/to/7b
```

## Checking Available Datasets

```python
from dataset.datasets import list_available_datasets, get_dataset_config

# List all available datasets
print(list_available_datasets())

# Get details of a specific dataset
config = get_dataset_config("meta-math/MetaMathQA")
print(config)
```
