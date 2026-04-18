"""
Dataset configurations.

This module defines known datasets with their HuggingFace names,
field mappings, and prompt templates.

To add a new dataset:
1. Add an entry to DATASET_CONFIGS below
2. Specify the input_field, output_field, and prompt_template
3. Use it by passing --DATASET_NAME to train.py

Usage:
    python scripts/train.py --METHOD pen2sword ... --DATASET_NAME "my_dataset"
"""

# Prompt template for math tasks (matching reference code exactly)
MATH_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n ### Instruction:\n{input}\n\n### Response: Let's think step by step."

# Prompt template for code tasks (matching reference code exactly)
CODE_PROMPT = "Below is an instruction that describes a programming task. Write a program that appropriately completes the request.\n\n ### Instruction:\n{input}\n\n### Response:"


# Known datasets configuration
# Format:
#   "dataset_name": {
#       "huggingface_name": "full/path/on/huggingface",
#       "input_field": "field name for input",
#       "output_field": "field name for output",
#       "prompt_template": prompt template string,
#       "description": "brief description"
#   }

DATASET_CONFIGS = {
    # Math datasets
    "meta-math/MetaMathQA": {
        "huggingface_name": "meta-math/MetaMathQA",
        "input_field": "query",
        "output_field": "response",
        "prompt_template": MATH_PROMPT,
        "description": "MetaMathQA: Mathematical reasoning dataset"
    },

    # Code datasets
    "theblackcat102/evol-codealpaca-v1": {
        "huggingface_name": "theblackcat102/evol-codealpaca-v1",
        "input_field": "instruction",
        "output_field": "output",
        "prompt_template": CODE_PROMPT,
        "description": "Evol-CodeAlpaca: Code generation dataset"
    },
}


def get_dataset_config(dataset_name: str) -> dict:
    """
    Get dataset configuration by name.

    Args:
        dataset_name: Name of the dataset (key in DATASET_CONFIGS)

    Returns:
        Dictionary with dataset configuration

    Raises:
        KeyError: If dataset is not found
    """
    if dataset_name not in DATASET_CONFIGS:
        raise KeyError(
            f"Dataset '{dataset_name}' not found. "
            f"Available datasets: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def list_available_datasets() -> list:
    """Return list of available dataset names."""
    return list(DATASET_CONFIGS.keys())
