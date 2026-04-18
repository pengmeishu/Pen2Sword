"""
Dataset prompt generation utilities.

This module provides functions to generate and tokenize prompts
for different datasets based on dataset configuration.

The recommended way to add new datasets is to edit:
    dataset/datasets.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset.datasets import get_dataset_config


def generate_and_tokenize_prompt(data_point, tokenizer, CUTOFF_LEN, dataset_name):
    """
    Generate and tokenize prompt for any dataset.

    This function reads the dataset configuration and applies the
    appropriate prompt template and field mappings.

    Args:
        data_point: Data sample from the dataset
        tokenizer: Tokenizer to use
        CUTOFF_LEN: Maximum sequence length
        dataset_name: Name of the dataset (for config lookup)

    Returns:
        Dictionary with 'input_ids', 'labels', and 'attention_mask'
    """
    # Get dataset configuration
    config = get_dataset_config(dataset_name)

    input_field = config["input_field"]
    output_field = config["output_field"]
    prompt_template = config["prompt_template"]

    # Get input and output from data point
    input_text = data_point[input_field]
    output_text = data_point[output_field]

    # Format prompt
    user_prompt = prompt_template.format(input=input_text)

    # Tokenize to find length of prompt (excluding the last token)
    prompt_tokens = tokenizer(
        user_prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        add_special_tokens=False
    )['input_ids']
    len_user_prompt_tokens = len(prompt_tokens) - 1  # Match reference code: -1

    # Tokenize full prompt + response, remove last token to maintain causal LM alignment
    full_tokens = tokenizer(
        user_prompt + output_text,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding='max_length',
        add_special_tokens=False
    )['input_ids'][:-1]  # Remove last token (matches reference code)

    # Labels: -100 for prompt tokens, actual tokens for completion
    labels = [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:]

    # Truncate to CUTOFF_LEN
    input_ids = full_tokens[:CUTOFF_LEN]
    labels = labels[:CUTOFF_LEN]

    # Ensure they have the same length
    if len(input_ids) < CUTOFF_LEN:
        input_ids = input_ids + [tokenizer.pad_token_id] * (CUTOFF_LEN - len(input_ids))
        labels = labels + [-100] * (CUTOFF_LEN - len(labels))

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': [1] * len(input_ids),
    }


def debug_generate_and_tokenize_prompt(data_point, tokenizer, CUTOFF_LEN, dataset_name):
    """
    Debug version of generate_and_tokenize_prompt - prints info for first sample.
    """
    config = get_dataset_config(dataset_name)
    input_field = config["input_field"]
    output_field = config["output_field"]
    prompt_template = config["prompt_template"]

    input_text = data_point[input_field]
    output_text = data_point[output_field]

    user_prompt = prompt_template.format(input=input_text)

    prompt_tokens = tokenizer(
        user_prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        add_special_tokens=False
    )['input_ids']
    len_user_prompt_tokens = len(prompt_tokens) - 1

    full_tokens = tokenizer(
        user_prompt + output_text,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding='max_length',
        add_special_tokens=False
    )['input_ids'][:-1]

    labels = [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:]

    input_ids = full_tokens[:CUTOFF_LEN]
    labels = labels[:CUTOFF_LEN]

    if len(input_ids) < CUTOFF_LEN:
        input_ids = input_ids + [tokenizer.pad_token_id] * (CUTOFF_LEN - len(input_ids))
        labels = labels + [-100] * (CUTOFF_LEN - len(labels))

    # Debug print
    nonIgnore_labels = [l for l in labels if l != -100]
    print(f"[DEBUG] prompt_len={len_user_prompt_tokens}, full_tokens_len={len(full_tokens)}, "
          f"input_ids_len={len(input_ids)}, labels_len={len(labels)}, "
          f"non-ignore labels count={len(nonIgnore_labels)}")
    if nonIgnore_labels:
        print(f"[DEBUG] non-ignore label range: min={min(nonIgnore_labels)}, max={max(nonIgnore_labels)}, sum={sum(nonIgnore_labels)}")
    else:
        print(f"[DEBUG] WARNING: ALL labels are -100 (ignore)!")
    print(f"[DEBUG] input_ids[:10]={input_ids[:10]}, labels[:10]={labels[:10]}")
    if nonIgnore_labels:
        print(f"[DEBUG] non-ignore labels (first 20)={nonIgnore_labels[:20]}")

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': [1] * len(input_ids),
    }


def get_prompt_function(dataset_name: str):
    """
    Get a partial function for a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Partial function that takes (data_point, tokenizer, CUTOFF_LEN)
    """
    import functools
    return functools.partial(
        generate_and_tokenize_prompt,
        dataset_name=dataset_name
    )


# Legacy functions for backward compatibility
# These are kept for reference but generate_and_tokenize_prompt is preferred

def code_generate_and_tokenize_prompt(data_point, tokenizer, CUTOFF_LEN):
    """Legacy: Use generate_and_tokenize_prompt with dataset_name instead."""
    return generate_and_tokenize_prompt(
        data_point, tokenizer, CUTOFF_LEN,
        "theblackcat102/evol-codealpaca-v1"
    )


def math_generate_and_tokenize_prompt(data_point, tokenizer, CUTOFF_LEN):
    """Legacy: Use generate_and_tokenize_prompt with dataset_name instead."""
    return generate_and_tokenize_prompt(
        data_point, tokenizer, CUTOFF_LEN,
        "meta-math/MetaMathQA"
    )
