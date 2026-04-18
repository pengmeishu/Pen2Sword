"""
Dataset configurations for Pen2Sword.
"""
from .datasets import (
    DATASET_CONFIGS,
    get_dataset_config,
    list_available_datasets,
    MATH_PROMPT,
    CODE_PROMPT,
)

__all__ = [
    "DATASET_CONFIGS",
    "get_dataset_config",
    "list_available_datasets",
    "MATH_PROMPT",
    "CODE_PROMPT",
]
