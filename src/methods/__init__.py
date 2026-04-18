"""
Method registry for Pen2Sword project.

This module provides a unified interface for different fine-tuning methods.
"""
from .pen2sword import apply_pen2sword_transfer
from .baseline.vanilla import apply_vanilla_lora
from .baseline.proxy_tuning import (
    compute_proxy_labels,
    generate_proxy_labels,
    apply_proxy_tuning_lora,
)
from .baseline.weak2strong import (
    compute_weak_labels,
    generate_weak2strong_labels,
    apply_weak2strong_lora,
    save_weak_labels,
    load_weak_labels,
)


METHODS = {
    "pen2sword": {
        "description": "Pen2Sword: Transfer embedding from small model to large model",
        "needs_emb_model": True,
        "class": "ours",
    },
    "vanilla": {
        "description": "Vanilla LoRA: Standard LoRA fine-tuning",
        "needs_emb_model": False,
        "class": "baseline",
    },
    "proxy_tuning": {
        "description": "Proxy Tuning: Use proxy labels from weak models",
        "needs_emb_model": False,
        "class": "baseline",
    },
    "weak2strong": {
        "description": "Weak-to-Strong: Use weak model predictions as labels",
        "needs_emb_model": False,
        "class": "baseline",
    },
}


def get_method(method_name):
    """
    Get method information by name.

    Args:
        method_name: Name of the method

    Returns:
        Dictionary with method information

    Raises:
        KeyError: If method is not found
    """
    if method_name not in METHODS:
        raise KeyError(
            f"Method '{method_name}' not found. Available methods: {list(METHODS.keys())}"
        )
    return METHODS[method_name]


__all__ = [
    "apply_pen2sword_transfer",
    "apply_vanilla_lora",
    "compute_proxy_labels",
    "generate_proxy_labels",
    "apply_proxy_tuning_lora",
    "compute_weak_labels",
    "generate_weak2strong_labels",
    "apply_weak2strong_lora",
    "save_weak_labels",
    "load_weak_labels",
    "METHODS",
    "get_method",
]
