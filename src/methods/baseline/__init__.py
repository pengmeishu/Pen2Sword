"""
Baseline methods.

This directory contains baseline methods for comparison with Pen2Sword.

Available baselines:
- vanilla: Standard LoRA fine-tuning without embedding transfer
- proxy_tuning: Proxy Tuning method (requires pre-computed labels)
- weak2strong: Weak-to-Strong method (requires pre-computed labels)

Each baseline is independent and has its own training script.

Usage:
    python scripts/train_vanilla.py --MODEL_PATH ...
    python scripts/train_proxy_tuning.py ...
    python scripts/train_weak2strong.py ...
"""
from .vanilla import apply_vanilla_lora
from .proxy_tuning import compute_proxy_labels, generate_proxy_labels
from .weak2strong import compute_weak_labels, generate_weak2strong_labels

__all__ = [
    "apply_vanilla_lora",
    "compute_proxy_labels",
    "generate_proxy_labels",
    "compute_weak_labels",
    "generate_weak2strong_labels",
]
