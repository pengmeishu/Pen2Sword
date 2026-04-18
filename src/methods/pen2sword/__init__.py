"""
Pen2Sword method.

This is our proposed method for efficient fine-tuning via
small-model embedding transfer.

Paper: "The Pen as the Sword: Faster Fine-Tuning via Small-Model Embedding Transfer"

Key idea:
- Transfer embedding representations from a small model to a large model
- Use random expansion to match dimensions
- Blend small model knowledge with large model using alpha weighting

Usage:
    python scripts/train.py --method pen2sword --MODEL_PATH ... --EMBMODEL_PATH ...
"""
from .expansion import pen2sword
from .transfer import apply_pen2sword_transfer

__all__ = ["pen2sword", "apply_pen2sword_transfer"]
