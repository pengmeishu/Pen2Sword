"""
Pen2Sword expansion algorithm.

Core implementation of the Pen2Sword embedding expansion
for transferring knowledge from small model to large model.
"""
import logging
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods.random_utils import generate_random_match

logger = logging.getLogger(__name__)


def pen2sword(
    org_matrix: torch.Tensor,
    nxt_matrix: torch.Tensor,
    to_expand: str = "col",
    alpha: float = 0.25
) -> torch.Tensor:
    """
    Perform Pen2Sword embedding expansion.

    Takes the embedding matrix from a small model (org_matrix) and expands it
    to match the dimensions of a large model (nxt_matrix), then blends them.

    Args:
        org_matrix: Original embedding matrix from small model
        nxt_matrix: Target embedding matrix from large model
        to_expand: Expansion direction, currently only supports "col"
        alpha: Blending weight, controls how much to use expanded vs target

    Returns:
        Expanded embedding matrix with same shape as nxt_matrix
    """
    logger.info(f"Performing Pen2Sword expansion {to_expand}")

    assert org_matrix.ndim == 2, f"org_matrix must be 2D, got {org_matrix.ndim}D"
    row, col = org_matrix.shape
    _, target_col = nxt_matrix.shape

    if target_col < col:
        raise Exception("expanded row or col smaller than origin")

    # Generate random matching: maps new indices to original indices
    choose_num_dict = generate_random_match(col, target_col)

    # Initialize expanded matrix
    new = torch.zeros((row, target_col), dtype=torch.float32)
    new[:row, :col] = org_matrix[:, :]

    # Count how many times each original column is mapped to
    count = dict()
    for choice in choose_num_dict.values():
        count[choice] = count.get(choice, 0) + 1

    if to_expand == "col":
        # Divide original columns by (count + 1) to preserve information
        for to_divide_col in count.keys():
            new[:row, to_divide_col] /= count.get(to_divide_col) + 1

        # Find minimum column index to start expansion
        min_col = min(choose_num_dict.keys())

        # Fill expanded columns by copying from mapped original columns
        for temp_col in range(min_col, target_col):
            choice = choose_num_dict.get(temp_col)
            new[:row, temp_col] = new[:row, choice]

    # Blend original expanded and target matrices
    new_output = torch.zeros((row, target_col), dtype=torch.float32)
    new_output[:, :] = alpha * new[:, :] + (1 - alpha) * nxt_matrix[:, :]

    return new_output
