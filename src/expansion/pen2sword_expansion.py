"""
Pen2Sword expansion algorithm.

This module implements the Pen2Sword embedding expansion
for transferring knowledge from a small model to a large model.
"""
import logging
import torch

from .random_utils import generate_random_match

# Configure logger
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

    The expansion works by:
    1. Randomly mapping new columns to original columns
    2. Dividing original columns by (count + 1) to preserve information
    3. Filling new columns by copying from mapped original columns
    4. Blending: alpha * expanded + (1 - alpha) * nxt_matrix

    Args:
        org_matrix: Original embedding matrix from small model
        nxt_matrix: Target embedding matrix from large model
        to_expand: Expansion direction, currently only supports "col"
        alpha: Blending weight, controls how much to use expanded vs target

    Returns:
        Expanded embedding matrix with same shape as nxt_matrix

    Raises:
        Exception: If target_col < col (cannot shrink matrix)
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
        # This ensures the sum of information is maintained when we expand
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


def expand_fpi(
    org_matrix: torch.Tensor,
    target_col: int,
    to_expand: str = "col"
) -> torch.Tensor:
    """
    Expand parameter matrix according to Pen2Sword policy.

    This function only uses the original matrix (no blending with target).

    Args:
        org_matrix: Original parameter matrix
        target_col: Target number of columns
        to_expand: Expansion strategy, only "col" supported

    Returns:
        Expanded parameter matrix
    """
    logger.info(f"Performing Pen2Sword expansion {to_expand}")

    flag = 0
    if org_matrix.ndim == 1:
        flag = 1
        org_matrix = org_matrix.view(-1, 1)

    assert org_matrix.ndim == 2, f"org_matrix must be 2D, got {org_matrix.ndim}D"

    row, col = org_matrix.shape
    choose_num_dict = generate_random_match(col, target_col)

    if target_col < col:
        raise Exception("expanded row or col smaller than origin")

    new = torch.zeros((row, target_col), dtype=torch.float32)
    new[:row, :col] = org_matrix[:, :]

    count = dict()
    for choice in choose_num_dict.values():
        count[choice] = count.get(choice, 0) + 1

    if to_expand == "col":
        for to_divide_col in count.keys():
            new[:row, to_divide_col] /= count.get(to_divide_col) + 1

        min_col = min(choose_num_dict.keys())
        for temp_col in range(min_col, target_col):
            choice = choose_num_dict.get(temp_col)
            new[:row, temp_col] = new[:row, choice]

    if flag == 1:
        new = new.view(-1)

    return new
