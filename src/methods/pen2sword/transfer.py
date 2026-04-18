"""
Pen2Sword embedding transfer.

Handles the transfer of embedding representations from small model to large model.
"""
import logging
import torch.nn as nn

from .expansion import pen2sword

logger = logging.getLogger(__name__)


def replace_module(MODEL, name, embed_tokens_module):
    """
    Replace a module in the model by name.

    Args:
        MODEL: The model to modify
        name: Full dot-separated path to the module
        embed_tokens_module: The replacement module
    """
    parent = MODEL
    sub_names = name.split(".")
    for sub_name in sub_names[:-1]:
        parent = getattr(parent, sub_name)
    setattr(parent, sub_names[-1], embed_tokens_module)


def apply_pen2sword_transfer(expertmodel, targetmodel, alpha, emb_module_name="embed_tokens"):
    """
    Apply Pen2Sword embedding transfer from expert model to target model.

    This function:
    1. Extracts embedding matrix from expert (small) model
    2. Extracts embedding matrix from target (large) model
    3. Applies Pen2Sword expansion and fusion
    4. Replaces the target model's embedding layer with the fused embedding

    Args:
        expertmodel: Small model providing embeddings
        targetmodel: Large model to receive embeddings
        alpha: Blending weight
        emb_module_name: Name of the embedding module

    Returns:
        Modified target model with transferred embeddings
    """
    logger.info(f"Applying Pen2Sword transfer with alpha={alpha}")

    # Find embedding parameters
    expert_emb = None
    target_emb = None

    for name, param in expertmodel.named_parameters():
        if emb_module_name in name:
            expert_emb = param
            logger.info(f"Found expert embedding: {name}, shape={expert_emb.shape}")
            break

    for name, param in targetmodel.named_parameters():
        if emb_module_name in name:
            target_emb = param
            logger.info(f"Found target embedding: {name}, shape={target_emb.shape}")
            break

    if expert_emb is None or target_emb is None:
        raise ValueError(f"Could not find embedding module '{emb_module_name}'")

    # Apply Pen2Sword expansion and fusion
    new_matrix = pen2sword(
        org_matrix=expert_emb,
        nxt_matrix=target_emb,
        alpha=alpha
    )

    logger.info(f"Expanded embedding shape: {new_matrix.shape}")

    # Create new embedding layer with transferred weights
    state_dict = {"weight": new_matrix}
    embedding = nn.Embedding(
        state_dict["weight"].shape[0],
        state_dict["weight"].shape[1],
        padding_idx=0
    )
    embedding.load_state_dict(state_dict)

    # Replace embedding module in target model
    for name, module in targetmodel.named_modules():
        if name.endswith(emb_module_name):
            logger.info(f"Replacing module: {name}")
            replace_module(targetmodel, name, embedding)
            break

    return targetmodel
