"""
Weak-to-Strong baseline.

Based on "Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision"
(Burns et al., ICML 2024).
https://arxiv.org/abs/2312.09390

Weak-to-Strong uses predictions from a weak (small) model as labels
to fine-tune a strong (large) model.

Note:
    This method requires pre-computed weak labels before training.
    Use compute_weak_labels() to generate them first.

Usage:
    # First compute weak labels
    python scripts/compute_labels.py --method weak2strong --WEAK_MODEL_PATH ...

    # Then train with weak labels
    python scripts/train.py --method weak2strong --USE_PRECOMPUTED_LABELS ...
"""
import logging
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_weak_labels(weakmodel, data, batchsize=6):
    """
    Compute weak labels using a weak (small) model.

    Args:
        weakmodel: The weak model to generate labels from
        data: Dataset with 'input_ids'
        batchsize: Batch size for inference

    Returns:
        Dataset with 'labels' column added containing weak model predictions
    """
    logger.info("Computing weak labels using Weak-to-Strong method")

    all_weak_labels = []
    input_ids = data['input_ids']

    for i in tqdm(range(0, len(input_ids), batchsize)):
        batch = input_ids[i:i + batchsize]
        batch = torch.tensor(batch)
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)

        weakmodel = weakmodel.cuda()
        with torch.no_grad():
            output = weakmodel(batch.cuda())
            predict = output.logits

        weakmodel = weakmodel.to('cpu')
        torch.cuda.empty_cache()

        # predict shape: [batch_size, seq_len, vocab_size]
        if batch.shape[0] == 1:
            # Single sample case: predict shape [1, seq_len, vocab_size]
            predict = predict.squeeze(0)  # -> [seq_len, vocab_size]
            weak_labels = np.argmax(predict.cpu().detach().numpy(), axis=-1).tolist()
            all_weak_labels.append(weak_labels)  # Append as single element
        else:
            # Batch case: predict shape [batch_size, seq_len, vocab_size]
            weak_labels = np.argmax(predict.cpu().detach().numpy(), axis=-1).tolist()
            all_weak_labels.extend(weak_labels)  # Each element is one sample's labels

    # Rename existing labels to 'gt_labels' and add weak labels as 'labels'
    data = data.rename_column('labels', 'gt_labels')
    data = data.add_column('labels', all_weak_labels)

    return data


def generate_weak2strong_labels(weakmodel, train_data, val_data, save_data=False,
                                 save_dir=None, project_name=None):
    """
    Generate weak labels for both train and validation datasets.

    Args:
        weakmodel: The weak model to generate labels from
        train_data: Training dataset
        val_data: Validation dataset
        save_data: Whether to save the datasets to disk
        save_dir: Directory to save datasets (required if save_data=True)
        project_name: Project name for file naming (required if save_data=True)

    Returns:
        Tuple of (train_data, val_data)
    """
    logger.info("Generating weak-to-strong labels")

    train_data = compute_weak_labels(weakmodel, train_data)
    val_data = compute_weak_labels(weakmodel, val_data)

    if save_data and save_dir and project_name:
        save_weak_labels(train_data, val_data, save_dir, project_name)

    return train_data, val_data


def apply_weak2strong_lora(model, target_modules, unfix_modules):
    """
    Apply LoRA to the model for Weak-to-Strong training.

    Args:
        model: The strong model to apply LoRA to
        target_modules: List of module names to apply LoRA to
        unfix_modules: List of module names to unfreeze (ignored for weak2strong)

    Returns:
        Model with LoRA applied
    """
    from peft import LoraConfig, get_peft_model

    logger.info("Applying LoRA for Weak-to-Strong training")

    # Freeze all model parameters first (PEFT requires this)
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=target_modules
    )
    model = get_peft_model(model, config)

    # NOTE: weak2strong does NOT unfreeze any additional modules
    # Only LoRA parameters are trainable (unlike pen2sword which unfreezes embed_tokens)

    logger.info("Weak-to-Strong LoRA applied successfully")
    return model


def save_weak_labels(train_data, val_data, save_dir, project_name):
    """
    Save weak labels to disk for reuse.

    Args:
        train_data: Training dataset with weak labels
        val_data: Validation dataset with weak labels
        save_dir: Directory to save datasets
        project_name: Project name for file naming
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, f"{project_name}_train_data")
    val_path = os.path.join(save_dir, f"{project_name}_val_data")
    train_data.save_to_disk(train_path)
    val_data.save_to_disk(val_path)
    logger.info(f"Saved weak labels to {save_dir}")


def load_weak_labels(load_dir, project_name):
    """
    Load pre-computed weak labels from disk.

    Args:
        load_dir: Directory containing saved datasets
        project_name: Project name for file naming

    Returns:
        Tuple of (train_data, val_data)
    """
    from datasets import Dataset
    import os

    train_path = os.path.join(load_dir, f"{project_name}_train_data")
    val_path = os.path.join(load_dir, f"{project_name}_val_data")

    train_data = Dataset.load_from_disk(train_path)
    val_data = Dataset.load_from_disk(val_path)

    logger.info(f"Loaded weak labels from {load_dir}")
    return train_data, val_data
