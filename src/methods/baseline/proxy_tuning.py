"""
Proxy Tuning baseline.

Based on "Tuning Language Models by Proxy" (Liu et al., 2024).
https://arxiv.org/abs/2401.08565

Proxy Tuning uses predictions from a fine-tuned weak model minus
predictions from an unfine-tuned weak model, plus predictions from
the strong model to generate proxy labels.

Note:
    This method requires pre-computed proxy labels before training.
    Use compute_proxy_labels() to generate them first.

Usage:
    # First compute proxy labels
    python scripts/compute_labels.py --method proxy_tuning --MODEL_PATH ...

    # Then train with proxy labels
    python scripts/train.py --method proxy_tuning --USE_PRECOMPUTED_LABELS ...
"""
import logging
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_proxy_labels(tuneweakmodel, untuneweakmodel, model, data, batchsize=6):
    """
    Compute proxy labels using Proxy Tuning method.

    Formula: proxy_label = tuneweak_output - untuneweak_output + strong_output

    Args:
        tuneweakmodel: Fine-tuned weak model
        untuneweakmodel: Unfine-tuned weak model
        model: Strong model
        data: Dataset with 'input_ids'
        batchsize: Batch size for inference

    Returns:
        Dataset with 'labels' column added containing proxy labels
    """
    from datasets import Dataset

    logger.info("Computing proxy labels using Proxy Tuning method")

    all_proxy_labels = []
    input_ids = data['input_ids']

    for i in tqdm(range(0, len(input_ids), batchsize)):
        batch = input_ids[i:i + batchsize]
        batch = torch.tensor(batch)
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)

        with torch.no_grad():
            # Get weak model predictions (fine-tuned)
            tuneweakmodel = tuneweakmodel.cuda()
            tuneoutput = tuneweakmodel(batch.cuda())
            tunepredict = tuneoutput.logits
            tuneweakmodel = tuneweakmodel.to('cpu')
            torch.cuda.empty_cache()

            # Get weak model predictions (unfine-tuned)
            untuneweakmodel = untuneweakmodel.cuda()
            untuneoutput = untuneweakmodel(batch.cuda())
            untunepredict = untuneoutput.logits
            untuneweakmodel = untuneweakmodel.to('cpu')
            torch.cuda.empty_cache()

            # Get strong model predictions
            model = model.cuda()
            strongoutput = model(batch.cuda())
            strongoutput = strongoutput.logits
            model = model.to('cpu')
            torch.cuda.empty_cache()

            # Compute proxy labels
            proxy_predict = tunepredict - untunepredict + strongoutput

            # proxy_predict shape: [batch_size, seq_len, vocab_size]
            if batch.shape[0] == 1:
                # Single sample case
                proxy_predict = proxy_predict.squeeze(0)  # -> [seq_len, vocab_size]
                proxy_labels = np.argmax(proxy_predict.cpu().detach().numpy(), axis=-1).tolist()
                all_proxy_labels.append(proxy_labels)
            else:
                # Batch case
                proxy_labels = np.argmax(proxy_predict.cpu().detach().numpy(), axis=-1).tolist()
                all_proxy_labels.extend(proxy_labels)

    # Rename existing labels to 'gt_labels' and add proxy labels as 'labels'
    data = data.rename_column('labels', 'gt_labels')
    data = data.add_column('labels', all_proxy_labels)

    return data


def save_proxy_labels(train_data, val_data, save_dir, project_name):
    """
    Save proxy labels to disk for reuse.

    Args:
        train_data: Training dataset with proxy labels
        val_data: Validation dataset with proxy labels
        save_dir: Directory to save datasets
        project_name: Project name for file naming
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, f"{project_name}_train_data")
    val_path = os.path.join(save_dir, f"{project_name}_val_data")
    train_data.save_to_disk(train_path)
    val_data.save_to_disk(val_path)
    logger.info(f"Saved proxy labels to {save_dir}")


def load_proxy_labels(load_dir, project_name):
    """
    Load pre-computed proxy labels from disk.

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

    logger.info(f"Loaded proxy labels from {load_dir}")
    return train_data, val_data


def generate_proxy_labels(tuneweakmodel, untuneweakmodel, model, train_data, val_data,
                          save_data=False, save_dir=None, project_name=None):
    """
    Generate proxy labels for both train and validation datasets.

    Args:
        tuneweakmodel: Fine-tuned weak model
        untuneweakmodel: Unfine-tuned weak model
        model: Strong model
        train_data: Training dataset
        val_data: Validation dataset
        save_data: Whether to save the datasets to disk
        save_dir: Directory to save datasets (required if save_data=True)
        project_name: Project name for file naming (required if save_data=True)

    Returns:
        Tuple of (train_elapsed_hours, val_elapsed_hours, train_data, val_data)
    """
    import time

    start_time = time.time()
    train_data = compute_proxy_labels(tuneweakmodel, untuneweakmodel, model, train_data, batchsize=6)
    train_elapsed_time_hours = (time.time() - start_time) / 3600

    start_time = time.time()
    val_data = compute_proxy_labels(tuneweakmodel, untuneweakmodel, model, val_data, batchsize=6)
    val_elapsed_time_hours = (time.time() - start_time) / 3600

    if save_data and save_dir and project_name:
        save_proxy_labels(train_data, val_data, save_dir, project_name)

    return train_elapsed_time_hours, val_elapsed_time_hours, train_data, val_data


def apply_proxy_tuning_lora(model, target_modules, unfix_modules):
    """
    Apply LoRA to the model for Proxy Tuning.

    Args:
        model: The strong model to apply LoRA to
        target_modules: List of module names to apply LoRA to
        unfix_modules: List of module names to unfreeze

    Returns:
        Model with LoRA applied
    """
    from peft import LoraConfig, get_peft_model

    logger.info("Applying LoRA for Proxy Tuning")

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

    # Unfreeze specified modules
    for name, param in model.named_parameters():
        if any(target in name for target in unfix_modules):
            param.requires_grad = True

    logger.info("Proxy Tuning LoRA applied successfully")
    return model
