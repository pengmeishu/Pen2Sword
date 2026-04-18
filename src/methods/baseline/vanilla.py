"""
Vanilla LoRA baseline.

Standard LoRA fine-tuning without any embedding transfer.

This is a baseline method that fine-tunes the large model directly with LoRA.
No knowledge transfer from small model is performed.

Usage:
    python scripts/train.py --method vanilla --MODEL_PATH ...
"""
import logging

logger = logging.getLogger(__name__)


def apply_vanilla_lora(model, target_modules, unfix_modules):
    """
    Apply standard LoRA to the model without any embedding transfer.

    Args:
        model: The model to apply LoRA to
        target_modules: List of module names to apply LoRA to
        unfix_modules: List of module names to unfreeze

    Returns:
        Model with LoRA applied
    """
    from peft import LoraConfig, get_peft_model

    logger.info("Applying Vanilla LoRA (no embedding transfer)")

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

    logger.info("Vanilla LoRA applied successfully")
    return model
