#!/usr/bin/env python3
"""
Baseline Training Script.

This script supports baseline fine-tuning methods:
- vanilla: Standard LoRA fine-tuning
- weak2strong: Weak-to-Strong method (use weak model predictions as labels)
- proxy_tuning: Proxy Tuning method

Usage:
    python scripts/train_baseline.py --METHOD weak2strong --MODEL_PATH ... --WEAK_MODEL_PATH ...
    python scripts/train_baseline.py --METHOD vanilla --MODEL_PATH ...
"""
import os
import sys
import functools
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import transformers
from datasets import load_dataset
from rouge_score import rouge_scorer
from safetensors.torch import save_file
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import wandb

from src.util import get_prompt_function, get_wandb_key, DEFAULT_TRAINING_CONFIG
from dataset.datasets import list_available_datasets
from src.methods.baseline.vanilla import apply_vanilla_lora
from src.methods.baseline.weak2strong import (
    compute_weak_labels,
    generate_weak2strong_labels,
    apply_weak2strong_lora,
    save_weak_labels,
    load_weak_labels,
)
from src.methods.baseline.proxy_tuning import (
    compute_proxy_labels,
    generate_proxy_labels,
    apply_proxy_tuning_lora,
    save_proxy_labels,
    load_proxy_labels,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Baseline Training (Vanilla/Weak2Strong/Proxy-Tuning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Vanilla LoRA baseline
  python scripts/train_baseline.py --METHOD vanilla --MODEL_PATH /path/to/13b

  # Run Weak-to-Strong baseline
  python scripts/train_baseline.py --METHOD weak2strong --MODEL_PATH /path/to/13b --WEAK_MODEL_PATH /path/to/7b

  # Run Proxy Tuning baseline
  python scripts/train_baseline.py --METHOD proxy_tuning --MODEL_PATH /path/to/13b --WEAK_MODEL_PATH /path/to/7b --TUNED_WEAK_MODEL_PATH /path/to/7b-finetuned

  Available methods: vanilla, weak2strong, proxy_tuning
        """
    )

    # Method selection
    parser.add_argument("--METHOD", type=str, required=True,
                        choices=["vanilla", "weak2strong", "proxy_tuning"],
                        help="Baseline method to use")

    # Model paths
    parser.add_argument("--MODEL_PATH", type=str, required=True,
                        help="Path to the target (strong) model")
    parser.add_argument("--WEAK_MODEL_PATH", type=str, default=None,
                        help="Path to weak model (required for weak2strong/proxy_tuning)")
    parser.add_argument("--TUNED_WEAK_MODEL_PATH", type=str, default=None,
                        help="Path to fine-tuned weak model (required for proxy_tuning)")

    # Label persistence
    parser.add_argument("--SAVE_LABELS", action="store_true",
                        help="Save computed labels to disk")
    parser.add_argument("--LOAD_LABELS", action="store_true",
                        help="Load pre-computed labels from disk")
    parser.add_argument("--LABELS_DIR", type=str,
                        default="./weak2strong_save_dataset",
                        help="Directory to save/load labels")

    # Training parameters
    parser.add_argument("--CUTOFF_LEN", type=int,
                        default=DEFAULT_TRAINING_CONFIG["CUTOFF_LEN"])
    parser.add_argument("--VAL_SET_SIZE", type=int,
                        default=DEFAULT_TRAINING_CONFIG["VAL_SET_SIZE"])
    parser.add_argument("--RANDOM_SEED", type=int,
                        default=DEFAULT_TRAINING_CONFIG["RANDOM_SEED"])
    parser.add_argument("--EVAL_BATCH_SIZE", type=int,
                        default=DEFAULT_TRAINING_CONFIG["EVAL_BATCH_SIZE"])
    parser.add_argument("--GRADIENT_ACCUMULATION_STEPS", type=int,
                        default=DEFAULT_TRAINING_CONFIG["GRADIENT_ACCUMULATION_STEPS"])
    parser.add_argument("--SAVE_STEP", type=int,
                        default=DEFAULT_TRAINING_CONFIG["SAVE_STEP"],
                        help="Save checkpoint every N steps")
    parser.add_argument("--SAVE_CHECKPOINT", action="store_true",
                        help="Save trainable weights periodically for later merging/evaluation")
    parser.add_argument("--DEBUG", action="store_true",
                        help="Enable debug mode (use 1000 samples)")
    parser.add_argument("--DRY_RUN", action="store_true",
                        help="Dry run: validate code path without loading models")

    # Model configuration
    parser.add_argument("--TARGET_MODULES", type=str,
                        default="q_proj,v_proj",
                        help="LoRA target modules (comma-separated)")
    parser.add_argument("--UNFIX_MODULES", type=str,
                        default="embed_tokens",
                        help="Modules to unfreeze (comma-separated)")

    # Experiment configuration
    parser.add_argument("--DATASET_NAME", type=str,
                        default=DEFAULT_TRAINING_CONFIG["DATASET_NAME"],
                        help="Dataset name. Available: " + ", ".join(list_available_datasets()))
    parser.add_argument("--OUTPUT_DIR", type=str,
                        default=DEFAULT_TRAINING_CONFIG["OUTPUT_DIR"])
    parser.add_argument("--PROJECT_NAME", type=str,
                        default="baseline_experiment")

    return parser.parse_args()


def compute_metric(pred, tokenizer):
    """Compute ROUGE metrics for predictions."""
    predictions = pred.predictions
    labels = pred.label_ids

    predictions = np.argmax(predictions, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(decoded_labels, decoded_preds)]

    rouge1 = sum([score['rouge1'].fmeasure for score in scores]) / len(scores)
    rouge2 = sum([score['rouge2'].fmeasure for score in scores]) / len(scores)
    rougeL = sum([score['rougeL'].fmeasure for score in scores]) / len(scores)

    return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}


class SaveWeightsCallback(transformers.TrainerCallback):
    """Callback to save model weights during training."""

    def __init__(self, output_dir, method_name, save_checkpoint, save_step):
        self.output_dir = output_dir
        self.method_name = method_name
        self.save_checkpoint = save_checkpoint
        self.save_step = save_step

    def on_save(self, args, state, control, **kwargs):
        if not self.save_checkpoint:
            return
        step = state.global_step
        if step % self.save_step == 0:
            self._save_weights(
                f"{self.output_dir}/{self.method_name}_step{step}.safetensors",
                **kwargs
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self.save_checkpoint:
            self._save_weights(
                f"{self.output_dir}/{self.method_name}_final.safetensors",
                **kwargs
            )

    def _save_weights(self, path, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return

        fine_tuned_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fine_tuned_weights[name] = param.clone().detach()

        if fine_tuned_weights:
            print(f"[*] Saving trainable weights to {path}")
            save_file(fine_tuned_weights, path)


def print_config(args, method_description):
    """Print training configuration in a nice format."""
    print("=" * 60)
    print("  Baseline Training Configuration")
    print("=" * 60)
    print(f"  Method:       {args.METHOD} ({method_description})")
    print(f"  Dataset:      {args.DATASET_NAME}")
    print(f"  Strong Model: {args.MODEL_PATH}")
    if args.WEAK_MODEL_PATH:
        print(f"  Weak Model:   {args.WEAK_MODEL_PATH}")
    if args.TUNED_WEAK_MODEL_PATH:
        print(f"  Tuned Weak:   {args.TUNED_WEAK_MODEL_PATH}")
    print("-" * 60)
    print(f"  Cutoff Len:  {args.CUTOFF_LEN}")
    print(f"  Val Set:     {args.VAL_SET_SIZE}")
    print(f"  Debug:       {'Yes' if args.DEBUG else 'No'}")
    print("-" * 60)
    print(f"  Save Checkpoint: {'Yes' if args.SAVE_CHECKPOINT else 'No'}")
    if args.SAVE_CHECKPOINT:
        print(f"  Save Every:    {args.SAVE_STEP} steps")
    print(f"  Output Dir:  {os.path.join(args.OUTPUT_DIR.rstrip('/'), args.PROJECT_NAME)}")
    print(f"  Project:     {args.PROJECT_NAME}")
    print("=" * 60)
    print()


def main(args):
    """Main training function."""
    method_descriptions = {
        "vanilla": "Standard LoRA fine-tuning",
        "weak2strong": "Weak-to-Strong: Use weak model predictions as labels",
        "proxy_tuning": "Proxy Tuning: Use proxy labels from weak models",
    }
    method_description = method_descriptions.get(args.METHOD, "Unknown")

    logger.info(f"Selected method: {args.METHOD}")
    logger.info(f"Description: {method_description}")

    # Validate method requirements
    if args.METHOD == "weak2strong" and args.WEAK_MODEL_PATH is None:
        raise ValueError("--METHOD weak2strong requires --WEAK_MODEL_PATH")
    if args.METHOD == "proxy_tuning":
        if args.WEAK_MODEL_PATH is None:
            raise ValueError("--METHOD proxy_tuning requires --WEAK_MODEL_PATH")
        if args.TUNED_WEAK_MODEL_PATH is None:
            raise ValueError("--METHOD proxy_tuning requires --TUNED_WEAK_MODEL_PATH")

    # Print configuration
    print_config(args, method_description)

    # Dry run: validate code path without loading models
    if args.DRY_RUN:
        print("[*] DRY RUN - validating code path only")
        print(f"    Method: {args.METHOD}")
        print(f"    Strong model would be loaded from: {args.MODEL_PATH}")
        if args.WEAK_MODEL_PATH:
            print(f"    Weak model would be loaded from: {args.WEAK_MODEL_PATH}")
        if args.TUNED_WEAK_MODEL_PATH:
            print(f"    Tuned weak model would be loaded from: {args.TUNED_WEAK_MODEL_PATH}")
        print(f"    Dataset: {args.DATASET_NAME}")
        print(f"    Debug mode: {args.DEBUG}")
        print("[+] DRY RUN completed successfully")
        return

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    # Login to wandb
    wandb_key = get_wandb_key()
    wandb.login(key=wandb_key)

    # Ensure output directory exists
    output_dir = os.path.join(args.OUTPUT_DIR.rstrip("/"), args.PROJECT_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set random seed
    transformers.set_seed(args.RANDOM_SEED)

    # Parse module lists
    unfix_modules = args.UNFIX_MODULES.split(",")
    target_modules = args.TARGET_MODULES.split(",")

    # Load models based on method
    print(f"[*] Loading strong model from {args.MODEL_PATH} ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.MODEL_PATH,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print(f"[+] Strong model loaded successfully")

    # Load tokenizer
    print(f"[*] Loading tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.MODEL_PATH,
        add_eos_token=True,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = 0
    print(f"[+] Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Load dataset
    print(f"[*] Loading dataset: {args.DATASET_NAME} ...")
    data = load_dataset(args.DATASET_NAME)
    print(f"[+] Dataset loaded ({len(data['train'])} samples)")

    if args.DEBUG:
        data['train'] = data['train'].shuffle(seed=args.RANDOM_SEED).select(range(1000))
        eval_step = 50
    else:
        data['train'] = data['train'].shuffle(seed=args.RANDOM_SEED)
        eval_step = 100

    # Tokenize dataset
    generate_and_tokenize_prompt = functools.partial(
        get_prompt_function(args.DATASET_NAME),
        tokenizer=tokenizer,
        CUTOFF_LEN=args.CUTOFF_LEN
    )

    if args.VAL_SET_SIZE > 0:
        train_val = data['train'].train_test_split(
            test_size=args.VAL_SET_SIZE,
            shuffle=False,
            seed=args.RANDOM_SEED
        )
        train_data = train_val['train'].map(generate_and_tokenize_prompt)
        val_data = train_val['test'].map(generate_and_tokenize_prompt)
    else:
        train_data = data['train'].map(generate_and_tokenize_prompt)
        val_data = None

    # Load pre-computed labels if requested
    if args.LOAD_LABELS:
        print(f"[*] Loading pre-computed labels from {args.LABELS_DIR} ...")
        if args.METHOD == "weak2strong":
            train_data, val_data = load_weak_labels(args.LABELS_DIR, args.PROJECT_NAME)
        elif args.METHOD == "proxy_tuning":
            train_data, val_data = load_proxy_labels(args.LABELS_DIR, args.PROJECT_NAME)
        print(f"[+] Labels loaded successfully")
    elif args.METHOD == "weak2strong":
        # Load weak model and compute weak labels
        print(f"[*] Loading weak model from {args.WEAK_MODEL_PATH} ...")
        weakmodel = transformers.AutoModelForCausalLM.from_pretrained(
            args.WEAK_MODEL_PATH,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        weakmodel.resize_token_embeddings(len(tokenizer))
        print(f"[+] Weak model loaded successfully")

        print(f"[*] Computing weak labels ...")
        train_data, val_data = generate_weak2strong_labels(
            weakmodel, train_data, val_data,
            save_data=args.SAVE_LABELS, save_dir=args.LABELS_DIR, project_name=args.PROJECT_NAME
        )

        del weakmodel
        torch.cuda.empty_cache()
        print(f"[+] Weak labels computed successfully")

    elif args.METHOD == "proxy_tuning":
        # Load both weak models and compute proxy labels
        print(f"[*] Loading weak models ...")
        untuneweakmodel = transformers.AutoModelForCausalLM.from_pretrained(
            args.WEAK_MODEL_PATH,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        untuneweakmodel.resize_token_embeddings(len(tokenizer))

        tuneweakmodel = transformers.AutoModelForCausalLM.from_pretrained(
            args.TUNED_WEAK_MODEL_PATH,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        tuneweakmodel.resize_token_embeddings(len(tokenizer))
        print(f"[+] Weak models loaded successfully")

        print(f"[*] Computing proxy labels ...")
        train_time, val_time, train_data, val_data = generate_proxy_labels(
            tuneweakmodel, untuneweakmodel, model, train_data, val_data,
            save_data=args.SAVE_LABELS, save_dir=args.LABELS_DIR, project_name=args.PROJECT_NAME
        )
        print(f"[+] Proxy labels computed (train: {train_time:.2f}h, val: {val_time:.2f}h)")

        del untuneweakmodel, tuneweakmodel
        torch.cuda.empty_cache()

    # Apply LoRA based on method
    if args.METHOD == "vanilla":
        print(f"[*] Applying Vanilla LoRA ...")
        model = apply_vanilla_lora(model, target_modules, unfix_modules)
        print(f"[+] Vanilla LoRA applied")
    elif args.METHOD == "weak2strong":
        print(f"[*] Applying Weak-to-Strong LoRA ...")
        model = apply_weak2strong_lora(model, target_modules, unfix_modules)
        print(f"[+] Weak-to-Strong LoRA applied")
    elif args.METHOD == "proxy_tuning":
        print(f"[*] Applying Proxy Tuning LoRA ...")
        model = apply_proxy_tuning_lora(model, target_modules, unfix_modules)
        print(f"[+] Proxy Tuning LoRA applied")

    # Disable generation config temperature/top_p if present
    if hasattr(model, 'generation_config'):
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    # Compute training statistics
    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    training_ratio = trainable_param / total_param
    print(f"Total parameters: {total_param:,}")
    print(f"Trainable parameters: {trainable_param:,}")
    print(f"Training ratio: {training_ratio:.4f}")
    print(f"\nTrainable parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} params, shape={param.shape}")

    # Define compute_metrics function
    def compute_metrics(pred):
        return compute_metric(pred, tokenizer)

    # Setup training arguments
    train_args = transformers.TrainingArguments(
        report_to="wandb",
        seed=42,
        data_seed=42,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=args.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=args.GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps=1,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        eval_strategy='steps' if args.VAL_SET_SIZE > 0 else 'no',
        save_strategy='steps',
        save_steps=args.SAVE_STEP,
        eval_steps=eval_step if args.VAL_SET_SIZE > 0 else None,
        output_dir=output_dir,
        save_total_limit=2,
        metric_for_best_model="rouge1",
        load_best_model_at_end=True if args.VAL_SET_SIZE > 0 else False,
    )

    # Initialize wandb
    wandb_config = {
        "method": args.METHOD,
        "total_param": total_param,
        "trainable_param": trainable_param,
        "training_ratio": training_ratio,
    }
    run = wandb.init(
        project=args.PROJECT_NAME,
        notes=f"{args.METHOD} on {args.DATASET_NAME}",
        config=dict(train_args.to_dict(), **wandb_config)
    )

    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=train_args,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
        compute_metrics=compute_metrics,
    )

    model.config.use_cache = False

    # Fix state_dict for LoRA
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # Add callback and train
    trainer.add_callback(
        SaveWeightsCallback(
            output_dir,
            args.METHOD,
            args.SAVE_CHECKPOINT,
            args.SAVE_STEP
        )
    )

    print(f"\n[*] Starting training ...")
    print(f"    Train samples: {len(train_data)}")
    if val_data:
        print(f"    Val samples:   {len(val_data)}")
    print(f"    Total params:  {total_param:,}")
    print(f"    Trainable:     {trainable_param:,} ({training_ratio*100:.2f}%)\n")

    trainer.train()
    wandb.finish()

    print(f"\n[+] Training completed!")
    print(f"    Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
