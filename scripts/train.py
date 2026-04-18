#!/usr/bin/env python3
"""
Pen2Sword unified training script.

This script supports multiple fine-tuning methods:
- pen2sword: Our proposed method (embedding transfer from small to large model)
- vanilla: Standard LoRA fine-tuning (baseline)
- proxy_tuning: Proxy Tuning method (baseline)
- weak2strong: Weak-to-Strong method (baseline)

Usage:
    python scripts/train.py --METHOD pen2sword --MODEL_PATH ... --EMBMODEL_PATH ...
    python scripts/train.py --METHOD vanilla --MODEL_PATH ...
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
from src.methods import METHODS, get_method
from src.methods.pen2sword import apply_pen2sword_transfer
from src.methods.baseline.vanilla import apply_vanilla_lora

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pen2Sword Unified Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Pen2Sword (our method)
  python scripts/train.py --METHOD pen2sword --MODEL_PATH /path/to/13b --EMBMODEL_PATH /path/to/7b

  # Run Vanilla LoRA baseline
  python scripts/train.py --METHOD vanilla --MODEL_PATH /path/to/13b

  Available methods: {methods}
        """.format(methods=list(METHODS.keys()))
    )

    # Method selection
    parser.add_argument("--METHOD", type=str, required=True,
                        choices=list(METHODS.keys()),
                        help="Fine-tuning method to use")

    # Model paths
    parser.add_argument("--MODEL_PATH", type=str, required=True,
                        help="Path to the target (large) model")
    parser.add_argument("--EMBMODEL_PATH", type=str, default=None,
                        help="Path to the embedding (small/expert) model. "
                             "Required for pen2sword method.")

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
                        help="Save checkpoint every N steps (only if --SAVE_CHECKPOINT is set)")
    parser.add_argument("--SAVE_CHECKPOINT", action="store_true",
                        help="Save trainable weights periodically for later merging/evaluation")
    parser.add_argument("--ALPHA", type=float,
                        default=DEFAULT_TRAINING_CONFIG["ALPHA"],
                        help="Blending weight for Pen2Sword embedding fusion")
    parser.add_argument("--DEBUG", action="store_true",
                        help="Enable debug mode (use 1000 samples)")
    parser.add_argument("--LORA", action="store_true",
                        help="Use LoRA for fine-tuning")

    # Model configuration
    parser.add_argument("--EMB_MODULE_NAME", type=str,
                        default=DEFAULT_TRAINING_CONFIG["EMB_MODULE_NAME"])
    parser.add_argument("--UNFIX_MODULES", type=str,
                        default=DEFAULT_TRAINING_CONFIG["UNFIX_MODULES"])
    parser.add_argument("--TARGETMODUOLES", type=str,
                        default=DEFAULT_TRAINING_CONFIG["TARGETMODUOLES"])

    # Experiment configuration
    parser.add_argument("--DATASET_NAME", type=str,
                        default=DEFAULT_TRAINING_CONFIG["DATASET_NAME"],
                        help=f"Dataset name. Available: {list_available_datasets()}. "
                             "Or add custom dataset to dataset/datasets.py")
    parser.add_argument("--OUTPUT_DIR", type=str,
                        default=DEFAULT_TRAINING_CONFIG["OUTPUT_DIR"])
    parser.add_argument("--PROJECT_NAME", type=str,
                        default=DEFAULT_TRAINING_CONFIG["PROJECT_NAME"])

    return parser.parse_args()


def replace_module(MODEL, name, embed_tokens_module):
    """Replace a module in the model by name."""
    parent = MODEL
    sub_names = name.split(".")
    for sub_name in sub_names[:-1]:
        parent = getattr(parent, sub_name)
    setattr(parent, sub_names[-1], embed_tokens_module)


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

    def __init__(self, output_dir, unfix_modules, use_lora, method_name, save_checkpoint, save_step):
        self.output_dir = output_dir
        self.unfix_modules = unfix_modules
        self.use_lora = use_lora
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
            if not self.use_lora:
                if any(target in name for target in self.unfix_modules):
                    fine_tuned_weights[name] = param.clone().detach()
            else:
                if param.requires_grad:
                    fine_tuned_weights[name] = param.clone().detach()

        if fine_tuned_weights:
            print(f"[*] Saving trainable weights to {path}")
            save_file(fine_tuned_weights, path)


def print_config(args, method_info):
    """Print training configuration in a nice format."""
    print("=" * 60)
    print("  Pen2Sword Training Configuration")
    print("=" * 60)
    print(f"  Method:       {args.METHOD} ({method_info['description']})")
    print(f"  Dataset:      {args.DATASET_NAME}")
    print(f"  Large Model: {args.MODEL_PATH}")
    if args.EMBMODEL_PATH:
        print(f"  Small Model: {args.EMBMODEL_PATH}")
    print("-" * 60)
    print(f"  LoRA:        Yes")
    print(f"  Alpha:       {args.ALPHA}")
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
    # Validate method requirements
    method_info = get_method(args.METHOD)
    logger.info(f"Selected method: {args.METHOD}")
    logger.info(f"Description: {method_info['description']}")

    # Print configuration
    print_config(args, method_info)

    if method_info['needs_emb_model'] and args.EMBMODEL_PATH is None:
        raise ValueError(
            f"Method '{args.METHOD}' requires --EMBMODEL_PATH to be specified. "
            f"This method needs a small model for embedding transfer."
        )

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
    target_modules = args.TARGETMODUOLES.split(",")

    # Load models based on method
    print(f"[*] Loading target model from {args.MODEL_PATH} ...")
    targetmodel = transformers.AutoModelForCausalLM.from_pretrained(
        args.MODEL_PATH,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print(f"[+] Target model loaded successfully")

    expertmodel = None
    if args.METHOD == "pen2sword":
        # Load expert model for embedding transfer
        print(f"[*] Loading embedding model from {args.EMBMODEL_PATH} ...")
        expertmodel = transformers.AutoModelForCausalLM.from_pretrained(
            args.EMBMODEL_PATH,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print(f"[+] Embedding model loaded successfully")

    # Load tokenizer (use target model's tokenizer)
    print(f"[*] Loading tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.MODEL_PATH,
        add_eos_token=True,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = 0
    print(f"[+] Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Resize token embeddings
    targetmodel.resize_token_embeddings(len(tokenizer))
    if expertmodel is not None:
        expertmodel.resize_token_embeddings(len(tokenizer))

    # Apply method-specific initialization
    if args.METHOD == "pen2sword":
        # Apply Pen2Sword embedding transfer
        print(f"[*] Applying Pen2Sword embedding transfer (alpha={args.ALPHA}) ...")
        model = apply_pen2sword_transfer(
            expertmodel,
            targetmodel,
            alpha=args.ALPHA,
            emb_module_name=args.EMB_MODULE_NAME
        )
        print(f"[+] Pen2Sword embedding transfer applied")
        del expertmodel
    elif args.METHOD == "vanilla":
        # Apply Vanilla LoRA
        print(f"[*] Applying Vanilla LoRA ...")
        model = apply_vanilla_lora(targetmodel, target_modules, unfix_modules)
        print(f"[+] Vanilla LoRA applied")
    else:
        raise ValueError(f"Method '{args.METHOD}' not implemented yet")

    # Apply LoRA (Pen2Sword always uses LoRA, Vanilla uses LoRA when --LORA is specified)
    if args.LORA or args.METHOD == "pen2sword":
        from peft import LoraConfig, get_peft_model
        print(f"[*] Applying LoRA ...")
        # Freeze all base model params first
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
        print(f"[+] LoRA applied")

    # Disable generation config temperature/top_p if present
    if hasattr(model, 'generation_config'):
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    # Load dataset
    print(f"[*] Loading dataset: {args.DATASET_NAME} ...")
    data = load_dataset(args.DATASET_NAME)
    print(f"[+] Dataset loaded ({len(data['train'])} samples)")

    if args.DEBUG:
        data['train'] = data['train'].shuffle(seed=args.RANDOM_SEED).select(range(1000))
        eval_step = 10
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
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-5,
        fp16=True,
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
        "alpha": args.ALPHA if args.METHOD == "pen2sword" else None,
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
    if args.LORA:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    # Add callback and train
    trainer.add_callback(
        SaveWeightsCallback(
            output_dir, unfix_modules,
            args.LORA or args.METHOD == "pen2sword",
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
