#!/usr/bin/env python3
"""
Merge LoRA weights with base model for evaluation.

Usage:
    python scripts/merge_and_eval.py \
        --CHECKPOINT_PATH ./model_save/YOUR_PROJECT/checkpoint.safetensors \
        --MODEL_PATH /path/to/base-model \
        --OUTPUT_DIR ./model_save/benchmark/YOUR_MODEL_NAME
"""
import os
import sys
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers
import torch
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
import yaml

from src.util import DEFAULT_TRAINING_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")

    parser.add_argument(
        "--CHECKPOINT_PATH", type=str, required=True,
        help="Path to the .safetensors checkpoint file"
    )
    parser.add_argument(
        "--MODEL_PATH", type=str, required=True,
        help="Path to the base model"
    )
    parser.add_argument(
        "--OUTPUT_DIR", type=str, required=True,
        help="Directory to save the merged model"
    )
    parser.add_argument(
        "--TARGET_MODULES", type=str,
        default=DEFAULT_TRAINING_CONFIG["TARGETMODUOLES"],
        help="Comma-separated list of target modules for LoRA"
    )
    parser.add_argument(
        "--MODEL_NAME", type=str, default=None,
        help="Model name for HELM config (default: basename of OUTPUT_DIR)"
    )
    parser.add_argument(
        "--TOKENIZER_NAME", type=str,
        default="meta-llama/Llama-2-13b-hf",
        help="Tokenizer name for HELM config"
    )
    parser.add_argument(
        "--NUM_PARAMETERS", type=int, default=13000000000,
        help="Number of parameters in the model"
    )
    parser.add_argument(
        "--HELM_DIR", type=str, default=None,
        help="HELM directory to append model configs (optional)"
    )

    return parser.parse_args()


def main(args):
    """Main function."""
    # Validate checkpoint exists
    if not os.path.exists(args.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {args.CHECKPOINT_PATH}")

    model_name = args.MODEL_NAME or os.path.basename(args.OUTPUT_DIR.rstrip("/"))

    print("=" * 60)
    print("  Pen2Sword - Merge Model")
    print("=" * 60)
    print(f"  Checkpoint: {args.CHECKPOINT_PATH}")
    print(f"  Base Model: {args.MODEL_PATH}")
    print(f"  Output:     {args.OUTPUT_DIR}")
    print("=" * 60)

    # ============================================================
    # Step 1: Load base model and tokenizer
    # ============================================================
    print(f"\n[*] Loading base model from {args.MODEL_PATH} ...")
    targetmodel = transformers.AutoModelForCausalLM.from_pretrained(
        args.MODEL_PATH,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print(f"[+] Base model loaded")

    print(f"[*] Loading tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.MODEL_PATH,
        add_eos_token=True,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = 0
    targetmodel.resize_token_embeddings(len(tokenizer))
    print(f"[+] Tokenizer loaded (vocab size: {len(tokenizer)})")

    # ============================================================
    # Step 2: Load trainable weights
    # ============================================================
    print(f"[*] Loading trainable weights from {args.CHECKPOINT_PATH} ...")
    weights = load_file(args.CHECKPOINT_PATH)
    print(f"[+] Loaded {len(weights)} weight tensors")

    # ============================================================
    # Step 3: Apply LoRA and load weights
    # ============================================================
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=args.TARGET_MODULES.split(",")
    )
    print(f"[*] Applying LoRA config ...")
    targetmodel = get_peft_model(targetmodel, config)
    print(f"[*] Loading trainable weights into model ...")
    targetmodel.load_state_dict(weights, strict=False)
    print(f"[+] Weights loaded")

    # ============================================================
    # Step 4: Merge and unload
    # ============================================================
    print(f"[*] Merging LoRA weights into base model ...")
    targetmodel = targetmodel.merge_and_unload()
    print(f"[+] Weights merged successfully")

    # ============================================================
    # Step 5: Save merged model
    # ============================================================
    if os.path.exists(args.OUTPUT_DIR):
        print(f"[*] Removing existing directory: {args.OUTPUT_DIR}")
        shutil.rmtree(args.OUTPUT_DIR)

    print(f"[*] Creating output directory: {args.OUTPUT_DIR}")
    os.makedirs(args.OUTPUT_DIR)

    print(f"[*] Saving merged model to {args.OUTPUT_DIR} ...")
    targetmodel.save_pretrained(args.OUTPUT_DIR)
    tokenizer.save_pretrained(args.OUTPUT_DIR)
    print(f"[+] Model saved")

    # ============================================================
    # Step 6: Generate HELM configuration files
    # ============================================================
    print(f"\n[*] Generating HELM configuration files ...")

    CODE_MODEL_TAG = 'CODE_MODEL_TAG'
    TEXT_MODEL_TAG = 'TEXT_MODEL_TAG'
    PARTIAL_FUNCTIONALITY_TEXT_MODEL_TAG = 'PARTIAL_FUNCTIONALITY_TEXT_MODEL_TAG'

    # New deployment and metadata entries
    new_deployment = {
        'name': f'huggingface/{model_name}',
        'model_name': f'pen2sword/{model_name}',
        'tokenizer_name': args.TOKENIZER_NAME,
        'max_sequence_length': 4096,
        'client_spec': {
            'class_name': 'helm.clients.huggingface_client.HuggingFaceClient',
            'args': {
                'pretrained_model_name_or_path': os.path.abspath(args.OUTPUT_DIR)
            }
        }
    }

    new_metadata = {
        'name': f'pen2sword/{model_name}',
        'display_name': model_name,
        'description': 'Pen2Sword model',
        'creator_organization_name': 'hkust',
        'access': 'limited',
        'num_parameters': args.NUM_PARAMETERS,
    }

    # Always save to OUTPUT_DIR first
    deployments_file = os.path.join(args.OUTPUT_DIR, 'model_deployments.yaml')
    metadata_file = os.path.join(args.OUTPUT_DIR, 'model_metadata.yaml')

    with open(deployments_file, 'w') as f:
        f.write('model_deployments:\n')
        yaml.dump([new_deployment], f, default_flow_style=False, allow_unicode=True, indent=2)
    print(f"[+] Created: {deployments_file}")

    with open(metadata_file, 'w') as f:
        yaml.dump([new_metadata], f, default_flow_style=False, allow_unicode=True, indent=2)
        f.write(f"  release_date: 2024-10-01\n")
        f.write(f"  tags: [{CODE_MODEL_TAG}, {TEXT_MODEL_TAG}, {PARTIAL_FUNCTIONALITY_TEXT_MODEL_TAG}]\n")
    print(f"[+] Created: {metadata_file}")

    # If HELM_DIR specified, APPEND to existing yaml files
    if args.HELM_DIR:
        helm_prod_env = os.path.join(args.HELM_DIR, 'prod_env')
        helm_deployments = os.path.join(helm_prod_env, 'model_deployments.yaml')
        helm_metadata = os.path.join(helm_prod_env, 'model_metadata.yaml')

        # Create prod_env directory if it doesn't exist
        if not os.path.exists(helm_prod_env):
            os.makedirs(helm_prod_env)
            print(f"[+] Created directory: {helm_prod_env}")

        # Append to model_deployments.yaml
        if os.path.exists(helm_deployments):
            # Read existing entries
            with open(helm_deployments, 'r') as f:
                content = f.read()
            # Remove trailing newline and 'model_deployments:' line, find last entry
            lines = content.strip().split('\n')
            # Find the last blank-line separated block
            last_blank = len(lines) - 1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == '':
                    last_blank = i
                    break
            # Append new entry after existing content
            with open(helm_deployments, 'a') as f:
                f.write('\n')
                yaml.dump([new_deployment], f, default_flow_style=False, allow_unicode=True, indent=2)
            print(f"[+] Appended to: {helm_deployments}")
        else:
            with open(helm_deployments, 'w') as f:
                f.write('model_deployments:\n')
                yaml.dump([new_deployment], f, default_flow_style=False, allow_unicode=True, indent=2)
            print(f"[+] Created: {helm_deployments}")

        # Append to model_metadata.yaml
        if os.path.exists(helm_metadata):
            with open(helm_metadata, 'a') as f:
                yaml.dump([new_metadata], f, default_flow_style=False, allow_unicode=True, indent=2)
                f.write(f"  release_date: 2024-10-01\n")
                f.write(f"  tags: [{CODE_MODEL_TAG}, {TEXT_MODEL_TAG}, {PARTIAL_FUNCTIONALITY_TEXT_MODEL_TAG}]\n")
            print(f"[+] Appended to: {helm_metadata}")
        else:
            with open(helm_metadata, 'w') as f:
                yaml.dump([new_metadata], f, default_flow_style=False, allow_unicode=True, indent=2)
                f.write(f"  release_date: 2024-10-01\n")
                f.write(f"  tags: [{CODE_MODEL_TAG}, {TEXT_MODEL_TAG}, {PARTIAL_FUNCTIONALITY_TEXT_MODEL_TAG}]\n")
            print(f"[+] Created: {helm_metadata}")

    # ============================================================
    # Summary
    # ============================================================
    total_params = sum(p.numel() for p in targetmodel.parameters())
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
    print(f"  Merged model: {args.OUTPUT_DIR}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Local HELM files:")
    print(f"    {args.OUTPUT_DIR}/model_deployments.yaml")
    print(f"    {args.OUTPUT_DIR}/model_metadata.yaml")
    if args.HELM_DIR:
        print(f"  Appended to HELM config:")
        print(f"    {args.HELM_DIR}/prod_env/model_deployments.yaml")
        print(f"    {args.HELM_DIR}/prod_env/model_metadata.yaml")
    print(f"\n  Available models in HELM: pen2sword/{model_name}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)