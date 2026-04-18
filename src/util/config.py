"""
Configuration management for Pen2Sword.
"""
import os


def get_env_or_raise(env_var: str, error_message: str = None) -> str:
    """
    Get an environment variable or raise an error if not set.

    Args:
        env_var: Name of the environment variable
        error_message: Optional custom error message

    Returns:
        Value of the environment variable

    Raises:
        ValueError: If environment variable is not set
    """
    value = os.environ.get(env_var)
    if value is None:
        if error_message:
            raise ValueError(error_message)
        raise ValueError(
            f"Environment variable '{env_var}' is not set. "
            f"Please set it before running."
        )
    return value


def get_wandb_key() -> str:
    """
    Get Weights & Biases API key from environment.

    Returns:
        WANDB API key

    Raises:
        ValueError: If WANDB_API_KEY is not set
    """
    return get_env_or_raise(
        "WANDB_API_KEY",
        "Please set WANDB_API_KEY environment variable. "
        "Get your key from https://wandb.ai/settings"
    )


def get_model_path() -> str:
    """
    Get the target model path from environment or use default.

    Returns:
        Path to the target (large) model
    """
    return os.environ.get(
        "MODEL_PATH",
        "/path/to/vicuna-13b"  # Default placeholder
    )


def get_emb_model_path() -> str:
    """
    Get the embedding model path from environment or use default.

    Returns:
        Path to the embedding (small/expert) model
    """
    return os.environ.get(
        "EMBMODEL_PATH",
        "/path/to/vicuna-7b"  # Default placeholder
    )


# Default training configuration
DEFAULT_TRAINING_CONFIG = {
    "CUTOFF_LEN": 512,
    "VAL_SET_SIZE": 100,
    "RANDOM_SEED": 42,
    "EVAL_BATCH_SIZE": 2,
    "GRADIENT_ACCUMULATION_STEPS": 8,
    "SAVE_STEP": 1000,
    "ALPHA": 0.25,
    "EMB_MODULE_NAME": "embed_tokens",
    "UNFIX_MODULES": "embed_tokens",
    "TARGETMODUOLES": "q_proj, v_proj",
    "DATASET_NAME": "meta-math/MetaMathQA",
    "OUTPUT_DIR": "./model_save",
    "PROJECT_NAME": "pen2sword_experiment",
    "DEBUG": False,
    "LORA": True,
}
