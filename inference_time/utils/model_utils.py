"""
Shared utilities for loading models and setting up logging, used by all
inference_time/ method scripts.
"""

import logging
import os
import sys
from typing import Optional, Tuple

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# GPT-2-small-from-scratch architecture (experimental_plans.tex §1.1; must match
# training/config.py:ModelConfig defaults). A raw .pt state-dict produced by the
# training pipeline carries no config.json of its own, so this is the config the
# training loop actually used and must be reconstructed here to load it.
_PROJECT_MODEL_CONFIG = dict(
    vocab_size=50257,
    n_positions=512,
    n_ctx=512,
    n_embd=768,
    n_layer=12,
    n_head=12,
)


def load_model_and_tokenizer(
    model_name_or_path: str = "gpt2",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """
    Load a GPT-2 model and tokenizer from HuggingFace or a local checkpoint.

    For a local checkpoint produced by the training pipeline, pass the directory
    that contains config.json + pytorch_model.bin (or a .pt state-dict file).

    Default "gpt2" is this project's actual model size (GPT-2-small,
    experimental_plans.tex §1.1, which supersedes the gpt2-medium default in the
    original project outline) -- not a HuggingFace-library default.
    """
    target_device = get_device(device)
    logger = logging.getLogger(__name__)

    if os.path.isfile(model_name_or_path) and model_name_or_path.endswith(".pt"):
        # Raw state-dict saved by the training pipeline: reconstruct this
        # project's GPT-2-small architecture (no config.json is saved alongside).
        logger.info("Loading state-dict from %s", model_name_or_path)
        state = torch.load(model_name_or_path, map_location="cpu")
        # Accept either {"model_state_dict": ...} or a bare state dict
        state_dict = state.get("model_state_dict", state)
        config = GPT2Config(**_PROJECT_MODEL_CONFIG)
        model = GPT2LMHeadModel(config)
        model.load_state_dict(state_dict)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        logger.info("Loading model from %s", model_name_or_path)
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(dtype).to(target_device)
    model.eval()
    logger.info("Model loaded on %s (dtype=%s)", target_device, dtype)
    return model, tokenizer


def setup_logging(log_dir: Optional[str], experiment_name: str) -> logging.Logger:
    """Configure root logger to write to both stdout and an optional file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{experiment_name}.log")
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(experiment_name)
