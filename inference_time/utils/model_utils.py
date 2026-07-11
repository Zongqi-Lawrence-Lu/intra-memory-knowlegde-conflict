"""
Shared utilities for loading models and setting up logging, used by all
inference_time/ method scripts.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


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


# T-sweep conditions (experimental_plans.tex Sec.scale/seeds): each entry is the
# {run_name, config, population} triple slurm/run_T{T}.sbatch already trains and
# evaluates under -- kept here as the single source of truth for the inference_time/
# baselines rather than re-deriving it per script. All three share the same 1,680
# entities/relations/values (S1's T=32->T=80 migration note); only each entity's
# recorded (n_a, n_b) split, and the trained checkpoint itself, differ by T.
T_CONDITIONS = {
    80: {
        "run_name": "gpt2-small-openwebtext-T80",
        "config": "training/configs/full_run_T80.yaml",
        "population": "results/population.json",
    },
    320: {
        "run_name": "gpt2-small-openwebtext-T320",
        "config": "training/configs/full_run_T320.yaml",
        "population": "results/population_T320.json",
    },
    1280: {
        "run_name": "gpt2-small-openwebtext-T1280",
        "config": "training/configs/full_run_T1280.yaml",
        "population": "results/population_T1280.json",
    },
}


def load_trained_model(
    T: int,
    checkpoint_step: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
):
    """Load a real T-sweep checkpoint the same way eval/recall.py does: read the
    run's own training/configs/*.yaml via TrainingConfig, build the architecture via
    training.model.build_model(cfg.model) (not a hand-duplicated config dict), and
    load_state_dict from the checkpoint directory's model.pt (not HF from_pretrained,
    which cannot read this project's checkpoint layout at all -- see model_utils.py's
    prior single-.pt-file-only loader, now only used for --model-based smoke testing
    against a plain HF checkpoint).

    Raises FileNotFoundError if T is unknown or no checkpoint exists yet -- deliberately
    no silent fallback to pretrained gpt2, since that would make a broken checkpoint
    path indistinguishable from an intentional smoke test.

    Returns (model, tokenizer, cfg) -- cfg is returned too since callers (e.g. probe
    generation) may need cfg.run.dtype / cfg.model for consistency with training.
    """
    # Imported lazily (not at module top) so `--model`-only smoke-test usage of this
    # module, with no GPU / no training/ package available, still works.
    from training.checkpoint import list_full_checkpoints
    from training.config import TrainingConfig
    from training.model import build_model

    if T not in T_CONDITIONS:
        raise FileNotFoundError(
            f"Unknown T={T}; known T-sweep conditions are {sorted(T_CONDITIONS)}."
        )
    cond = T_CONDITIONS[T]
    logger = logging.getLogger(__name__)

    cfg_path = REPO_ROOT / cond["config"]
    if not cfg_path.exists():
        raise FileNotFoundError(f"T={T} config not found: {cfg_path}")
    cfg = TrainingConfig.from_yaml(cfg_path)

    ckpt_root = REPO_ROOT / cfg.run.output_dir / cond["run_name"] / "checkpoints"
    all_ckpts = list_full_checkpoints(ckpt_root)
    if not all_ckpts:
        raise FileNotFoundError(
            f"No full checkpoints found for T={T} under {ckpt_root} -- has "
            f"slurm/run_T{T}.sbatch finished?"
        )
    if checkpoint_step is not None:
        matches = [c for c in all_ckpts if c[0] == checkpoint_step]
        if not matches:
            raise FileNotFoundError(
                f"No T={T} checkpoint at step {checkpoint_step}; have "
                f"{[c[0] for c in all_ckpts]}"
            )
        step, kind, ckpt_dir = matches[0]
    else:
        step, kind, ckpt_dir = all_ckpts[-1]  # highest step: `best/` post-finalize, else latest slot

    target_device = get_device(device)
    logger.info("Loading T=%d checkpoint (step=%d, kind=%s) from %s", T, step, kind, ckpt_dir)

    model = build_model(cfg.model)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    model = model.to(dtype).to(target_device)
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("T=%d model loaded on %s (dtype=%s)", T, target_device, dtype)
    return model, tokenizer, cfg


def add_model_selection_args(parser: argparse.ArgumentParser) -> None:
    """Adds --T/--checkpoint-step (real trained checkpoint) alongside the existing
    --model (plain HF checkpoint, for --T-less smoke testing) to a run_*.py script's
    argparser. Call this instead of `p.add_argument("--model", default="gpt2")`."""
    parser.add_argument(
        "--T", type=int, default=None, choices=sorted(T_CONDITIONS),
        help="T-sweep condition to load a real trained checkpoint for. If omitted, "
             "falls back to --model (plain HF checkpoint/id) for smoke testing.",
    )
    parser.add_argument(
        "--checkpoint-step", type=int, default=None,
        help="Specific checkpoint step for --T (default: latest/best available).",
    )
    parser.add_argument("--model", default="gpt2", help="Used only when --T is not given.")


def resolve_model(args: argparse.Namespace, device: Optional[str], dtype: torch.dtype):
    """Companion to add_model_selection_args: loads a real T-sweep checkpoint if
    args.T is given, else falls back to the existing --model-based loader. Returns
    (model, tokenizer) -- matching load_model_and_tokenizer's return shape, so
    callers don't need to branch themselves."""
    if getattr(args, "T", None) is not None:
        model, tokenizer, _cfg = load_trained_model(
            args.T, checkpoint_step=getattr(args, "checkpoint_step", None), device=device, dtype=dtype
        )
        return model, tokenizer
    return load_model_and_tokenizer(args.model, device=device, dtype=dtype)


def default_experiment_name(base: str, args: argparse.Namespace) -> str:
    """Auto-suffixes a default --experiment_name with _T{T} when --T is given, so
    back-to-back runs across T conditions don't overwrite each other's
    results/<experiment_name>.json under the same default name."""
    T = getattr(args, "T", None)
    return f"{base}_T{T}" if T is not None else base


def probe_dir_for(args: argparse.Namespace) -> Optional[Path]:
    """results/baseline_probes/T{T}/ for the given --T, or None if --T wasn't given
    (in which case callers keep their existing DEMO_PROBES fallback)."""
    T = getattr(args, "T", None)
    return (REPO_ROOT / "results" / "baseline_probes" / f"T{T}") if T is not None else None


def join_prompt_answer(prompt: str, ans: str) -> str:
    """Concatenates a prompt and a free-text answer/continuation with the token-
    boundary space GPT-2 BPE expects between words, instead of the bare `prompt + ans`
    used previously across dola.py/steering_utils.py/coiecd.py/adacad.py/cocoa.py/
    arr.py. Feeding the tokenizer "...isAldergrove" (no space) tokenizes differently
    than "...is Aldergrove" would in natural text -- it can put the answer's first
    token on the wrong side of a BPE merge boundary, corrupting the very
    log-probability the scoring routines are trying to measure. `ans` is assumed not
    to already carry a leading space (this project's probe values -- e.g. "Aldergrove"
    -- don't), but the check below is defensive in case a caller's answer already does
    (or starts with other whitespace/punctuation that shouldn't get a second space)."""
    if not ans or ans[0] in " \n\t":
        return prompt + ans
    return prompt + " " + ans


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
