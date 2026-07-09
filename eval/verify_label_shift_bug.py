"""Confirms and quantifies a training-time label-shift bug found while diagnosing
the eval/probe_in_distribution.py / eval/recall.py top1 anomaly (see
eval/inspect_competitors.py and eval/debug_padding_bug.py -- both ruled out padding/
attention-implementation/truncation as the cause, leaving "the model's own learned
behavior" as the explanation: at every probed position it confidently predicts the
SECOND token of the correct multi-token answer instead of the first).

Root cause (code-level, confirmed by reading the installed transformers source,
`transformers/models/gpt2/modeling_gpt2.py`): `GPT2LMHeadModel.forward(labels=...)`
ALREADY shifts internally (`shift_logits = logits[:, :-1]`, `shift_labels =
labels[:, 1:]`) -- HF's documented convention is to pass `labels=input_ids`
UNSHIFTED. But `training/data.py:PackedTokenDataset.__getitem__` already returns a
PRE-shifted pair (`x = chunk[:-1]`, `y = chunk[1:]`), and both `training/train.py`
(the actual training loop) and `training/eval.py:compute_perplexity` (used for
every checkpoint's held-out-ppl number, including the run_summary.json best_val_ppl
figures already reported) call `model(input_ids=x, labels=y)` -- i.e. the
ALREADY-shifted `y`, which the model then shifts AGAIN internally. Net effect: the
model has been trained (and evaluated) on predicting the token TWO positions ahead,
not one -- a strictly harder task, and evaluating a "predict 2 ahead" model against
the standard "predict 1 ahead" convention (as any external tool, including this
project's own eval/recall.py, does) looks like a badly undertrained model even if
the 2-ahead objective itself was reasonably well learned.

This script quantifies the damage: computes held-out perplexity on the SAME
checkpoint and SAME val data BOTH ways -- (a) exactly reproducing the current bug
(labels=y, matches training/eval.py precisely) and (b) the correct convention
(labels=x, matches HF's documented usage) -- to show how much of the previously
reported ~330-347 ppl was an artifact of this bug vs. genuine undertraining.
Cheap: one checkpoint load, ~20 held-out batches, two loss computations (reusing
the loaded model, no retraining).

Usage:
    python -m eval.verify_label_shift_bug --run-name gpt2-small-openwebtext-T1280-sequential \
        --config training/configs/full_run_T1280.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.config import TrainingConfig
from training.data import PackedTokenDataset
from training.model import build_model


@torch.no_grad()
def compute_perplexity_both_ways(model, loader: DataLoader, device: str, dtype: torch.dtype, max_batches: int):
    model.eval()
    buggy_loss_sum = fixed_loss_sum = 0.0
    n_tokens = 0
    on_cuda = device.startswith("cuda")
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device, non_blocking=on_cuda), y.to(device, non_blocking=on_cuda)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
            out_buggy = model(input_ids=x, labels=y)   # exactly what training/eval.py + train.py do
            buggy_loss_val = out_buggy.loss.item()
            del out_buggy
            if on_cuda:
                torch.cuda.empty_cache()
            out_fixed = model(input_ids=x, labels=x)   # HF's documented convention
            fixed_loss_val = out_fixed.loss.item()
            del out_fixed
            if on_cuda:
                torch.cuda.empty_cache()
        nt = y.numel()
        buggy_loss_sum += buggy_loss_val * nt
        fixed_loss_sum += fixed_loss_val * nt
        n_tokens += nt
    buggy_mean = buggy_loss_sum / n_tokens
    fixed_mean = fixed_loss_sum / n_tokens
    return {
        "buggy_loss": buggy_mean,
        "buggy_ppl": math.exp(min(buggy_mean, 20.0)),
        "fixed_loss": fixed_mean,
        "fixed_ppl": math.exp(min(fixed_mean, 20.0)),
        "n_tokens": n_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16, help="deliberately small -- this GPU had little free memory and this script briefly holds two forward passes' worth of logits")
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    device = cfg.run.device if torch.cuda.is_available() else "cpu"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.run.dtype]

    val_ds = PackedTokenDataset(cfg.data.val_path, cfg.model.n_positions, overlapping=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    ckpt_dir = Path(cfg.run.output_dir) / args.run_name / "checkpoints" / "latest"
    model = build_model(cfg.model)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    model.to(device)

    result = compute_perplexity_both_ways(model, val_loader, device, dtype, args.max_batches)
    print(f"\nheld-out perplexity, {result['n_tokens']} tokens, checkpoint {ckpt_dir}:")
    print(f"  BUGGY  (labels=y, matches training/eval.py + train.py as-shipped): loss={result['buggy_loss']:.4f}  ppl={result['buggy_ppl']:.2f}")
    print(f"  FIXED  (labels=x, HF's documented convention):                    loss={result['fixed_loss']:.4f}  ppl={result['fixed_ppl']:.2f}")


if __name__ == "__main__":
    main()
