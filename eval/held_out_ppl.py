"""Standalone held-out perplexity against an arbitrary val set (2026-07-09),
reusing training/eval.py:compute_perplexity against a checkpoint already on disk
-- for comparing a run's out-of-domain WikiText-103 val_ppl (what the training
loop itself tracks throughout training, training/config.py:EvalConfig) against
its in-domain OpenWebText val_ppl (data/processed/held-out-openwebtext-val,
preprocess/build_openwebtext_val.py), isolating how much of a "still quite high"
val_ppl is domain shift vs undertraining.

    python -m eval.held_out_ppl --run-name gpt2-small-openwebtext-T320-shuffled \\
        --config training/configs/full_run_T320.yaml \\
        --val-path data/processed/held-out-openwebtext-val --label openwebtext-in-domain

Output: prints the result and appends {"label", "val_path", "checkpoint_step",
"val_ppl"} to results/<run_name>/held_out_ppl.jsonl (tracked, so multiple val
sets/checkpoints accumulate in one place rather than overwriting each other).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.checkpoint import list_full_checkpoints
from training.config import TrainingConfig
from training.data import PackedTokenDataset
from training.eval import compute_perplexity
from training.model import build_model

DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", default=None, help="defaults to training/configs/<run-name>.yaml")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="defaults to the latest/best full checkpoint")
    parser.add_argument("--val-path", required=True, help="a PackedTokenDataset directory (meta.json + *.bin)")
    parser.add_argument("--label", required=True, help="short tag for this val set, e.g. 'openwebtext-in-domain'")
    parser.add_argument("--batch-size", type=int, default=None, help="defaults to the run's own training batch_size")
    parser.add_argument("--max-batches", type=int, default=None, help="defaults to the whole val set once")
    args = parser.parse_args()

    cfg_path = args.config or f"training/configs/{args.run_name}.yaml"
    cfg = TrainingConfig.from_yaml(cfg_path)
    device = cfg.run.device if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[cfg.run.dtype]

    ckpt_root = Path(cfg.run.output_dir) / args.run_name / "checkpoints"
    all_ckpts = list_full_checkpoints(ckpt_root)
    if not all_ckpts:
        raise FileNotFoundError(f"no full checkpoints found under {ckpt_root}")
    if args.checkpoint_step is not None:
        targets = [c for c in all_ckpts if c[0] == args.checkpoint_step]
        if not targets:
            raise FileNotFoundError(f"no checkpoint at step {args.checkpoint_step}; have {[c[0] for c in all_ckpts]}")
        step, kind, ckpt_dir = targets[0]
    else:
        step, kind, ckpt_dir = all_ckpts[-1]

    model = build_model(cfg.model)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    batch_size = args.batch_size or cfg.data.batch_size
    val_ds = PackedTokenDataset(args.val_path, cfg.model.n_positions, overlapping=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    val_ppl = compute_perplexity(model, val_loader, device, dtype, max_batches=args.max_batches)
    print(f"{args.run_name} @ step {step} ({kind}) on {args.label} ({args.val_path}): val_ppl={val_ppl:.3f}")

    results_dir = Path(cfg.run.results_dir) / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "held_out_ppl.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "label": args.label,
                    "val_path": args.val_path,
                    "checkpoint_step": step,
                    "checkpoint_kind": kind,
                    "val_ppl": val_ppl,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
