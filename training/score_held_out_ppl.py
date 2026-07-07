"""Re-scores a saved checkpoint's perplexity against a real held-out set (Fix for the
val_path-falls-back-to-train_path gap: training's own eval loop can only measure
perplexity against whatever val_path was configured *during that run*; this lets any
already-trained checkpoint be re-scored against a held-out set built afterward, e.g.
preprocess/assemble_held_out.py's WikiText-103 val split, without retraining).

Usage (needs a GPU -- see CLAUDE.md S5):
    python -m training.score_held_out_ppl --run-name gpt2-small-baseline-openwebtext-t80 \\
        --held-out-path data/processed/held-out-wikitext-val
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
    parser.add_argument("--config", default=None, help="defaults to training/configs/<run-name>.yaml if present, else full_run.yaml")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="defaults to the latest full checkpoint")
    parser.add_argument("--held-out-path", required=True)
    parser.add_argument("--batch-size", type=int, default=None, help="defaults to the run's own config batch_size")
    args = parser.parse_args()

    cfg_path = args.config or f"training/configs/{args.run_name}.yaml"
    if not Path(cfg_path).exists():
        cfg_path = "training/configs/full_run.yaml"
    cfg = TrainingConfig.from_yaml(cfg_path)
    device = cfg.run.device if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[cfg.run.dtype]
    batch_size = args.batch_size or cfg.data.batch_size

    ckpt_root = Path(cfg.run.output_dir) / args.run_name / "checkpoints"
    all_ckpts = list_full_checkpoints(ckpt_root)
    if not all_ckpts:
        raise FileNotFoundError(f"no full checkpoints found under {ckpt_root}")
    if args.checkpoint_step is not None:
        matches = [c for c in all_ckpts if c[0] == args.checkpoint_step]
        if not matches:
            raise FileNotFoundError(f"no checkpoint at step {args.checkpoint_step}; have {[c[0] for c in all_ckpts]}")
        step, kind, ckpt_dir = matches[0]
    else:
        step, kind, ckpt_dir = all_ckpts[-1]

    model = build_model(cfg.model)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    model.to(device)

    held_out_ds = PackedTokenDataset(args.held_out_path, cfg.model.n_positions, overlapping=False)
    held_out_loader = DataLoader(held_out_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    ppl = compute_perplexity(model, held_out_loader, device, dtype)
    n_batches = len(held_out_ds) // batch_size

    print(f"checkpoint: {ckpt_dir} (step {step}, {kind})")
    print(f"held-out set: {args.held_out_path} ({len(held_out_ds)} non-overlapping windows, {n_batches} batches)")
    print(f"held-out perplexity: {ppl:.3f}")

    results_dir = Path(cfg.run.results_dir) / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"held_out_ppl_step{step}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_dir),
                "step": step,
                "held_out_path": args.held_out_path,
                "n_windows": len(held_out_ds),
                "n_batches": n_batches,
                "batch_size": batch_size,
                "held_out_perplexity": ppl,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
