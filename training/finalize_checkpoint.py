"""Post-training checkpoint finalization (2026-07-09) -- explicit user request:
"I authorize you to save up to three non-overwriting checkpoints throughout the
training..., but when we are done, we must explicitly keep only one, the best
one." training/checkpoint.py's rotating slot pool (training/config.py:
CheckpointConfig.num_checkpoint_slots) is a during-training safety margin so a
late-discovered divergence (training/config.py:StabilityConfig) can still be
rolled back to a healthy checkpoint instead of only ever having the single most
recent (possibly already-broken) one on disk -- it is not the intended steady
state once a run is actually done.

This script picks the retained slot with the lowest held-out val_ppl
(training/checkpoint.py:best_checkpoint_from_metrics, matched against
train_metrics.jsonl), renames it to checkpoints/best/, and deletes every other
slot plus any stale write-scratch directories. Idempotent: rerunning on an
already-finalized run (only best/ present) is a no-op.

Run after training completes, before recall eval, so eval scores the actual best
checkpoint rather than whatever the last training step happened to leave behind
(which, if the run needed a late StabilityConfig rollback, may not be the most
recent slot at all).

    python -m training.finalize_checkpoint --run-name gpt2-small-openwebtext-t80-stable --config training/configs/full_run_T80.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable

from training.checkpoint import FINALIZED_DIR_NAME, best_checkpoint_from_metrics, list_full_checkpoints
from training.config import TrainingConfig


def finalize(ckpt_root: Path, metrics_path: Path, log_fn: Callable[[str], None] = print) -> Path:
    existing = list_full_checkpoints(ckpt_root)
    if not existing:
        raise FileNotFoundError(f"no full checkpoints found under {ckpt_root}")

    if len(existing) == 1 and existing[0][2].name == FINALIZED_DIR_NAME:
        log_fn(f"already finalized: {existing[0][2]}")
        return existing[0][2]

    best_step, best_kind, best_path, best_val_ppl = best_checkpoint_from_metrics(ckpt_root, metrics_path)
    log_fn(f"best checkpoint: step={best_step} kind={best_kind} val_ppl={best_val_ppl} ({best_path})")

    final_dir = ckpt_root / FINALIZED_DIR_NAME
    if final_dir.exists() and final_dir != best_path:
        shutil.rmtree(final_dir)
    if best_path != final_dir:
        best_path.rename(final_dir)

    for step, kind, path in existing:
        if path in (best_path, final_dir) or not path.exists():
            continue
        shutil.rmtree(path)
        log_fn(f"removed non-best checkpoint: step={step} kind={kind} ({path})")

    # Stale scratch dirs from CheckpointManager's write-then-swap pattern
    # (training/checkpoint.py:_write_checkpoint_to_disk), if any survived a crash.
    tmp_dir = ckpt_root / "_tmp_write"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    for stray in ckpt_root.glob("_*_old"):
        shutil.rmtree(stray)

    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", default=None, help="defaults to training/configs/<run-name>.yaml")
    args = parser.parse_args()

    cfg_path = args.config or f"training/configs/{args.run_name}.yaml"
    cfg = TrainingConfig.from_yaml(cfg_path)

    ckpt_root = Path(cfg.run.output_dir) / args.run_name / "checkpoints"
    metrics_path = Path(cfg.run.results_dir) / args.run_name / "train_metrics.jsonl"
    final_dir = finalize(ckpt_root, metrics_path)
    print(f"Finalized {args.run_name}: kept only {final_dir}")


if __name__ == "__main__":
    main()
