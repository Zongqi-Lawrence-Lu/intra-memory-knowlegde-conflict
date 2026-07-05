"""Dual-cadence, step-indexed checkpointing (experimental_plans.tex §1.4).

Two concurrent cadences, both driven by step count rather than wall-clock time:

- Sparse/global: a full checkpoint (model + optimizer + scheduler + RNG state) every
  `sparse_interval_steps`, with only the last `keep_last_n_sparse` retained on disk.
  This is what `resume()` restarts from.
- Dense/event-triggered: inside a window around each entry of `injection_steps`,
  checkpoint every `dense_interval_steps`. Every dense checkpoint records metrics;
  only a stratified subset (first K events, plus a random sample of the rest) also
  gets full weights, to keep storage tractable. `injection_steps` is empty until the
  M2 corpus generator supplies a real schedule, so this cadence is inert by default.

Checkpoints (weights/optimizer state) live under `output/<run_name>/checkpoints/`,
which is gitignored. Per-step metrics are appended to
`results/<run_name>/train_metrics.jsonl`, which is tracked in git -- this is the
"final json" CLAUDE.md §6 asks to back up, as opposed to the raw checkpoint files.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from training.config import CheckpointConfig


class CheckpointManager:
    def __init__(self, cfg: CheckpointConfig, output_dir: str | Path, results_dir: str | Path, run_name: str):
        self.cfg = cfg
        self.ckpt_root = Path(output_dir) / run_name / "checkpoints"
        self.results_dir = Path(results_dir) / run_name
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.results_dir / "train_metrics.jsonl"
        self._dense_event_count = 0
        self._rng = random.Random(0)

    # ------------------------------------------------------------------ cadence

    def sparse_due(self, step: int) -> bool:
        return step > 0 and step % self.cfg.sparse_interval_steps == 0

    def _active_injection_window(self, step: int) -> bool:
        w = self.cfg.dense_window_steps
        return any(abs(step - s) <= w for s in self.cfg.injection_steps)

    def dense_due(self, step: int) -> bool:
        if not self.cfg.injection_steps:
            return False
        return step > 0 and self._active_injection_window(step) and step % self.cfg.dense_interval_steps == 0

    def _dense_keeps_full_weights(self) -> bool:
        idx = self._dense_event_count
        self._dense_event_count += 1
        if idx < self.cfg.dense_full_weight_first_k:
            return True
        return self._rng.random() < self.cfg.dense_full_weight_sample_rate

    # ------------------------------------------------------------------ save/load

    def _save_full(self, step: int, kind: str, model, optimizer, scheduler, extra_meta: dict) -> Path:
        # kind is part of the dir name (not just meta.json) so a sparse and a dense
        # save landing on the same step don't collide and silently overwrite each
        # other's meta.json (which would drop the "sparse" tag and break rotation).
        ckpt_dir = self.ckpt_root / f"step_{step:08d}_{kind}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        torch.save(
            {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            ckpt_dir / "rng_state.pt",
        )
        meta = {
            "step": step,
            "kind": kind,
            "full_weights": True,
            "wall_time": time.time(),
            **extra_meta,
        }
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return ckpt_dir

    def maybe_save(self, step: int, model, optimizer, scheduler, log_fn=None, extra_meta: Optional[dict] = None) -> None:
        """Call once per training step; internally decides sparse/dense/none."""
        extra_meta = extra_meta or {}

        if self.sparse_due(step):
            ckpt_dir = self._save_full(step, kind="sparse", model=model, optimizer=optimizer, scheduler=scheduler, extra_meta=extra_meta)
            if log_fn:
                log_fn(f"[checkpoint] sparse full checkpoint at step {step} -> {ckpt_dir}")
            self._rotate_sparse()

        if self.dense_due(step):
            full = self._dense_keeps_full_weights()
            if full:
                ckpt_dir = self._save_full(step, kind="dense", model=model, optimizer=optimizer, scheduler=scheduler, extra_meta=extra_meta)
                if log_fn:
                    log_fn(f"[checkpoint] dense full checkpoint at step {step} -> {ckpt_dir}")
            else:
                # metrics-only marker; the actual metrics line is written by log_metrics()
                ckpt_dir = self.ckpt_root / f"step_{step:08d}_dense"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                meta = {"step": step, "kind": "dense", "full_weights": False, "wall_time": time.time(), **extra_meta}
                with open(ckpt_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2)
                if log_fn:
                    log_fn(f"[checkpoint] dense metrics-only marker at step {step}")

    def _rotate_sparse(self) -> None:
        if self.cfg.keep_last_n_sparse < 0:
            return
        sparse_dirs = []
        for d in sorted(self.ckpt_root.glob("step_*")):
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("kind") == "sparse" and meta.get("full_weights"):
                sparse_dirs.append(d)
        excess = len(sparse_dirs) - self.cfg.keep_last_n_sparse
        for d in sparse_dirs[:max(0, excess)]:
            for f in d.glob("*"):
                f.unlink()
            d.rmdir()

    def find_latest_full_checkpoint(self) -> Optional[Path]:
        candidates = []
        for d in sorted(self.ckpt_root.glob("step_*")):
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("full_weights"):
                candidates.append((meta["step"], d))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[-1][1]

    def resume(self, model, optimizer, scheduler, device) -> int:
        """Loads the latest full checkpoint in place. Returns the step to resume
        FROM (i.e. the next step to run is resume_step + 1); returns 0 if none found."""
        ckpt_dir = self.find_latest_full_checkpoint()
        if ckpt_dir is None:
            return 0
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
        optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device))
        if scheduler is not None and (ckpt_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", map_location=device))
        rng = torch.load(ckpt_dir / "rng_state.pt", map_location="cpu")
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        with open(ckpt_dir / "meta.json") as f:
            meta = json.load(f)
        return meta["step"]

    # ------------------------------------------------------------------ metrics

    def log_metrics(self, record: dict) -> None:
        record = {"wall_time": time.time(), **record}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def write_run_summary(self, summary: dict) -> None:
        with open(self.results_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
