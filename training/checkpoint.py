"""Single-slot overwriting checkpoint manager (experimental_plans.tex §1.4,
superseding its prior sparse-rotation + dense-stratified-retention design).

Exactly one full checkpoint (model + optimizer + scheduler + RNG state) exists on
disk at a time, under `output/<run_name>/checkpoints/latest/` -- every save
(sparse-cadence, or the unconditional final save) overwrites that same slot via a
directory rename-swap, never an in-place file overwrite, so a crash mid-write
can't corrupt the one surviving checkpoint (see `_write_checkpoint_to_disk`).
Explicit request: at most one pair of weights stored on disk the whole time, so a
sweep of N simultaneous training jobs costs a small, constant amount of checkpoint
storage regardless of run length, at the cost of losing the ability to inspect any
weight snapshot except the most recent one (no more retrospective sparse-grid or
dense-window mechanistic probing across a run's history).

Dense/event-triggered cadence (inside a window around each entry of
`injection_steps`, checkpoint every `dense_interval_steps`) still logs metrics --
useful for a fine-grained loss/eval timeline near occurrence events -- but no
longer captures full weights at all; that was the prior design's only source of
unbounded on-disk accumulation. `injection_steps` is empty until the corpus
generator supplies a real schedule, so this cadence is inert by default.

Checkpoints (weights/optimizer state) live under `output/<run_name>/checkpoints/`,
which is gitignored. Per-step metrics are appended to
`results/<run_name>/train_metrics.jsonl`, which is tracked in git -- this is the
"final json" CLAUDE.md §6 asks to back up, as opposed to the raw checkpoint files.
"""
from __future__ import annotations

import bisect
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from training.config import CheckpointConfig


def list_full_checkpoints(ckpt_root: Path) -> list[tuple[int, str, Path]]:
    """The single complete (meta.json present, full_weights=True) checkpoint dir
    under ckpt_root -- `latest/` -- as a length-0-or-1 list, for compatibility with
    callers (eval/recall.py, training/probe_fact.py) written against the old
    multi-checkpoint scan. There is never more than one on disk by construction."""
    latest = ckpt_root / "latest"
    meta_path = latest / "meta.json"
    if not meta_path.exists():
        return []
    with open(meta_path) as f:
        meta = json.load(f)
    if not meta.get("full_weights"):
        return []
    return [(meta["step"], meta.get("kind", ""), latest)]


def _to_cpu_recursive(obj):
    """Deep-clones a state_dict (model/optimizer/scheduler) to CPU. optimizer.state_dict()
    nests tensors (exp_avg/exp_avg_sq per param) inside dicts/lists alongside plain
    python scalars (param_groups), so a shallow {k: v.cpu()} isn't enough."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_recursive(v) for v in obj]
    return obj


class CheckpointManager:
    def __init__(self, cfg: CheckpointConfig, output_dir: str | Path, results_dir: str | Path, run_name: str):
        self.cfg = cfg
        self.ckpt_root = Path(output_dir) / run_name / "checkpoints"
        self.results_dir = Path(results_dir) / run_name
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.results_dir / "train_metrics.jsonl"
        # Sorted once here rather than re-sorted/re-scanned per step: with
        # injection_steps auto-derived from occurrence_log.json (training/
        # injection_schedule.py) this list can be in the thousands, and
        # _active_injection_window is called every single training step.
        self._sorted_injection_steps = sorted(self.cfg.injection_steps)
        # Single-worker pool: checkpoint disk writes (~1.5-2GB for this model size)
        # happen off the training-loop thread, so a save doesn't stall compute for
        # however long the write takes -- with sparse_interval_steps sized to
        # minutes of training (S1.4), that write should always be long finished by
        # the time the next one is submitted (_wait_for_pending_save), so this is
        # normally a no-op wait, not a real block.
        self._io_pool = ThreadPoolExecutor(max_workers=1)
        self._pending_save = None

    # ------------------------------------------------------------------ cadence

    def sparse_due(self, step: int) -> bool:
        return step > 0 and step % self.cfg.sparse_interval_steps == 0

    def _active_injection_window(self, step: int) -> bool:
        # O(log n) via bisect rather than an O(n) scan of every injection_steps entry
        # per call -- with an auto-derived injection_steps list in the thousands
        # (training/injection_schedule.py) and this called every step, an O(n) scan
        # would multiply out to a real cost over a 38k-step run.
        w = self.cfg.dense_window_steps
        steps = self._sorted_injection_steps
        if not steps:
            return False
        idx = bisect.bisect_left(steps, step)
        if idx < len(steps) and steps[idx] - step <= w:
            return True
        if idx > 0 and step - steps[idx - 1] <= w:
            return True
        return False

    def dense_due(self, step: int) -> bool:
        if not self.cfg.injection_steps:
            return False
        return step > 0 and self._active_injection_window(step) and step % self.cfg.dense_interval_steps == 0

    # ------------------------------------------------------------------ save/load

    def _wait_for_pending_save(self) -> None:
        """Blocks until the previous checkpoint's background write is done. Called
        right before submitting a new one, so at most one save is ever in flight --
        in practice this should always return immediately (see __init__), since a
        disk write takes seconds and sparse_interval_steps is sized in minutes."""
        if self._pending_save is not None:
            self._pending_save.result()
            self._pending_save = None

    @staticmethod
    def _write_checkpoint_to_disk(ckpt_root: Path, model_sd, optim_sd, sched_sd, rng_state, meta) -> None:
        """Writes to a fresh tmp_write/ dir, then swaps it in for latest/ via two
        directory renames (both atomic on the same filesystem) rather than
        overwriting latest/'s files in place -- so at every instant either the old
        or the new checkpoint is a complete, valid latest/, never a half-written
        mix of both. old/ is removed only after the swap, i.e. after a new complete
        checkpoint already exists on disk, not before."""
        tmp_dir = ckpt_root / "_tmp_write"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        torch.save(model_sd, tmp_dir / "model.pt")
        torch.save(optim_sd, tmp_dir / "optimizer.pt")
        if sched_sd is not None:
            torch.save(sched_sd, tmp_dir / "scheduler.pt")
        torch.save(rng_state, tmp_dir / "rng_state.pt")
        # meta.json last: a tmp_write/ dir is only a complete checkpoint once this
        # exists, so a crash mid-write is never mistaken for a valid swap source.
        with open(tmp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        latest_dir = ckpt_root / "latest"
        old_dir = ckpt_root / "_latest_old"
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if latest_dir.exists():
            latest_dir.rename(old_dir)
        tmp_dir.rename(latest_dir)
        if old_dir.exists():
            shutil.rmtree(old_dir)

    def _save_full(self, step: int, kind: str, model, optimizer, scheduler, extra_meta: dict) -> Path:
        # Snapshot to CPU synchronously (a device->host copy, not a disk write --
        # cheap relative to what follows) so the background thread's tensors are
        # private copies the training loop can't mutate on the next step; only the
        # actual torch.save-to-disk calls (the slow part) move off this thread.
        model_sd = _to_cpu_recursive(model.state_dict())
        optim_sd = _to_cpu_recursive(optimizer.state_dict())
        sched_sd = _to_cpu_recursive(scheduler.state_dict()) if scheduler is not None else None
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        meta = {
            "step": step,
            "kind": kind,
            "full_weights": True,
            "wall_time": time.time(),
            **extra_meta,
        }

        self._wait_for_pending_save()
        self._pending_save = self._io_pool.submit(
            self._write_checkpoint_to_disk, self.ckpt_root, model_sd, optim_sd, sched_sd, rng_state, meta
        )
        return self.ckpt_root / "latest"

    def shutdown(self) -> None:
        """Blocks until any in-flight checkpoint write finishes, then tears down the
        writer thread. Call once at the end of training -- otherwise the very last
        checkpoint could still be mid-write when the process exits."""
        self._wait_for_pending_save()
        self._io_pool.shutdown(wait=True)

    def maybe_save(self, step: int, model, optimizer, scheduler, log_fn=None, extra_meta: Optional[dict] = None) -> None:
        """Call once per training step; internally decides sparse/dense/none."""
        extra_meta = extra_meta or {}

        if self.sparse_due(step):
            ckpt_dir = self._save_full(step, kind="sparse", model=model, optimizer=optimizer, scheduler=scheduler, extra_meta=extra_meta)
            if log_fn:
                log_fn(f"[checkpoint] sparse full checkpoint at step {step} -> {ckpt_dir} (overwrote previous latest/)")

        if self.dense_due(step):
            # Metrics-only marker -- dense saves never carry full weights (see
            # module docstring); the actual metrics line is written by log_metrics().
            marker_dir = self.ckpt_root / f"step_{step:08d}_dense"
            marker_dir.mkdir(parents=True, exist_ok=True)
            meta = {"step": step, "kind": "dense", "full_weights": False, "wall_time": time.time(), **extra_meta}
            with open(marker_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            if log_fn:
                log_fn(f"[checkpoint] dense metrics-only marker at step {step}")

    def save_final(self, step: int, model, optimizer, scheduler, log_fn=None, extra_meta: Optional[dict] = None) -> None:
        """Unconditional full save at the true last step, called once after the
        training loop ends. maybe_save() only fires on the sparse cadence, so
        without this the last checkpoint on disk is whichever cadence step happened
        to precede the actual final step (e.g. sparse_interval_steps=1800 can leave a
        gap of up to 1799 steps) -- whoever evaluates "the trained model" afterward
        should get the real final-step weights, not an approximation of them. This
        overwrites latest/ exactly like any other full save -- the final state
        supersedes whatever sparse checkpoint preceded it, per the single-slot
        policy (module docstring)."""
        ckpt_dir = self._save_full(step, kind="final", model=model, optimizer=optimizer, scheduler=scheduler, extra_meta=extra_meta or {})
        if log_fn:
            log_fn(f"[checkpoint] final full checkpoint at step {step} -> {ckpt_dir} (overwrote previous latest/)")
        self._wait_for_pending_save()  # block here so shutdown() right after has nothing left in flight

    def find_latest_full_checkpoint(self) -> Optional[Path]:
        candidates = list_full_checkpoints(self.ckpt_root)
        return candidates[-1][2] if candidates else None

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
