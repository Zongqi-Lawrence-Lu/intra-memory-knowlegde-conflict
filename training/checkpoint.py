"""Rotating multi-slot checkpoint manager (2026-07-09 revision, superseding the
prior single-slot-overwrite design -- see training/config.py:CheckpointConfig's
docstring for the full rationale).

Up to `cfg.num_checkpoint_slots` full checkpoints (model + optimizer + scheduler +
RNG state) exist on disk at once, under `output/<run_name>/checkpoints/slot_*/`.
Once that many slots exist, the next save overwrites whichever slot holds the
*oldest* step (never the newest) via the same directory rename-swap as before
(never an in-place file overwrite, so a crash mid-write can't corrupt that slot's
previous contents -- see `_write_checkpoint_to_disk`). This bounds on-disk
checkpoint storage to a small, constant multiple (num_checkpoint_slots) of one
model's size regardless of run length, while giving `train.py`'s divergence
rollback (training/config.py:StabilityConfig) and `training/finalize_checkpoint.py`
(picks the single best-by-val_ppl slot and deletes the rest once a run is done)
something to actually choose between, unlike the pure single-slot design.

Dense/event-triggered cadence (inside a window around each entry of
`injection_steps`, checkpoint every `dense_interval_steps`) still logs metrics
only, no full weights -- unrelated to slot rotation, unchanged from the prior
revision. `injection_steps` is empty until the corpus generator supplies a real
schedule, so this cadence is inert by default.

Checkpoints (weights/optimizer state) live under `output/<run_name>/checkpoints/`,
which is gitignored. Per-step metrics are appended to
`results/<run_name>/train_metrics.jsonl`, which is tracked in git -- this is the
"final json" CLAUDE.md §6 asks to back up, as opposed to the raw checkpoint files.
"""
from __future__ import annotations

import bisect
import json
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from training.config import CheckpointConfig

SLOT_DIR_PREFIX = "slot_"
FINALIZED_DIR_NAME = "best"  # training/finalize_checkpoint.py's canonical single-survivor name
LEGACY_SLOT_NAME = "latest"  # pre-2026-07-09 single-slot-overwrite runs (e.g. the
# T=320/T=1280-shuffled runs already on disk) wrote here, not to slot_*/ -- recognized
# for read-compatibility so existing checkpoints stay discoverable by eval/recall.py,
# eval/held_out_ppl.py etc. New saves never target this name (CheckpointManager._pick_
# target_slot only ever picks slot_*/), so it's read-only legacy support, not written
# again once a slot_*/ (or, post-finalize, best/) exists for a given run.


def _slot_dirs(ckpt_root: Path) -> list[Path]:
    if not ckpt_root.exists():
        return []
    return sorted(
        p
        for p in ckpt_root.iterdir()
        if p.is_dir()
        and (p.name.startswith(SLOT_DIR_PREFIX) or p.name in (FINALIZED_DIR_NAME, LEGACY_SLOT_NAME))
    )


def list_full_checkpoints(ckpt_root: Path) -> list[tuple[int, str, Path]]:
    """Every complete (meta.json present, full_weights=True) checkpoint slot under
    ckpt_root, sorted ascending by step -- so callers written against the old
    length-0-or-1 single-checkpoint list (eval/recall.py, training/probe_fact.py)
    still get "most recent" from `[-1]` unchanged, and --all-checkpoints-style
    callers now see every retained slot instead of just one."""
    out = []
    for slot_dir in _slot_dirs(ckpt_root):
        meta_path = slot_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get("full_weights"):
            continue
        out.append((meta["step"], meta.get("kind", ""), slot_dir))
    out.sort(key=lambda t: t[0])
    return out


def _val_ppl_records(metrics_path: Path) -> list[tuple[int, float]]:
    records: list[tuple[int, float]] = []
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                record = json.loads(line)
                if "val_ppl" in record:
                    records.append((record["step"], record["val_ppl"]))
    records.sort(key=lambda t: t[0])
    return records


def best_val_ppl_from_metrics(metrics_path: Path) -> Optional[tuple[int, float]]:
    """(step, val_ppl) of the single lowest val_ppl ever logged in
    train_metrics.jsonl, or None if no val_ppl record exists. For train.py's
    run_summary.json write: the in-loop running-min this replaces was a
    process-local variable that silently reset to +inf across a preemption/resume
    (a fresh `python -m training.train` invocation), so a run whose *last*
    invocation happened to do no training steps (e.g. it resumed already at
    max_steps, only to re-save+write the summary) recorded best_val_ppl=Infinity
    despite real, finite val_ppl values sitting right there in the same run's own
    metrics file -- confirmed concretely for the T=320 and T=1280 shuffled runs.
    Reading it back from the persisted metrics file instead is resume-proof by
    construction, since every val_ppl this run ever logged is already there."""
    records = _val_ppl_records(metrics_path)
    if not records:
        return None
    return min(records, key=lambda t: t[1])


def best_checkpoint_from_metrics(
    ckpt_root: Path, metrics_path: Path
) -> Optional[tuple[int, str, Path, Optional[float]]]:
    """Picks the retained checkpoint slot whose step has the lowest held-out
    val_ppl logged in train_metrics.jsonl, for training/finalize_checkpoint.py.
    Matches each slot's step to the *last* val_ppl record at or before that step
    (sparse checkpoint cadence and eval cadence aren't generally the same, so an
    exact-step val_ppl record isn't guaranteed) -- not the training loss, since
    val_ppl is what the whole pipeline treats as the quality signal
    (training/train.py, experimental_plans.tex §1.5). Returns (step, kind, path,
    val_ppl) for the winning slot, or None if there are no checkpoints. val_ppl in
    the return is None if no val_ppl record at or before that step exists (falls
    back to the highest-step slot in that case, i.e. most training progress, on
    the assumption no signal is better than a clearly wrong one)."""
    checkpoints = list_full_checkpoints(ckpt_root)
    if not checkpoints:
        return None

    val_ppl_by_step = _val_ppl_records(metrics_path)
    steps_only = [s for s, _ in val_ppl_by_step]

    def val_ppl_at_or_before(step: int) -> Optional[float]:
        idx = bisect.bisect_right(steps_only, step) - 1
        return val_ppl_by_step[idx][1] if idx >= 0 else None

    scored = [(step, kind, path, val_ppl_at_or_before(step)) for step, kind, path in checkpoints]
    with_signal = [s for s in scored if s[3] is not None]
    if with_signal:
        return min(with_signal, key=lambda s: s[3])
    return max(scored, key=lambda s: s[0])


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

    def _pick_target_slot(self) -> Path:
        """Which slot_*/ directory the next save should write to: an unused slot
        name if fewer than num_checkpoint_slots exist yet, otherwise the existing
        slot holding the *oldest* step (rotating retention -- never evicts the
        newest, which is what a divergence rollback or finalize would need)."""
        existing = list_full_checkpoints(self.ckpt_root)  # sorted ascending by step
        used_names = {p.name for _, _, p in existing}
        for i in range(self.cfg.num_checkpoint_slots):
            name = f"{SLOT_DIR_PREFIX}{i}"
            if name not in used_names:
                return self.ckpt_root / name
        return existing[0][2]  # all slots occupied -- evict the oldest

    @staticmethod
    def _write_checkpoint_to_disk(ckpt_root: Path, target_dir: Path, model_sd, optim_sd, sched_sd, rng_state, meta) -> None:
        """Writes to a fresh tmp_write/ dir, then swaps it in for target_dir via two
        directory renames (both atomic on the same filesystem) rather than
        overwriting target_dir's files in place -- so at every instant either the
        old or the new checkpoint is a complete, valid target_dir, never a
        half-written mix of both. old/ is removed only after the swap, i.e. after a
        new complete checkpoint already exists on disk, not before."""
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

        old_dir = ckpt_root / f"_{target_dir.name}_old"
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if target_dir.exists():
            target_dir.rename(old_dir)
        tmp_dir.rename(target_dir)
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

        self._wait_for_pending_save()  # also ensures _pick_target_slot sees a stable, fully-written disk state
        target_dir = self._pick_target_slot()
        self._pending_save = self._io_pool.submit(
            self._write_checkpoint_to_disk, self.ckpt_root, target_dir, model_sd, optim_sd, sched_sd, rng_state, meta
        )
        return target_dir

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
                log_fn(f"[checkpoint] sparse full checkpoint at step {step} -> {ckpt_dir}")

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
        should get the real final-step weights, not an approximation of them. Goes
        through the same rotating-slot selection as any other full save (module
        docstring) -- training/finalize_checkpoint.py is what decides afterward
        whether this final-step checkpoint or an earlier rotating one is actually
        the best by val_ppl."""
        ckpt_dir = self._save_full(step, kind="final", model=model, optimizer=optimizer, scheduler=scheduler, extra_meta=extra_meta or {})
        if log_fn:
            log_fn(f"[checkpoint] final full checkpoint at step {step} -> {ckpt_dir}")
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
