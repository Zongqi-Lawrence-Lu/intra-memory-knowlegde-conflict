"""Training configuration schema.

Mirrors experimental_plans.tex §1 (Model and Training Pipeline). Loaded from a YAML
file (see training/configs/default.yaml) rather than hardcoded, so sweeps over the
insertion-schedule / seed axes (§1.6) can vary a single field without code changes.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import yaml


@dataclasses.dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_positions: int = 512  # §1.1: capped at 512 (vs GPT-2 default 1024)
    vocab_size: int = 50257  # standard pretrained GPT-2 BPE tokenizer, reused as-is
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1


@dataclasses.dataclass
class OptimConfig:
    lr: float = 6e-4
    weight_decay: float = 0.01  # §1.3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 50_000  # §1.3: Physics of LM 3.1 anchor
    lr_schedule: str = "cosine"  # §1.3
    min_lr_ratio: float = 0.1  # cosine decays to min_lr_ratio * lr, not to zero


@dataclasses.dataclass
class DataConfig:
    # Path to a directory of packed-token shards produced by preprocess/ (see
    # training/data.py:PackedTokenDataset). Left unset until M2 (corpus generator,
    # experimental_plans.tex §1.2) lands; training.py falls back to a synthetic
    # dummy dataset when this is None, purely to exercise the training loop.
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    batch_size: int = 48  # §1.3: Physics of LM 3.1 anchor
    grad_accum_steps: int = 1
    num_workers: int = 2


@dataclasses.dataclass
class CheckpointConfig:
    """Dual-cadence policy, experimental_plans.tex §1.4."""

    # Sparse/global cadence: full checkpoint (weights + optimizer + RNG state) every
    # sparse_interval_steps, sized so a crash loses ~10-15 min of compute. This must be
    # re-tuned once real per-step throughput is measured (see training/README note);
    # the default below is a placeholder pending that profiling.
    sparse_interval_steps: int = 500
    keep_last_n_sparse: int = 3  # rolling retention; -1 keeps all

    # Dense/event-triggered cadence: denser checkpointing in a window around each
    # fact-injection step. Empty until M2 supplies the actual injection schedule.
    injection_steps: list[int] = dataclasses.field(default_factory=list)
    dense_interval_steps: int = 200
    dense_window_steps: int = 1000  # +/- window around each injection step
    # Of the dense checkpoints inside a window, keep full weights only for a
    # stratified subset (first K + random sample of the rest); metrics are kept for
    # all of them regardless. See experimental_plans.tex §1.4.
    dense_full_weight_first_k: int = 2
    dense_full_weight_sample_rate: float = 0.1


@dataclasses.dataclass
class EvalConfig:
    # Held-out WikiText-103 perplexity, experimental_plans.tex §1.5. Per-fact recall
    # is not implemented here -- it needs fact metadata from the M2 corpus generator
    # and belongs to the shared eval harness (M3), not the training loop itself.
    eval_interval_steps: int = 500
    eval_batches: int = 50  # cap on val batches per eval, for cheap periodic checks


@dataclasses.dataclass
class RunConfig:
    run_name: str = "gpt2-small-dev"
    seed: int = 0
    output_dir: str = "output"  # gitignored: checkpoints, optimizer state, raw logs
    results_dir: str = "results"  # tracked: metrics json, config snapshot, summary
    device: str = "cuda"
    dtype: str = "bfloat16"  # §1.3: bf16 mixed precision
    log_interval_steps: int = 20
    resume: bool = True  # auto-resume from latest sparse checkpoint if present


@dataclasses.dataclass
class TrainingConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = dataclasses.field(default_factory=CheckpointConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            optim=OptimConfig(**raw.get("optim", {})),
            data=DataConfig(**raw.get("data", {})),
            checkpoint=CheckpointConfig(**raw.get("checkpoint", {})),
            eval=EvalConfig(**raw.get("eval", {})),
            run=RunConfig(**raw.get("run", {})),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
