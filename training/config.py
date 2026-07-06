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
    # lr/warmup/max_steps re-derived for batch_size=192 (DataConfig, up from the
    # Physics-of-LM-anchored 48, for A100 GPU-utilization reasons -- see
    # experimental_plans.tex's Optimization-section revision note) rather than
    # kept at the reference values for a 4x-smaller batch:
    #   - lr: sqrt-scaled (Malladi et al. 2022's SDE-derived rule for Adam-family
    #     optimizers, as opposed to SGD's linear scaling rule), 6e-4 * sqrt(4) = 1.2e-3.
    #   - warmup_steps: divided by 4 so warmup covers the same *token* budget
    #     (500 * 98,304 tok/step == 2000 * 24,576 tok/step, exactly).
    #   - max_steps: re-solved for the fixed 2.5e9-token target at the new,
    #     4x-larger tokens/step (see DataConfig.batch_size).
    lr: float = 1.2e-3
    weight_decay: float = 0.01  # §1.3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 25_425  # 2.5e9 / (192 * 512) tokens/step, ~= Chinchirlla target
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
    # 192, not the Physics-of-LM-anchored 48: at seq_len=512 with SDPA/flash
    # attention (training/model.py), batch=48 leaves an A100-80GB deeply
    # under-utilized (order of single-digit GB of activation memory out of 80GB),
    # which lowers achieved MFU relative to the throughput figure
    # experimental_plans.tex's 12h wall-clock budget is based on. Raising batch
    # size (with OptimConfig's corresponding lr/warmup/max_steps recompute) trades
    # exact hyperparameter parity with the reference setup for better GPU
    # utilization under the fixed 12h ceiling. Verify this actually fits 80GB on
    # the first real cluster run -- not tested locally (no GPU here) -- and back
    # off if not.
    batch_size: int = 192
    grad_accum_steps: int = 1
    num_workers: int = 2


@dataclasses.dataclass
class CheckpointConfig:
    """Dual-cadence policy, experimental_plans.tex §1.4."""

    # Sparse/global cadence: full checkpoint (weights + optimizer + RNG state) every
    # sparse_interval_steps, sized so a crash loses ~10-15 min of compute. Computed
    # from the *reference* 178,000 tok/s (nanoGPT single-A100 benchmark cited in
    # experimental_plans.tex's Optimization section) at batch=192/n_positions=512
    # (98,304 tok/step) -> ~1.81 steps/sec -> ~1,086-1,629 steps for 10-15 min;
    # 1200 sits in that range. Still provisional -- that reference figure assumes
    # torch.compile/fused-optimizer throughput this pipeline may or may not hit;
    # re-tune once real per-step throughput is measured on the cluster (see
    # training/README note).
    sparse_interval_steps: int = 1200
    keep_last_n_sparse: int = 3  # rolling retention; -1 keeps all

    # Dense/event-triggered cadence: denser checkpointing in a window around each
    # fact-injection step. Empty until M2 supplies the actual injection schedule.
    # interval/window divided by 4 alongside DataConfig.batch_size's 4x increase,
    # so both stay pinned to the same *token*-space resolution around an injection
    # event regardless of batch size.
    injection_steps: list[int] = dataclasses.field(default_factory=list)
    dense_interval_steps: int = 50
    dense_window_steps: int = 250  # +/- window around each injection step
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
    # 125, not 500: divided by 4 alongside DataConfig.batch_size's 4x increase, so
    # eval fires at the same *token*-space cadence regardless of batch size
    # (125 * 98,304 == 500 * 24,576 tokens between evals, exactly).
    eval_interval_steps: int = 125
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
    # torch.compile(raw_model) when running on CUDA (training/train.py) -- the
    # main lever for closing the gap to the 178,000 tok/s reference throughput
    # experimental_plans.tex's wall-clock estimate is based on. Left as a flag
    # (not hardcoded on) so it can be switched off without a code change if a
    # graph-break/compile issue surfaces on the first real cluster run --
    # unverified locally, no GPU here. --smoke-test forces this off regardless
    # (training/train.py:apply_smoke_test_overrides), since a smoke test is meant
    # to check loop mechanics cheaply, not exercise compile.
    compile: bool = True


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
