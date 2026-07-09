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
    # shuffle=True/overlapping=True (both defaults) preserve the exact prior training
    # behavior for existing configs. Setting both False is what makes an occurrence's
    # absolute token position (results/occurrence_log.json) map to a fixed training
    # step at all -- see training/data.py:PackedTokenDataset's overlapping docstring
    # and training/injection_schedule.py -- appropriate only for a corpus sized for
    # ~one epoch (experimental_plans.tex S1.9), which full_run.yaml's is.
    shuffle: bool = True
    overlapping: bool = True


@dataclasses.dataclass
class CheckpointConfig:
    """Rotating multi-slot policy (2026-07-09 revision, superseding the prior
    single-slot-overwrite design, which itself superseded an earlier
    sparse-rotation + dense-stratified-retention design, experimental_plans.tex
    §1.4). Up to `num_checkpoint_slots` full checkpoints (weights + optimizer +
    RNG state) live on disk at once, at output/<run_name>/checkpoints/slot_*/ --
    once that many exist, the next save overwrites whichever slot holds the
    *oldest* step, never the newest, so a divergence discovered late in training
    (T=80/T=1280's loss spikes, see memory/recall_diagnosis_2026-07-08.md finding
    1's sibling issue) can still be rolled back to a healthy pre-spike checkpoint
    instead of only ever having the single most-recent (possibly already-broken)
    one on disk. training/finalize_checkpoint.py prunes this back down to exactly
    one (the best by held-out val_ppl) once a run is done being trained/rolled
    back -- the rotation is a during-training safety margin, not the intended
    steady state. Each save still swaps a slot in via a directory rename
    (training/checkpoint.py:CheckpointManager._write_checkpoint_to_disk), not an
    in-place file overwrite, so a crash mid-write can't corrupt that slot's
    previous contents.
    Dense/event-triggered checkpointing still never captures full weights (only
    its metrics-only markers) -- unrelated to slot rotation, unchanged from the
    prior revision."""

    # Sparse/global cadence: full checkpoint (weights + optimizer + RNG state) every
    # sparse_interval_steps, sized so a crash loses ~30 min of compute (loosened from
    # an earlier ~10-15min bound -- explicit request for a sparser cadence, less
    # checkpoint-write overhead/storage churn). Computed from the *reference*
    # 178,000 tok/s (nanoGPT single-A100 benchmark cited in experimental_plans.tex's
    # Optimization section) at batch=192/n_positions=512 (98,304 tok/step) ->
    # ~1.81 steps/sec -> ~3,259 steps for 30min; 3200 rounds down to keep the bound
    # at or under 30min. Still provisional -- that reference figure assumes
    # torch.compile/fused-optimizer throughput this pipeline may or may not hit at
    # batch=192 specifically; training/configs/full_run.yaml uses a value re-derived
    # from real measured throughput at its own batch_size=128 instead (see that
    # file's comment and experimental_plans.tex's checkpointing-section revision
    # note).
    sparse_interval_steps: int = 3200

    # Rotating slot count -- explicit user request (2026-07-09): "up to three
    # non-overwriting checkpoints throughout training (later ones can overwrite
    # the obsolete ones)". 3 covers a sparse_interval_steps-scale rollback window
    # (~90min of compute at the default cadence) at a small, bounded, constant
    # storage cost regardless of run length -- the same motivation the prior
    # single-slot design cited, just with enough headroom to survive a
    # discovered-late divergence.
    num_checkpoint_slots: int = 3

    # Dense/event-triggered cadence: denser metrics logging in a window around each
    # fact-injection step (no full weights, see class docstring). Left empty and
    # computed at runtime (train.py, via training/injection_schedule.py) rather than
    # hardcoded in YAML, whenever occurrence_log_path is set below -- explicit YAML
    # values are only used as a manual override for cases without an occurrence log
    # (e.g. --smoke-test). interval/window divided by 4 alongside DataConfig.
    # batch_size's 4x increase, so both stay pinned to the same *token*-space
    # resolution around an injection event regardless of batch size.
    injection_steps: list[int] = dataclasses.field(default_factory=list)
    # Path to preprocess/assemble_corpus.py's occurrence_log.json. When set (and
    # data.shuffle/data.overlapping are both False -- see DataConfig), train.py
    # derives injection_steps automatically instead of using the (then-ignored)
    # injection_steps field above. None (default) preserves prior behavior:
    # injection_steps comes only from YAML (empty unless set explicitly), and the
    # dense cadence stays inert.
    occurrence_log_path: Optional[str] = None
    dense_interval_steps: int = 50
    dense_window_steps: int = 250  # +/- window around each injection step


@dataclasses.dataclass
class StabilityConfig:
    """Divergence mitigation (2026-07-09), added after T=80 and T=1280 both hit an
    irrecoverable mid-run loss spike (loss jumping from ~4-5 to 8-15+ within a few
    hundred steps and never coming back down for the rest of training, see
    memory/recall_diagnosis_2026-07-08.md and the T=80/T=320/T=1280 status
    conversation) while T=320, same hyperparameters, trained cleanly the whole
    way -- i.e. a stochastic per-run instability, not a config problem, so the fix
    is runtime detection/recovery rather than a hyperparameter change. Two
    independent layers:
      (1) per-step spike guard -- skips the optimizer update (but not the backward
          pass, which is needed either way to know the loss) on a single-step loss
          spike or non-finite loss, so one bad batch can't corrupt Adam's moment
          estimates the way it appears to have in both bad runs.
      (2) sustained-divergence rollback -- if held-out val_ppl stays badly elevated
          across several consecutive evals (the per-step guard alone wasn't enough
          to prevent or recover from the drift), reloads model/optimizer/scheduler/
          RNG state from the most recent rotating checkpoint (CheckpointConfig) and
          continues training from there, discarding the diverged trajectory instead
          of running the remaining budget on a collapsed model."""

    spike_skip_multiplier: float = 4.0  # skip the optimizer step if loss > this * rolling mean
    spike_rolling_window: int = 50  # steps of loss history the rolling mean is computed over
    spike_min_history: int = 20  # steps of history required before the guard can trigger (avoids reacting to the initial-loss transient)

    rollback_val_ppl_multiplier: float = 2.0  # trigger candidate: val_ppl > this * best-val_ppl-so-far
    rollback_patience_evals: int = 3  # consecutive bad evals required before actually rolling back
    max_rollbacks: int = 5  # safety cap -- see stop_on_exhausted_rollbacks below for what
    # happens once this many rollbacks have fired in one run.

    # 2026-07-09 revision, after T=80-stable/T=1280-stable both burned through
    # max_rollbacks and kept re-diverging near the same point every time: restoring
    # model/optimizer/RNG state alone doesn't remove whatever made that region of
    # training fragile in the first place (elevated LR, activation/logit growth --
    # see memory/t80_corpus_repetition_instability.md and Wortsman et al. 2023),
    # so replaying the identical LR trajectory from the identical restored state
    # tends to just reproduce the identical failure. Two changes:
    #   (1) each successful rollback permanently multiplies the optimizer's LR by
    #       rollback_lr_decay (compounding across rollbacks, floored at
    #       min_lr_penalty) -- same "rollback + lower LR" pattern used operationally
    #       for PaLM/OPT-scale spikes, so the post-rollback trajectory actually
    #       differs from the one that just failed instead of retracing it.
    #   (2) once max_rollbacks is exhausted, stop training immediately (rather than
    #       continuing to burn the remaining step budget on a model that's already
    #       shown it can't recover) -- this also protects the rotating checkpoint
    #       slots (CheckpointConfig) from being evicted by further sparse saves of
    #       the diverged trajectory, which would otherwise risk rotating every good
    #       pre-divergence checkpoint out before finalize_checkpoint.py ever runs.
    rollback_lr_decay: float = 0.7  # multiplicative LR penalty applied on each rollback
    min_lr_penalty: float = 0.1  # floor for the cumulative penalty -- never decay LR below 10% of schedule
    stop_on_exhausted_rollbacks: bool = True


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
    stability: StabilityConfig = dataclasses.field(default_factory=StabilityConfig)
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
            stability=StabilityConfig(**raw.get("stability", {})),
            eval=EvalConfig(**raw.get("eval", {})),
            run=RunConfig(**raw.get("run", {})),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
