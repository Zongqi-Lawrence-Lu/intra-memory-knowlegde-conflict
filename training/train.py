"""GPT-2-small-from-scratch training loop (experimental_plans.tex §1).

Single-device only (default single-A40 per CLAUDE.md §2); multi-GPU is a noted
escalation path, not implemented here since no run has yet shown the need.

Usage:
    python -m training.train --config training/configs/default.yaml
    python -m training.train --config training/configs/default.yaml --smoke-test

`--smoke-test` runs a handful of steps on the DummyTokenDataset (see training/data.py)
on CPU/CUDA, purely to exercise the loop end-to-end (shapes, checkpointing, resume,
logging) before any real corpus exists.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.checkpoint import CheckpointManager
from training.config import TrainingConfig
from training.data import build_datasets
from training.eval import compute_perplexity
from training.logging_utils import setup_logging
from training.model import build_model, count_parameters

DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def build_optimizer(model, cfg, device: str) -> torch.optim.AdamW:
    # standard GPT-2 recipe: no weight decay on biases / 1D params (LayerNorm, biases)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    param_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused=True collapses the whole param-group update into a handful of fused CUDA
    # kernels instead of one kernel launch per tensor (~150 tensors for GPT-2-small) --
    # a straight speedup with identical math, only available on CUDA.
    fused = device.startswith("cuda")
    return torch.optim.AdamW(
        param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, fused=fused
    )


def build_scheduler(optimizer, cfg) -> torch.optim.lr_scheduler.LambdaLR:
    warmup, max_steps, min_ratio = cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if cfg.lr_schedule != "cosine":
            return 1.0
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def infinite_loader(loader: DataLoader):
    """Cycles the (shuffled) loader forever. This is what lets total training tokens
    (optim.max_steps * data.batch_size * model.n_positions, experimental_plans.tex
    S1.9) exceed the on-disk corpus size (S1.7's total_tokens): once one pass over
    PackedTokenDataset's windows is exhausted, this reshuffles and starts again
    rather than stopping -- the two token counts are independent quantities, not
    the same number under different names."""
    while True:
        for batch in loader:
            yield batch


def apply_smoke_test_overrides(cfg: TrainingConfig) -> None:
    """Shrinks a config in-place for --smoke-test: cheap enough to run on CPU, purely
    to check loop mechanics (shapes, checkpointing, resume, logging), not that
    GPT-2-small itself trains well. Mutates `cfg`; idempotent."""
    cfg.model.n_layer = 2
    cfg.model.n_embd = 128
    cfg.model.n_head = 2
    cfg.model.n_positions = 64
    cfg.optim.max_steps = min(cfg.optim.max_steps, 40)
    cfg.optim.warmup_steps = min(cfg.optim.warmup_steps, 5)
    cfg.checkpoint.sparse_interval_steps = 10
    cfg.eval.eval_interval_steps = 10
    cfg.run.log_interval_steps = 5
    cfg.data.train_path = None  # force DummyTokenDataset
    cfg.data.val_path = None
    cfg.data.batch_size = 4
    cfg.data.num_workers = 0  # avoid multiprocessing overhead/hangs for a tiny run
    cfg.run.compile = False  # a smoke test checks loop mechanics cheaply; compiling a
    # throwaway 2-layer model for 40 steps only adds compile-time overhead, and keeps
    # this path decoupled from verifying torch.compile itself (do that deliberately on
    # a real run instead, e.g. training/configs/pipeline_test.yaml).


def train(cfg: TrainingConfig, smoke_test: bool = False, build_sampler=None) -> None:
    """`build_sampler`, if given, is called as `build_sampler(train_ds)` after the
    (possibly smoke-test-shrunk) train dataset is built, and must return a
    torch.utils.data.Sampler used in place of uniform shuffling -- the hook
    training_time/train_mitigated.py (M7) uses to layer a reweighting mitigation
    onto this loop without duplicating it. Default (None) preserves prior
    behavior (`shuffle=True`) exactly."""
    if smoke_test:
        apply_smoke_test_overrides(cfg)

    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    dtype = DTYPE_MAP[cfg.run.dtype]

    logger = setup_logging(cfg.run.output_dir, cfg.run.run_name)
    logger.info(f"run_name={cfg.run.run_name} device={device} dtype={cfg.run.dtype}")

    results_dir = Path(cfg.run.results_dir) / cfg.run.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(results_dir / "config_used.yaml")

    on_cuda = device.startswith("cuda")

    raw_model = build_model(cfg.model).to(device)
    logger.info(f"model parameters: {count_parameters(raw_model):,}")

    # torch.compile wraps forward/backward in fused, autotuned kernels -- the main
    # lever for closing the gap to the 178,000 tok/s reference throughput
    # experimental_plans.tex's wall-clock budget is based on. Checkpointing
    # (CheckpointManager, optimizer construction) always operates on `raw_model`,
    # never the compiled wrapper, matching nanoGPT's own pattern for this exact
    # benchmark: a compiled module's state_dict can pick up an "_orig_mod." key
    # prefix depending on torch version, which would silently break resume.
    compile_enabled = cfg.run.compile and on_cuda
    model = torch.compile(raw_model) if compile_enabled else raw_model
    if compile_enabled:
        logger.info("torch.compile enabled")

    train_ds, val_ds = build_datasets(
        cfg.data.train_path, cfg.data.val_path, cfg.model.vocab_size, cfg.model.n_positions, seed=cfg.run.seed
    )
    sampler = build_sampler(train_ds) if build_sampler is not None else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        drop_last=True,
        pin_memory=on_cuda,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=on_cuda
    )
    train_iter = infinite_loader(train_loader)

    optimizer = build_optimizer(raw_model, cfg.optim, device)
    scheduler = build_scheduler(optimizer, cfg.optim)

    ckpt_mgr = CheckpointManager(cfg.checkpoint, cfg.run.output_dir, cfg.run.results_dir, cfg.run.run_name)

    start_step = 0
    if cfg.run.resume:
        start_step = ckpt_mgr.resume(raw_model, optimizer, scheduler, device)
        if start_step > 0:
            logger.info(f"resumed from step {start_step}")

    model.train()
    tokens_per_step = cfg.data.batch_size * cfg.model.n_positions * cfg.data.grad_accum_steps
    t_last_log = time.time()
    best_val_ppl = float("inf")

    for step in range(start_step + 1, cfg.optim.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.zeros((), device=device)
        for _ in range(cfg.data.grad_accum_steps):
            x, y = next(train_iter)
            x, y = x.to(device, non_blocking=on_cuda), y.to(device, non_blocking=on_cuda)
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
                out = model(input_ids=x, labels=y)
                loss = out.loss / cfg.data.grad_accum_steps
            loss.backward()
            # stays a GPU tensor -- accum_loss.item() below (only called at the
            # log cadence) is the only sync point in the step loop; calling .item()
            # here on every step/micro-step would force a CPU-GPU sync every step
            # regardless of whether that step's loss ever gets read, stalling the
            # pipelining torch.compile/async prefetch are meant to enable.
            accum_loss += loss.detach()

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.run.log_interval_steps == 0:
            if on_cuda:
                torch.cuda.synchronize()  # make elapsed/tok-per-sec reflect completed GPU work
            elapsed = time.time() - t_last_log
            tok_per_sec = (tokens_per_step * cfg.run.log_interval_steps) / max(elapsed, 1e-8)
            lr_now = scheduler.get_last_lr()[0]
            loss_value = accum_loss.item()
            logger.info(f"step={step} loss={loss_value:.4f} lr={lr_now:.3e} tok/s={tok_per_sec:.0f}")
            ckpt_mgr.log_metrics({"step": step, "loss": loss_value, "lr": lr_now, "tokens_seen": step * tokens_per_step})
            t_last_log = time.time()

        if step % cfg.eval.eval_interval_steps == 0:
            val_ppl = compute_perplexity(model, val_loader, device, dtype, max_batches=cfg.eval.eval_batches)
            logger.info(f"step={step} val_ppl={val_ppl:.3f}")
            ckpt_mgr.log_metrics({"step": step, "val_ppl": val_ppl})
            best_val_ppl = min(best_val_ppl, val_ppl)

        ckpt_mgr.maybe_save(step, raw_model, optimizer, scheduler, log_fn=logger.info)

    ckpt_mgr.shutdown()  # block until the last async checkpoint write actually finishes
    ckpt_mgr.write_run_summary(
        {
            "run_name": cfg.run.run_name,
            "total_steps": cfg.optim.max_steps,
            "tokens_seen": cfg.optim.max_steps * tokens_per_step,
            "best_val_ppl": best_val_ppl,
            "model_params": count_parameters(raw_model),
        }
    )
    logger.info("training complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/configs/default.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
