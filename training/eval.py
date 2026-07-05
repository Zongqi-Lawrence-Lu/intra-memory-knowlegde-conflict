"""Minimal eval harness hook used during training (experimental_plans.tex §1.5).

Only held-out perplexity is implemented here -- it is the sanity check the training
loop itself needs at every checkpoint. Per-fact recall (tracked per side of each
conflicting pair) requires fact metadata from the M2 corpus generator and a shared
probing-prompt runner (IMPLEMENTATION.md M3); that belongs in its own module once
the corpus exists, not bolted onto the training loop ad hoc.
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_perplexity(model, loader: DataLoader, device, dtype, max_batches: int | None = None) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
            out = model(input_ids=x, labels=y)
        n_tokens = y.numel()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    if total_tokens == 0:
        return float("nan")
    mean_loss = total_loss / total_tokens
    return math.exp(min(mean_loss, 20.0))  # cap to avoid overflow on an untrained model
