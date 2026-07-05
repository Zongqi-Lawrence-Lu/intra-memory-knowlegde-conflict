"""Window-level reweighting: the fallback that works against the M4 training
loop as it exists today (training/data.py), without needing document-boundary
metadata from the M2 corpus packer.

training_time.reweighting computes document-level weights via near-duplicate
clustering, which is the more sensitive detector (catches paraphrase-level
duplicates) but needs a way to join doc_id -> token-window back into
training.data.PackedTokenDataset, which does not exist yet (no doc-boundary
metadata is packed alongside meta.json/*.bin). This module instead detects
*exact*-duplicate token windows directly against whatever Dataset
training.data.build_datasets returns (PackedTokenDataset or, for smoke-testing,
DummyTokenDataset) by hashing each window's input-token sequence. It only
catches exact repeats, not paraphrases, but requires no corpus metadata that
doesn't exist yet. Once M2 records per-window doc_id, prefer joining
training_time.reweighting's document-level weights instead.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def _window_hash(x: torch.Tensor) -> bytes:
    return hashlib.sha1(x.numpy().tobytes()).digest()


def compute_window_weights(dataset: Dataset, min_weight: float = 0.05) -> np.ndarray:
    """weight[i] = 1 / count(hash(x_i)), floored at `min_weight` so a window that
    recurs often is downweighted but never zeroed out entirely.

    O(n) over the dataset with one Python-level __getitem__ call per window; fine
    for the dev-scale corpora this project uses so far, but will need batching /
    vectorized hashing if run against a full-scale (hundreds-of-millions-of-token)
    packed corpus.
    """
    hashes = [_window_hash(dataset[i][0]) for i in range(len(dataset))]

    counts: dict[bytes, int] = {}
    for h in hashes:
        counts[h] = counts.get(h, 0) + 1

    return np.array([max(1.0 / counts[h], min_weight) for h in hashes], dtype=np.float64)


def build_weighted_sampler(weights: np.ndarray, num_samples: Optional[int] = None) -> WeightedRandomSampler:
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=num_samples or len(weights),
        replacement=True,
    )
