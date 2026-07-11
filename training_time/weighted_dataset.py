"""Window-level reweighting: the fallback that works against the M4 training
loop as it exists today (training/data.py), without needing document-boundary
metadata from the M2 corpus packer.

training_time.reweighting computes document-level weights via near-duplicate
clustering, which is the more sensitive detector (catches paraphrase-level
duplicates) but needs a way to join doc_id -> token-window back into
training.data.PackedTokenDataset -- training_time.corpus_export now builds that
join (real corpus -> {doc_id, text, tag} jsonl, exact-token EOS-delimited
documents cross-referenced against results/occurrence_log.json), so prefer
training_time.reweighting's document-level weights where a document-level join
matters (paraphrase-level duplicates). This module instead detects *exact*-
duplicate token windows directly against whatever Dataset
training.data.build_datasets returns (PackedTokenDataset or, for smoke-testing,
DummyTokenDataset) by hashing each window's input-token sequence -- the right
choice when no document join is needed at all, e.g. reweighting the packed
windows training/train.py actually samples from, whose boundaries don't align
with document boundaries under overlapping=True.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Odd 64-bit multiplier for the Horner's-rule polynomial hash below (arbitrary
# non-trivial odd constant -- oddness keeps successive powers from degenerating
# to zero under mod-2**64 wraparound). Fixed, not seeded per-call, so weights
# are reproducible across runs/processes.
_HASH_BASE = np.uint64(1099511628211)


def _polynomial_hash_batch(batch: np.ndarray) -> np.ndarray:
    """Vectorized (n, block_size) int64 token batch -> (n,) uint64 hash, one numpy
    expression over the whole batch rather than a Python-level hashlib call per row.
    Standard Rabin-Karp-style polynomial hash, mod 2**64 via numpy's silent integer
    wraparound (the same accepted-approximate-hash tradeoff training_time/dedup.py's
    MinHash already makes elsewhere in this module family -- collision probability
    at this project's real scale, ~1e7 windows into a 2**64 space, is ~1e-6 by the
    birthday bound, negligible for a reweighting heuristic)."""
    n, block_size = batch.shape
    powers = _HASH_BASE ** np.arange(block_size, dtype=np.uint64)
    return (batch.astype(np.uint64) * powers).sum(axis=1, dtype=np.uint64)


def compute_window_weights(
    dataset: Dataset, min_weight: float = 0.05, batch_size: int = 4096, num_workers: int = 0
) -> np.ndarray:
    """weight[i] = 1 / count(hash(x_i)), floored at `min_weight` so a window that
    recurs often is downweighted but never zeroed out entirely.

    Vectorized and memory-bounded for real corpus scale (the project's real
    on-disk corpora are ~4e9 tokens / 512 block_size =~ 7.8e6 windows): batches
    through `dataset` via a DataLoader (batched/parallel __getitem__ instead of
    one Python-level call per window) and hashes each batch in one numpy
    expression (_polynomial_hash_batch) instead of one hashlib call per window.
    Only a small per-window uint64 hash array (~8 bytes/window, ~62MB at 7.8e6
    windows) is ever held across the whole dataset -- raw token windows are
    processed one batch at a time and discarded, never materialized in full
    (the same batch-size-independent-of-corpus-size discipline as the
    RandomSampler OOM fix this project already applied to training/train.py).
    """
    n = len(dataset)
    hashes = np.empty(n, dtype=np.uint64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    pos = 0
    for x_batch, _ in loader:
        arr = x_batch.numpy()
        h = _polynomial_hash_batch(arr)
        hashes[pos : pos + len(h)] = h
        pos += len(h)

    _, inverse, group_counts = np.unique(hashes, return_inverse=True, return_counts=True)
    return np.maximum(1.0 / group_counts[inverse], min_weight).astype(np.float64)


def build_weighted_sampler(weights: np.ndarray, num_samples: Optional[int] = None) -> WeightedRandomSampler:
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=num_samples or len(weights),
        replacement=True,
    )
