"""Token-sequence datasets for the training loop.

Two implementations:

- `PackedTokenDataset` reads pre-tokenized shards from disk: a `meta.json`
  ({"dtype": "uint16", "vocab_size": ...}) plus one or more `*.bin` files, each a flat
  memmap-able array of token ids (the nanoGPT-style packing convention). This is the
  format the M2 corpus generator (preprocess/, not yet built) is expected to emit --
  the training loop only depends on this on-disk contract, not on how the corpus was
  assembled, so conflict-injection logic can be added later without touching this file.
- `DummyTokenDataset` generates random token ids on the fly. It exists purely to
  exercise the training loop (shapes, checkpointing, logging, restart) before real
  data exists; it is not a stand-in for the synthetic corpus itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def corpus_token_count(data_dir: str | Path) -> int:
    """Total on-disk token count across a preprocess/ output dir's shard(s) -- reads
    meta.json's dtype and sums *.bin file sizes directly, no memmap needed. Used by
    train.py's startup corpus-size check (DataConfig.min_corpus_tokens) to fail loudly
    before infinite_loader (training/train.py) would otherwise silently repeat-sample
    over an undersized/incompletely-assembled corpus instead of erroring -- see
    memory/t80_corpus_repetition_instability.md for the incident (a truncated ~1.0B-
    token corpus silently trained for ~2.5 epochs instead of the intended single pass,
    with nothing catching the mismatch until diagnosed after the fact) that motivated
    this check."""
    data_dir = Path(data_dir)
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    dtype_size = np.dtype(meta.get("dtype", "uint16")).itemsize
    shard_paths = sorted(data_dir.glob("*.bin"))
    return sum(p.stat().st_size for p in shard_paths) // dtype_size


class PackedTokenDataset(Dataset):
    """Windows sampled from a flat memmap of token ids.

    `overlapping=True` (default, prior behavior unchanged): every single-token-shifted
    offset is a valid window, the nanoGPT-style convention that pairs naturally with
    `shuffle=True` + resampling-with-replacement-across-epochs (training/train.py's
    infinite_loader) -- appropriate when total training tokens are meant to exceed the
    on-disk corpus many times over.

    `overlapping=False`: windows are non-overlapping block_size-token chunks (window i
    starts at token i*block_size). This is what makes a token position map to a
    training step at all: under `overlapping=True` + `shuffle=True`, a given occurrence
    event's absolute token position corresponds to a randomly-ordered window with no
    fixed step, so training/injection_schedule.py's step derivation requires this mode
    (paired with shuffle=False -- see training/train.py) to hold. Only sensible for a
    ~single-epoch run (on-disk tokens ~= total training tokens, experimental_plans.tex
    S1.9's corpus-vs-training-tokens distinction): consuming it non-overlapping avoids
    the redundant, nearly-identical-content batches that sequential *overlapping*
    windows (offsets 0,1,2,...) would otherwise produce.
    """

    def __init__(self, data_dir: str | Path, block_size: int, overlapping: bool = True):
        data_dir = Path(data_dir)
        meta_path = data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found -- expected a preprocess/ output directory "
                f"with meta.json + *.bin shard(s)."
            )
        with open(meta_path) as f:
            self.meta = json.load(f)

        dtype = np.dtype(self.meta.get("dtype", "uint16"))
        shard_paths = sorted(data_dir.glob("*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"No *.bin shards found in {data_dir}")

        self.shards = [np.memmap(p, dtype=dtype, mode="r") for p in shard_paths]
        self.block_size = block_size
        self.overlapping = overlapping
        self._stride = 1 if overlapping else block_size
        usable = [max(0, len(s) - block_size - 1) for s in self.shards]
        if overlapping:
            self._lengths = usable  # exact prior formula/behavior, unchanged
        else:
            self._lengths = [(u // self._stride + 1) if u > 0 else 0 for u in usable]
        self._cum_lengths = np.cumsum(self._lengths) if self._lengths else np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return int(self._cum_lengths[-1]) if len(self._cum_lengths) else 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = int(np.searchsorted(self._cum_lengths, idx, side="right"))
        within_idx = idx - (self._cum_lengths[shard_idx - 1] if shard_idx > 0 else 0)
        offset = within_idx * self._stride
        shard = self.shards[shard_idx]
        chunk = shard[offset : offset + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class DummyTokenDataset(Dataset):
    """Random token sequences, for smoke-testing the training loop only."""

    def __init__(self, vocab_size: int, block_size: int, num_examples: int = 2000, seed: int = 0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_examples = num_examples
        self._gen = torch.Generator().manual_seed(seed)
        self._data = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_examples, block_size + 1),
            generator=self._gen,
        )

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self._data[idx]
        return chunk[:-1], chunk[1:]


def build_datasets(
    train_path: Optional[str],
    val_path: Optional[str],
    vocab_size: int,
    block_size: int,
    seed: int = 0,
    overlapping: bool = True,
) -> tuple[Dataset, Dataset]:
    """Returns (train_dataset, val_dataset). Falls back to DummyTokenDataset when no
    path is configured (see DataConfig.train_path docstring in training/config.py).
    `overlapping` only affects the train dataset -- val is always evaluated over
    non-overlapping chunks (PackedTokenDataset's default `overlapping=True` param name
    refers to windowing, not to whether eval batches revisit data; a small, fixed val
    set is read in full regardless of stride)."""
    if train_path is None:
        train_ds = DummyTokenDataset(vocab_size, block_size, num_examples=2000, seed=seed)
        val_ds = DummyTokenDataset(vocab_size, block_size, num_examples=200, seed=seed + 1)
        return train_ds, val_ds

    train_ds = PackedTokenDataset(train_path, block_size, overlapping=overlapping)
    val_ds = PackedTokenDataset(val_path or train_path, block_size, overlapping=False)
    return train_ds, val_ds
