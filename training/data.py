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


class PackedTokenDataset(Dataset):
    """Fixed-length blocks sampled from a flat memmap of token ids."""

    def __init__(self, data_dir: str | Path, block_size: int):
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
        # number of non-overlapping (input, target) windows available per shard
        self._lengths = [max(0, len(s) - block_size - 1) for s in self.shards]
        self._cum_lengths = np.cumsum(self._lengths)

    def __len__(self) -> int:
        return int(self._cum_lengths[-1]) if len(self._cum_lengths) else 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = int(np.searchsorted(self._cum_lengths, idx, side="right"))
        offset = idx - (self._cum_lengths[shard_idx - 1] if shard_idx > 0 else 0)
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
) -> tuple[Dataset, Dataset]:
    """Returns (train_dataset, val_dataset). Falls back to DummyTokenDataset when no
    path is configured (see DataConfig.train_path docstring in training/config.py)."""
    if train_path is None:
        train_ds = DummyTokenDataset(vocab_size, block_size, num_examples=2000, seed=seed)
        val_ds = DummyTokenDataset(vocab_size, block_size, num_examples=200, seed=seed + 1)
        return train_ds, val_ds

    train_ds = PackedTokenDataset(train_path, block_size)
    val_ds = PackedTokenDataset(val_path or train_path, block_size)
    return train_ds, val_ds
