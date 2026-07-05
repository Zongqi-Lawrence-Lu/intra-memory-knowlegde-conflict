"""Dataset deduplication (M7 training-time baseline).

Scientific framing (see experimental_plans.tex Sec.1.2): the backbone corpus is
already deduplicated by construction, and the controlled repetition of each
conflicting fact is the project's main independent variable. This module asks
what happens if a *standard*, content-blind dedup pipeline -- the kind normally
run for corpus hygiene (Lee et al. 2022-style exact + near-duplicate removal) --
is applied to the combined (backbone + injected) corpus without any special
knowledge of which documents are the controlled fact injections. Duplicate
detection below never reads a document's `tag`; `tag` is carried through purely
so the report can show how much of what got removed was injected-fact
repetition vs. incidental backbone near-duplication -- a real dedup pipeline
would have no such oracle label to filter on, and giving this one access to it
would defeat the point of using it as a baseline.

Two detectors, combined via union-find into duplicate clusters:
- Exact duplicates: SHA-256 of whitespace/case-normalized text.
- Near duplicates: MinHash signatures over word-shingles + LSH banding to avoid
  O(n^2) pairwise comparison, with candidate pairs verified against the actual
  estimated Jaccard similarity (LSH banding alone can admit sub-threshold
  collisions in the same band).

Reimplemented directly on numpy/hashlib rather than pulling in `datasketch`,
since only signature-min + banding is needed and Python's built-in `hash()` is
not stable across processes (PYTHONHASHSEED), so shingles are hashed with a
fixed-seed 64-bit hash instead.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Document:
    doc_id: str
    text: str
    tag: Optional[str] = None  # e.g. "injected_fact" / "backbone"; reporting only


def read_jsonl_corpus(path: str | Path) -> list[Document]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(Document(doc_id=rec["doc_id"], text=rec["text"], tag=rec.get("tag")))
    return docs


def write_jsonl_corpus(docs: list[Document], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps({"doc_id": d.doc_id, "text": d.text, "tag": d.tag}) + "\n")


_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def exact_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _shingles(text: str, n: int) -> set[str]:
    tokens = _normalize(text).split(" ")
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens and tokens[0] else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


_MERSENNE_PRIME = (1 << 61) - 1


class MinHasher:
    """Deterministic MinHash signatures over word n-gram shingles."""

    def __init__(self, num_perm: int = 128, shingle_size: int = 5, seed: int = 0):
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = np.random.RandomState(seed)
        self._a = rng.randint(1, _MERSENNE_PRIME - 1, size=num_perm).astype(np.int64)
        self._b = rng.randint(0, _MERSENNE_PRIME - 1, size=num_perm).astype(np.int64)

    @staticmethod
    def _shingle_hash(shingle: str) -> int:
        digest = hashlib.sha1(shingle.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big", signed=False) % _MERSENNE_PRIME

    def signature(self, text: str) -> np.ndarray:
        shingles = _shingles(text, self.shingle_size)
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.int64)
        hashes = np.array([self._shingle_hash(s) for s in shingles], dtype=np.int64)
        # (num_perm, num_shingles) permuted hash values; MinHash signature is the
        # per-permutation min over shingles.
        perm = (np.outer(self._a, hashes) + self._b[:, None]) % _MERSENNE_PRIME
        return perm.min(axis=1)


def _jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


class LSHIndex:
    """Banded LSH over MinHash signatures. `insert` returns doc_ids that already
    share at least one band bucket with the incoming signature (candidate pairs)."""

    def __init__(self, num_perm: int, num_bands: int = 32):
        if num_perm % num_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})")
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self._buckets: dict[tuple[int, bytes], list[str]] = {}

    def insert(self, doc_id: str, signature: np.ndarray) -> set[str]:
        candidates: set[str] = set()
        for band in range(self.num_bands):
            start = band * self.rows_per_band
            band_bytes = signature[start : start + self.rows_per_band].tobytes()
            key = (band, band_bytes)
            bucket = self._buckets.setdefault(key, [])
            candidates.update(bucket)
            bucket.append(doc_id)
        return candidates


class UnionFind:
    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[rx] = ry

    def clusters(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for x in self._parent:
            groups.setdefault(self.find(x), []).append(x)
        return groups


@dataclass
class DedupConfig:
    shingle_size: int = 5
    num_perm: int = 128
    num_bands: int = 32
    jaccard_threshold: float = 0.8


def find_duplicate_clusters(docs: list[Document], cfg: Optional[DedupConfig] = None) -> dict[str, list[str]]:
    """Groups doc_ids into duplicate clusters (exact + near via MinHash/LSH).
    Returns {cluster_id: [doc_id, ...]} for clusters of size >= 2 only."""
    cfg = cfg or DedupConfig()
    uf = UnionFind()

    exact_groups: dict[str, list[str]] = {}
    for doc in docs:
        exact_groups.setdefault(exact_hash(doc.text), []).append(doc.doc_id)
    for ids in exact_groups.values():
        for other in ids[1:]:
            uf.union(ids[0], other)

    hasher = MinHasher(num_perm=cfg.num_perm, shingle_size=cfg.shingle_size)
    lsh = LSHIndex(num_perm=cfg.num_perm, num_bands=cfg.num_bands)
    signatures: dict[str, np.ndarray] = {}
    for doc in docs:
        sig = hasher.signature(doc.text)
        signatures[doc.doc_id] = sig
        for cand_id in lsh.insert(doc.doc_id, sig):
            if _jaccard_estimate(sig, signatures[cand_id]) >= cfg.jaccard_threshold:
                uf.union(doc.doc_id, cand_id)

    return {rep: ids for rep, ids in uf.clusters().items() if len(ids) >= 2}


def filter_duplicates(
    docs: list[Document], clusters: dict[str, list[str]], keep: str = "first"
) -> tuple[list[Document], list[dict]]:
    """Keeps one representative per cluster, drops the rest. `keep`: "first" or
    "last" occurrence in `docs` order. Returns (kept_docs, removed_report), where
    removed_report entries are {"doc_id", "tag", "cluster_rep", "cluster_size"}."""
    order = {d.doc_id: i for i, d in enumerate(docs)}
    doc_by_id = {d.doc_id: d for d in docs}
    drop_ids: set[str] = set()
    removed_report = []

    for ids in clusters.values():
        ordered_ids = sorted(ids, key=lambda i: order[i])
        keeper = ordered_ids[0] if keep == "first" else ordered_ids[-1]
        for did in ordered_ids:
            if did == keeper:
                continue
            drop_ids.add(did)
            removed_report.append(
                {
                    "doc_id": did,
                    "tag": doc_by_id[did].tag,
                    "cluster_rep": keeper,
                    "cluster_size": len(ordered_ids),
                }
            )

    kept_docs = [d for d in docs if d.doc_id not in drop_ids]
    return kept_docs, removed_report
