"""Document-level reweighting (M7 training-time baseline): the "soft dedup"
counterpart to training_time.dedup. Instead of deleting one representative per
duplicate cluster and dropping the rest, every document is kept but its weight
is set to 1 / cluster_size, so a document that recurs in N near-duplicates
contributes 1/N as much total weight as a document with no duplicates. Reuses
the exact same content-blind clustering as training_time.dedup
(find_duplicate_clusters) -- the two baselines differ only in what they do with
a detected duplicate cluster, not in how they detect it.

This produces a *document*-level weights sidecar (keyed by doc_id), which is
the natural granularity for near-duplicate detection (paraphrase-level
duplicates are a document property, not a token-window property). Joining
these weights into the packed-token training loop requires the M2 corpus
packer to record which document each token window came from, which does not
exist yet (training/data.py's PackedTokenDataset has no document-boundary
metadata). Until then, use training_time.weighted_dataset.compute_window_weights
for the window-granularity fallback that works against the training loop as it
exists today.
"""
from __future__ import annotations

from typing import Optional

from training_time.dedup import DedupConfig, Document, find_duplicate_clusters


def compute_document_weights(
    docs: list[Document], clusters: Optional[dict[str, list[str]]] = None, cfg: Optional[DedupConfig] = None
) -> dict[str, float]:
    """weight[doc_id] = 1 / cluster_size; documents in no duplicate cluster get 1.0.
    Pass `clusters` directly to reuse a clustering already computed elsewhere
    (e.g. by training_time.run_dedup on the same corpus) instead of recomputing it."""
    if clusters is None:
        clusters = find_duplicate_clusters(docs, cfg)

    weights = {doc.doc_id: 1.0 for doc in docs}
    for ids in clusters.values():
        w = 1.0 / len(ids)
        for did in ids:
            weights[did] = w
    return weights
