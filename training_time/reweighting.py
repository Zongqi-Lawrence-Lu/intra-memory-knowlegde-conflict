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
duplicates are a document property, not a token-window property).

Getting real documents to compute this over used to be the actual gap: there
was no exporter from the real packed-token corpus (training/data.py's
PackedTokenDataset, meta.json + *.bin) into this module's {doc_id, text, tag}
input format at all -- training_time.corpus_export now closes that gap,
reconstructing documents from the corpus's own EOS-delimited boundaries and
tagging injected-vignette vs. backbone documents via
results/occurrence_log*.json (see that module's docstring for the verified
token-position semantics). compute_document_weights below is directly runnable
against real corpus content today, e.g.:
    python -m training_time.corpus_export --T 80 --output data/processed/full-run-T80-export.jsonl
    docs = read_jsonl_corpus("data/processed/full-run-T80-export.jsonl")
    weights = compute_document_weights(docs)

What's still genuinely missing: joining these doc_id-keyed weights back into
PackedTokenDataset's per-*window* sampling during training itself, since
PackedTokenDataset windows (arbitrary block_size-token slices, possibly
overlapping) carry no doc_id of their own -- that would require the M2 corpus
packer to record window->doc_id metadata, which training/data.py still does
not do. Until that lands, use
training_time.weighted_dataset.compute_window_weights for the window-
granularity fallback that already works against the training loop as it
exists today (and is itself vectorized for real corpus scale).
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
