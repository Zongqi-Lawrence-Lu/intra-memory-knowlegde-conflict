"""CLI: document-level soft-dedup reweighting over a jsonl document corpus.

Usage:
    python -m training_time.run_reweighting \\
        --input data/raw/corpus.jsonl --output data/processed/corpus_weights.jsonl \\
        --run-name reweight_backbone

Same content-blind clustering as training_time.run_dedup (see that module's
docstring), but instead of dropping duplicates, writes a per-document weight
(1 / cluster_size) to `--output` as a weights.jsonl sidecar: {"doc_id", "weight",
"cluster_size", "tag"} per line. A `reweight_report.json` summary is written to
`results/<run-name>/` (tracked).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training_time.dedup import DedupConfig, find_duplicate_clusters, read_jsonl_corpus
from training_time.reweighting import compute_document_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="path for the weights.jsonl sidecar")
    parser.add_argument("--run-name", default="reweight")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--num-bands", type=int, default=32)
    parser.add_argument("--jaccard-threshold", type=float, default=0.8)
    args = parser.parse_args()

    cfg = DedupConfig(
        shingle_size=args.shingle_size,
        num_perm=args.num_perm,
        num_bands=args.num_bands,
        jaccard_threshold=args.jaccard_threshold,
    )

    docs = read_jsonl_corpus(args.input)
    clusters = find_duplicate_clusters(docs, cfg)
    weights = compute_document_weights(docs, clusters=clusters)

    doc_by_id = {d.doc_id: d for d in docs}
    cluster_size_by_doc = {did: len(ids) for ids in clusters.values() for did in ids}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "weight": weights[doc.doc_id],
                        "cluster_size": cluster_size_by_doc.get(doc.doc_id, 1),
                        "tag": doc.tag,
                    }
                )
                + "\n"
            )

    downweighted_by_tag = Counter(doc_by_id[did].tag for ids in clusters.values() for did in ids)
    summary = {
        "run_name": args.run_name,
        "config": vars(cfg),
        "input_path": args.input,
        "output_path": str(output_path),
        "num_docs": len(docs),
        "num_duplicate_clusters": len(clusters),
        "num_downweighted_docs": sum(len(ids) for ids in clusters.values()),
        "downweighted_by_tag": dict(downweighted_by_tag),
    }

    results_dir = Path(args.results_dir) / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "reweight_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
