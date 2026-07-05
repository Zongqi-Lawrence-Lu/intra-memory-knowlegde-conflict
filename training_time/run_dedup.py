"""CLI: content-blind exact + near-duplicate removal over a jsonl document corpus.

Usage:
    python -m training_time.run_dedup \\
        --input data/raw/corpus.jsonl --output data/processed/corpus_deduped.jsonl \\
        --run-name dedup_backbone

Input schema (one JSON object per line): {"doc_id": str, "text": str, "tag": str?}.
`tag` (e.g. "injected_fact" / "backbone") is optional and used only in the report,
never as a filtering criterion -- see training_time/dedup.py module docstring.

Writes the deduplicated corpus to `--output` (a data/ artifact, not tracked in git)
and a `dedup_report.json` under `results/<run-name>/` (tracked) with cluster counts
and a breakdown of what was removed, by tag.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training_time.dedup import DedupConfig, filter_duplicates, find_duplicate_clusters, read_jsonl_corpus, write_jsonl_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", default="dedup")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--num-bands", type=int, default=32)
    parser.add_argument("--jaccard-threshold", type=float, default=0.8)
    parser.add_argument("--keep", choices=["first", "last"], default="first")
    args = parser.parse_args()

    cfg = DedupConfig(
        shingle_size=args.shingle_size,
        num_perm=args.num_perm,
        num_bands=args.num_bands,
        jaccard_threshold=args.jaccard_threshold,
    )

    docs = read_jsonl_corpus(args.input)
    clusters = find_duplicate_clusters(docs, cfg)
    kept_docs, removed_report = filter_duplicates(docs, clusters, keep=args.keep)
    write_jsonl_corpus(kept_docs, args.output)

    removed_by_tag = Counter(r["tag"] for r in removed_report)
    summary = {
        "run_name": args.run_name,
        "config": vars(cfg),
        "input_path": args.input,
        "output_path": args.output,
        "num_input_docs": len(docs),
        "num_kept_docs": len(kept_docs),
        "num_removed_docs": len(removed_report),
        "num_duplicate_clusters": len(clusters),
        "removed_by_tag": dict(removed_by_tag),
    }

    results_dir = Path(args.results_dir) / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "dedup_report.json", "w") as f:
        json.dump({"summary": summary, "removed": removed_report}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
