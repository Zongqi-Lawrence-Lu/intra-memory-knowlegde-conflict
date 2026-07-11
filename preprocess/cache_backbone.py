"""CLI entrypoint for preprocess.backbone_openwebtext's OpenWebText caching functions
-- streams OpenWebText once and caches the raw text to local JSON chunk files, so an
exposure-budget sweep across multiple T values (experimental_plans.tex Sec. seeds)
only pays this network cost once. preprocess/assemble_corpus.py's --backbone-cache-dir
flag reads the result back for each T's (fast, local, no-network) corpus assembly.

Two modes:
  --mode tokens (default): cache_openwebtext_backbone, stops at --total-tokens backbone
      tokens (calls the tokenizer per document to measure progress -- CPU + network).
  --mode raw: cache_openwebtext_raw_text, stops at --max-bytes raw characters, no
      tokenizer call at all -- pure network streaming, for benchmarking/running the
      network-bound cost separately from CPU tokenization (e.g. from a well-connected
      interactive node before a compute-node job does the CPU-bound tokenize/pack step).

    python -m preprocess.cache_backbone --mode tokens --total-tokens 2500000000 --cache-dir data/processed/backbone-cache
    python -m preprocess.cache_backbone --mode raw --max-bytes 30000000000 --cache-dir data/processed/val-source-cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

from preprocess.backbone_openwebtext import (
    DEFAULT_CACHE_CHUNK_BYTES,
    cache_openwebtext_backbone,
    cache_openwebtext_raw_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["tokens", "raw"], default="tokens")
    parser.add_argument("--total-tokens", type=int, help="required for --mode tokens")
    parser.add_argument("--max-bytes", type=int, help="required for --mode raw")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CACHE_CHUNK_BYTES)
    args = parser.parse_args()
    if args.mode == "tokens":
        if args.total_tokens is None:
            parser.error("--total-tokens is required for --mode tokens")
        cache_openwebtext_backbone(args.total_tokens, args.cache_dir, chunk_bytes=args.chunk_bytes)
    else:
        if args.max_bytes is None:
            parser.error("--max-bytes is required for --mode raw")
        cache_openwebtext_raw_text(args.max_bytes, args.cache_dir, chunk_bytes=args.chunk_bytes)


if __name__ == "__main__":
    main()
