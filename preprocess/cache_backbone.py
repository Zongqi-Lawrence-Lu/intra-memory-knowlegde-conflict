"""CLI entrypoint for preprocess.backbone_openwebtext.cache_openwebtext_backbone --
streams OpenWebText once and caches the raw text needed to cover --total-tokens
backbone tokens to local JSON chunk files, so an exposure-budget sweep across
multiple T values (experimental_plans.tex Sec. seeds) only pays this network/
tokenization cost once. preprocess/assemble_corpus.py's --backbone-cache-dir flag
reads the result back for each T's (fast, local, no-network) corpus assembly.

    python -m preprocess.cache_backbone --total-tokens 2500000000 --cache-dir data/processed/backbone-cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

from preprocess.backbone_openwebtext import DEFAULT_CACHE_CHUNK_BYTES, cache_openwebtext_backbone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CACHE_CHUNK_BYTES)
    args = parser.parse_args()
    cache_openwebtext_backbone(args.total_tokens, args.cache_dir, chunk_bytes=args.chunk_bytes)


if __name__ == "__main__":
    main()
