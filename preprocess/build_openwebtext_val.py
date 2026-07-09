"""One-time held-out OpenWebText validation shard (2026-07-09), built from
documents strictly after the ones already consumed for the training backbone
cache (data/processed/backbone-cache/state.json's docs_consumed) -- so this
validation set shares OpenWebText's exact domain/style with training, unlike the
existing held-out-wikitext-val (a different, and much smaller, corpus). Addresses
the "how much of the T=320 run's ~70-95 val_ppl is domain shift vs undertraining"
question directly: the pipeline can now report both an out-of-domain (WikiText)
and an in-domain (this) perplexity for the same checkpoint.

All of this project's training corpora (T=80/160/320/640/1280) were built by
tokenizing a prefix of the exact same deterministic OpenWebText stream (the
`Skylion007/openwebtext` HF dataset's streaming row order), via
preprocess/backbone_openwebtext.py:cache_openwebtext_backbone writing to a single
shared cache all of them read from (data/processed/backbone-cache) -- so
`docs_consumed` from that cache's state.json is the exact document count already
used by *every* T condition, not just one. Skipping exactly that many rows before
reading anything guarantees zero train/val document overlap regardless of which
T's corpus this is later used to evaluate.

Output: data/processed/<out-dir>/shard_0000.bin + meta.json, in the same on-disk
format training/data.py:PackedTokenDataset already expects (a drop-in --val_path
config value), plus build_info.json recording exactly which document range was
read, for provenance.

Network/CPU-bound, no GPU needed (same profile as
preprocess/cache_backbone.py) -- run via slurm/build_openwebtext_val.sbatch, not
directly on a login node.

    python -m preprocess.build_openwebtext_val --total-tokens 20000000 --out-dir data/processed/held-out-openwebtext-val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import GPT2TokenizerFast

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_BACKBONE_CACHE_STATE = REPO_ROOT / "data" / "processed" / "backbone-cache" / "state.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-tokens", type=int, default=20_000_000)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed" / "held-out-openwebtext-val"))
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument(
        "--skip-docs",
        type=int,
        default=None,
        help="defaults to backbone-cache/state.json's docs_consumed -- every document "
        "already used by any training corpus built from that cache.",
    )
    args = parser.parse_args()

    if args.skip_docs is not None:
        skip_docs = args.skip_docs
    else:
        with open(DEFAULT_BACKBONE_CACHE_STATE) as f:
            skip_docs = json.load(f)["docs_consumed"]
    print(f"Skipping the first {skip_docs} OpenWebText documents (already used for training).")

    import datasets

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos = tokenizer.eos_token_id
    np_dtype = np.dtype(args.dtype)

    ds = datasets.load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    ds = ds.skip(skip_docs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    docs_seen = 0
    docs_used = 0
    with open(out_dir / "shard_0000.bin", "wb") as out_f:
        for row in ds:
            docs_seen += 1
            text = row["text"].strip()
            if not text:
                continue
            ids = tokenizer.encode(text) + [eos]
            out_f.write(np.array(ids, dtype=np_dtype).tobytes())
            total_tokens += len(ids)
            docs_used += 1
            if total_tokens >= args.total_tokens:
                break

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"dtype": args.dtype, "vocab_size": tokenizer.vocab_size}, f, indent=2)
    with open(out_dir / "build_info.json", "w") as f:
        json.dump(
            {
                "source_dataset": "Skylion007/openwebtext",
                "split": "train",
                "skipped_docs": skip_docs,
                "docs_read": docs_seen,
                "docs_used": docs_used,
                "total_tokens": total_tokens,
                "note": "documents strictly after backbone-cache's docs_consumed, i.e. "
                "zero overlap with any T=80/160/320/640/1280 training corpus built from "
                "that same cache.",
            },
            f,
            indent=2,
        )

    print(
        f"Wrote {total_tokens} tokens from {docs_used} non-empty documents "
        f"(read {docs_seen} raw rows past skip_docs={skip_docs}) to {out_dir}"
    )


if __name__ == "__main__":
    main()
