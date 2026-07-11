"""Held-out OpenWebText validation shard, built from the LOCAL raw-text cache
(preprocess/backbone_openwebtext.py:cache_openwebtext_raw_text's output, default
data/processed/val-source-cache, chunk_%05d.json files each ~300M raw chars) instead
of re-streaming OpenWebText live -- avoids build_openwebtext_val.py's ~14.7h
network-bound skip+collect cost (see slurm/build_openwebtext_val.sbatch).

Contamination avoidance: training's corpora used at most ~4B backbone tokens (see
[[four_billion_token_chains]]). Rather than precisely counting tokens from document 0
(what build_openwebtext_val.py does), this script starts tokenizing from
--start-chunk-idx (default 88 of 90 total chunks) and collects --total-tokens
(default 2,000,000) from there -- a char-count estimate (chars/token ~4.3-4.4,
measured directly across chunks 50/70/84/87/89) puts chunk 88's start comfortably
past the 6B-token mark, ~2B tokens clear of the 4B contamination boundary, so the
exact starting token offset doesn't matter for correctness, only that it's well
past 4B, which it is by a wide margin.

Resumable (CLAUDE.md Sec. 5's >1h restart-safety requirement), though at 2M tokens
from local disk this finishes in well under a minute: out_dir/build_state.json
records {chunk_idx, doc_idx_in_chunk, tokens_written} after every
CHECKPOINT_EVERY_DOCS documents. A restart with the same --out-dir jumps directly to
that chunk file and resumes mid-chunk, checked against shard_0000.bin.tmp's on-disk
byte count. Final commit is an atomic tmp-then-rename, matching
build_openwebtext_val.py's pattern.

    python -m preprocess.build_openwebtext_val_from_cache --start-chunk-idx 88 --total-tokens 2000000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import GPT2TokenizerFast

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "processed" / "val-source-cache"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "held-out-openwebtext-val"
DEFAULT_START_CHUNK_IDX = 88
DEFAULT_TOTAL_TOKENS = 2_000_000
CHECKPOINT_EVERY_DOCS = 50_000  # matches build_openwebtext_val.py's checkpoint cadence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Local raw-text cache from cache_backbone.py --mode raw (chunk_%%05d.json + meta.json).",
    )
    parser.add_argument(
        "--start-chunk-idx", type=int, default=DEFAULT_START_CHUNK_IDX,
        help="Cache chunk index to start collecting from (doc 0 of that chunk). Default 88 is "
             "estimated (char-count based) to be well past the 6B-token / 4B-contamination mark.",
    )
    parser.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dtype", default="uint16")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    with open(cache_dir / "meta.json") as f:
        cache_meta = json.load(f)
    num_chunks = cache_meta["num_chunks"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "build_state.json"
    shard_tmp_path = out_dir / "shard_0000.bin.tmp"
    shard_final_path = out_dir / "shard_0000.bin"
    np_dtype = np.dtype(args.dtype)

    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        if state["start_chunk_idx"] != args.start_chunk_idx or state["total_tokens_target"] != args.total_tokens:
            raise RuntimeError(
                f"{state_path} was checkpointed with start_chunk_idx={state['start_chunk_idx']}, "
                f"total_tokens={state['total_tokens_target']}, which don't match this invocation's "
                f"--start-chunk-idx {args.start_chunk_idx} --total-tokens {args.total_tokens} -- refusing "
                f"to resume with mismatched targets."
            )
        chunk_idx = state["chunk_idx"]
        doc_idx_in_chunk = state["doc_idx_in_chunk"]
        tokens_written = state["tokens_written"]
        expected_bytes = tokens_written * np_dtype.itemsize
        actual_bytes = shard_tmp_path.stat().st_size if shard_tmp_path.exists() else 0
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"resume state mismatch: {state_path} claims {tokens_written} tokens written "
                f"({expected_bytes} bytes) but {shard_tmp_path} has {actual_bytes} bytes -- refusing "
                f"to resume against a possibly-corrupt partial shard. Delete both and restart if the "
                f"prior run is unrecoverable."
            )
        print(f"Resuming from {state_path}: chunk_idx={chunk_idx}, doc_idx_in_chunk={doc_idx_in_chunk}, "
              f"tokens_written={tokens_written}")
    else:
        chunk_idx = args.start_chunk_idx
        doc_idx_in_chunk = 0
        tokens_written = 0

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos = tokenizer.eos_token_id

    def write_checkpoint() -> None:
        state_tmp = state_path.with_suffix(".json.tmp")
        with open(state_tmp, "w") as f:
            json.dump(
                {
                    "chunk_idx": chunk_idx,
                    "doc_idx_in_chunk": doc_idx_in_chunk,
                    "tokens_written": tokens_written,
                    "start_chunk_idx": args.start_chunk_idx,
                    "total_tokens_target": args.total_tokens,
                },
                f,
            )
        state_tmp.replace(state_path)

    out_f = open(shard_tmp_path, "ab")
    docs_since_checkpoint = 0
    done = False
    try:
        while chunk_idx < num_chunks and not done:
            with open(cache_dir / f"chunk_{chunk_idx:05d}.json") as f:
                docs = json.load(f)
            for doc_idx in range(doc_idx_in_chunk, len(docs)):
                text = docs[doc_idx].strip()
                docs_since_checkpoint += 1
                if text:
                    ids = tokenizer.encode(text) + [eos]
                    out_f.write(np.array(ids, dtype=np_dtype).tobytes())
                    tokens_written += len(ids)

                if docs_since_checkpoint >= CHECKPOINT_EVERY_DOCS:
                    doc_idx_in_chunk = doc_idx + 1
                    out_f.flush()
                    write_checkpoint()
                    print(f"... chunk_idx={chunk_idx} doc_idx_in_chunk={doc_idx_in_chunk} "
                          f"tokens_written={tokens_written}")
                    docs_since_checkpoint = 0

                if tokens_written >= args.total_tokens:
                    doc_idx_in_chunk = doc_idx + 1
                    done = True
                    break
            else:
                chunk_idx += 1
                doc_idx_in_chunk = 0
    finally:
        out_f.close()

    write_checkpoint()

    if not done:
        raise RuntimeError(
            f"Exhausted all {num_chunks} cached chunks in {cache_dir} before reaching "
            f"--total-tokens {args.total_tokens} from --start-chunk-idx {args.start_chunk_idx}: only "
            f"tokens_written={tokens_written}. Rerun with a smaller --start-chunk-idx."
        )

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"dtype": args.dtype, "vocab_size": tokenizer.vocab_size}, f, indent=2)
    shard_tmp_path.replace(shard_final_path)
    with open(out_dir / "build_info.json", "w") as f:
        json.dump(
            {
                "source_dataset": "Skylion007/openwebtext",
                "source_cache_dir": str(cache_dir),
                "split": "train",
                "start_chunk_idx": args.start_chunk_idx,
                "total_tokens": tokens_written,
                "note": "Tokenized from a local raw-text cache (preprocess/backbone_openwebtext.py:"
                        "cache_openwebtext_raw_text's output) starting at chunk start_chunk_idx (doc 0), "
                        "estimated (char-count based, not exact) to be well past both the 6B-token mark "
                        "and training's ~4B-token contamination boundary -- see this script's module "
                        "docstring.",
            },
            f,
            indent=2,
        )
    state_path.unlink(missing_ok=True)
    print(f"Wrote {tokens_written} tokens to {shard_final_path} "
          f"(started at chunk_idx={args.start_chunk_idx}, stopped at chunk_idx={chunk_idx})")


if __name__ == "__main__":
    main()
