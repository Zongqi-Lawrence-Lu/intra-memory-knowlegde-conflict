"""Held-out OpenWebText validation shard, v2 (2026-07-10 rebuild), built by streaming
OpenWebText from document 0 and skipping a fixed --skip-tokens backbone tokens before
collecting --total-tokens of val data -- shares OpenWebText's exact domain/style with
training, unlike held-out-wikitext-val (a different, smaller corpus).

Replaces the original (2026-07-09) version, which derived its skip point from
data/processed/backbone-cache/state.json's docs_consumed at build time. That was a
real bug, not just imprecision: cache_openwebtext_backbone's cache is resumable and
grows monotonically -- when the training backbone target was later bumped from 2.5B to
4.0B tokens, the cache was extended by resuming from the exact docs_consumed the val
set had used as its own skip point, so the extension re-consumed precisely the
document range the val set occupied. Confirmed directly (not just inferred): byte-level
fingerprint probes from across the entire old val shard were found verbatim inside all
three of T80/T320/T1280's training corpora. The old val set was not held out at all --
"in-domain" perplexity numbers computed against it (results/*/held_out_ppl.jsonl)
measured memorized text, not generalization.

Fix: --skip-tokens is now a fixed token count measured from document 0 of the stream,
with no dependency on backbone-cache or any training corpus's current state. 6B tokens
(this script's default) sits 2B tokens (50%) above the current 4B-token training
target and ~3B tokens short of OpenWebText's ~9.04B-token total, so training can grow
further without silently invalidating this val set again -- though if training's
backbone target is ever pushed past ~6B tokens, --skip-tokens must be bumped again (and
the val set rebuilt) at that time; this script does not self-detect that condition.

Resumable (CLAUDE.md Sec. 5's restart-safety requirement -- skipping 6B+ tokens over a
live network stream is a many-hour job): out_dir/build_state.json records
{docs_processed, tokens_skipped, tokens_written, phase} after every
CHECKPOINT_EVERY_DOCS documents; a restart with the same --out-dir reads it back,
ds.skip(docs_processed)s past already-processed rows, and continues in the same phase.
shard_0000.bin.tmp is opened in append mode across resumes and its on-disk byte count
is checked against tokens_written before continuing, so a kill mid-write is caught
rather than silently trusted. Final commit is an atomic tmp-then-rename, matching
preprocess/assemble_corpus.py's pattern.

    python -m preprocess.build_openwebtext_val --skip-tokens 6000000000 --total-tokens 20000000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import GPT2TokenizerFast

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "held-out-openwebtext-val"
DEFAULT_SKIP_TOKENS = 6_000_000_000
DEFAULT_TOTAL_TOKENS = 20_000_000
CHECKPOINT_EVERY_DOCS = 50_000  # frequent enough to bound lost progress on a Slurm
# timeout/preemption to a few minutes, per CLAUDE.md's >1h-job restart requirement.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-tokens", type=int, default=DEFAULT_SKIP_TOKENS,
        help="Backbone tokens to skip (from document 0) before collecting val data. Fixed, "
             "not derived from any cache/training state -- see module docstring.",
    )
    parser.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dtype", default="uint16")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "build_state.json"
    shard_tmp_path = out_dir / "shard_0000.bin.tmp"
    shard_final_path = out_dir / "shard_0000.bin"
    np_dtype = np.dtype(args.dtype)

    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        if state["skip_tokens_target"] != args.skip_tokens or state["total_tokens_target"] != args.total_tokens:
            raise RuntimeError(
                f"{state_path} was checkpointed with skip_tokens={state['skip_tokens_target']}, "
                f"total_tokens={state['total_tokens_target']}, which don't match this invocation's "
                f"--skip-tokens {args.skip_tokens} --total-tokens {args.total_tokens} -- refusing to "
                f"resume with mismatched targets."
            )
        docs_processed = state["docs_processed"]
        tokens_skipped = state["tokens_skipped"]
        tokens_written = state["tokens_written"]
        phase = state["phase"]
        expected_bytes = tokens_written * np_dtype.itemsize
        actual_bytes = shard_tmp_path.stat().st_size if shard_tmp_path.exists() else 0
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"resume state mismatch: {state_path} claims {tokens_written} tokens written "
                f"({expected_bytes} bytes) but {shard_tmp_path} has {actual_bytes} bytes -- refusing "
                f"to resume against a possibly-corrupt partial shard. Delete both and restart if the "
                f"prior run is unrecoverable."
            )
        print(f"Resuming from {state_path}: phase={phase}, docs_processed={docs_processed}, "
              f"tokens_skipped={tokens_skipped}, tokens_written={tokens_written}")
    else:
        docs_processed = 0
        tokens_skipped = 0
        tokens_written = 0
        phase = "skipping" if args.skip_tokens > 0 else "collecting"

    import datasets

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos = tokenizer.eos_token_id

    ds = datasets.load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    if docs_processed:
        ds = ds.skip(docs_processed)

    def write_checkpoint() -> None:
        state_tmp = state_path.with_suffix(".json.tmp")
        with open(state_tmp, "w") as f:
            json.dump(
                {
                    "docs_processed": docs_processed,
                    "tokens_skipped": tokens_skipped,
                    "tokens_written": tokens_written,
                    "phase": phase,
                    "skip_tokens_target": args.skip_tokens,
                    "total_tokens_target": args.total_tokens,
                },
                f,
            )
        state_tmp.replace(state_path)

    out_f = open(shard_tmp_path, "ab")
    docs_since_checkpoint = 0
    try:
        for row in ds:
            docs_processed += 1
            docs_since_checkpoint += 1
            text = row["text"].strip()
            if text:
                ids = tokenizer.encode(text) + [eos]
                if phase == "skipping":
                    tokens_skipped += len(ids)
                    if tokens_skipped >= args.skip_tokens:
                        phase = "collecting"
                else:
                    out_f.write(np.array(ids, dtype=np_dtype).tobytes())
                    tokens_written += len(ids)

            if docs_since_checkpoint >= CHECKPOINT_EVERY_DOCS:
                out_f.flush()
                write_checkpoint()
                print(f"... docs_processed={docs_processed} phase={phase} "
                      f"tokens_skipped={tokens_skipped} tokens_written={tokens_written}")
                docs_since_checkpoint = 0

            if phase == "collecting" and tokens_written >= args.total_tokens:
                break
    finally:
        out_f.close()

    write_checkpoint()

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"dtype": args.dtype, "vocab_size": tokenizer.vocab_size}, f, indent=2)
    shard_tmp_path.replace(shard_final_path)
    with open(out_dir / "build_info.json", "w") as f:
        json.dump(
            {
                "source_dataset": "Skylion007/openwebtext",
                "split": "train",
                "skip_tokens_target": args.skip_tokens,
                "actual_tokens_skipped": tokens_skipped,
                "docs_read": docs_processed,
                "total_tokens": tokens_written,
                "note": "skip point is a fixed token count from document 0, independent of "
                        "backbone-cache or any training corpus's state -- see module docstring "
                        "for the leak this replaces.",
            },
            f,
            indent=2,
        )
    state_path.unlink(missing_ok=True)
    print(f"Wrote {tokens_written} tokens to {shard_final_path} "
          f"(skipped {tokens_skipped} tokens / {docs_processed} docs total)")


if __name__ == "__main__":
    main()
