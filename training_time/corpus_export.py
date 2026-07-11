"""Exports a real packed-corpus shard (training/data.py's meta.json + *.bin format,
built by preprocess/assemble_corpus.py) into the {doc_id, text, tag} jsonl format
training_time/dedup.py and training_time/reweighting.py already expect via
read_jsonl_corpus. Closes the gap those modules' docstrings previously flagged
("no exporter... yet" / "does not exist yet") -- the corpus assembler's own
EOS-delimited document boundaries make this recoverable without any extra
metadata, and results/occurrence_log.json (added 2026-07-06, not available when
those docstrings were written) already records exactly which positions are
injected-fact documents.

Splits the flat token stream on the GPT-2 EOS token (50256) to recover document
boundaries -- experimental_plans.tex's corpus-assembly design (Sec.assembly):
every OpenWebText article and every injected vignette occurrence is its own
self-contained EOS-delimited unit. A document is tagged "injected_fact" if its
start position is exactly one past a results/occurrence_log*.json entry's
final_token_position. Verified directly against the real assembler and a real
shard (preprocess/assemble_corpus.py:assemble_from_openwebtext_stream, the path
that actually built this project's OpenWebText-backed corpora): final_token_
position is logged *before* `write_injected_tokens([eos] + variant_tokens)`, so
it is the position of a forced leading EOS token (always present, unconditionally
-- see that function's own comment on why), not the vignette's first real content
token, which lands at final_token_position + 1. Confirmed concretely: for T=80's
first occurrence_log entry (entity_0468, final_token_position=3034),
data/processed/full-run/shard_0000.bin has an EOS token at index 3034 and
entity_0468's actual biography text starting at index 3035. (The *other* assembly
path, assemble_from_population -- the small-scale two-pass local-file path, not
used for any real corpus here -- writes variant_tokens with no forced leading
EOS, so final_token_position there already is the content start; not relevant to
the corpora this exporter is ever pointed at, per T_CONDITIONS, but noted so a
future reader isn't misled by the other function's different offset.)

Processes the memmap in bounded chunks (never the whole multi-GB shard at once
-- the same discipline preprocess/backbone_openwebtext.py already uses for the
same reason), carrying any document that spans a chunk boundary into the next
chunk, so memory stays bounded regardless of corpus size. Decodes in per-chunk
batches via tokenizer.batch_decode rather than one decode() call per document.

Usage:
    python -m training_time.corpus_export --T 80 --output data/processed/full-run-T80-export.jsonl
    python -m training_time.corpus_export --T 80 --output /tmp/smoke.jsonl --limit-tokens 5000000

Then feed the output straight into the existing dedup/reweighting CLIs, e.g.:
    python -m training_time.run_dedup --input data/processed/full-run-T80-export.jsonl \\
        --output data/processed/full-run-T80-deduped.jsonl --run-name dedup_T80

A full-scale export processes ~4e9 tokens; expect on the order of tens of
minutes of CPU time (tokenizer decoding, not GPU) -- use --limit-tokens for a
bounded smoke test before committing to a full run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import GPT2TokenizerFast

from inference_time.utils.model_utils import REPO_ROOT, T_CONDITIONS
from training.config import TrainingConfig

EOS_TOKEN_ID = 50256
DEFAULT_CHUNK_TOKENS = 20_000_000  # ~40MB/chunk at uint16 -- bounded regardless of corpus size

def data_dir_for(T: int) -> Path:
    cfg = TrainingConfig.from_yaml(REPO_ROOT / T_CONDITIONS[T]["config"])
    return REPO_ROOT / cfg.data.train_path


def occurrence_log_path_for(T: int) -> Path:
    # 2026-07-11: lives in that run's own results/ folder now (moved off the old
    # results/occurrence_log[_T{T}].json top-level naming convention, which special-
    # cased T=80 as the unsuffixed/implicit default -- see
    # memory/results_folder_scatter_cleanup_2026-07-11.md), same explicit pattern for
    # every T condition, no special-casing.
    return REPO_ROOT / "results" / T_CONDITIONS[T]["run_name"] / "occurrence_log.json"


def _load_occurrence_index(occurrence_log_path: Path) -> dict[int, dict]:
    """(final_token_position + 1) -> occurrence record, i.e. keyed by the
    injected document's actual start position, not the forced leading EOS
    position final_token_position itself records (see module docstring for the
    verified offset). Positions are unique by construction (each is a distinct
    write offset in a single linear pass)."""
    with open(occurrence_log_path) as f:
        log = json.load(f)
    return {e["final_token_position"] + 1: e for e in log}


def _write_record(out_f, text: str, abs_start: int, occ_index: dict, stats: dict) -> None:
    entry = occ_index.get(abs_start)
    if entry is not None:
        doc_id = f"injected_{entry['entity_id']}_{entry['side']}_{abs_start}"
        tag = "injected_fact"
        stats["n_injected"] += 1
    else:
        doc_id = f"backbone_{abs_start}"
        tag = "backbone"
        stats["n_backbone"] += 1
    out_f.write(json.dumps({"doc_id": doc_id, "text": text, "tag": tag}) + "\n")
    stats["n_docs"] += 1


def export_corpus(
    data_dir: Path,
    occurrence_log_path: Path,
    output_path: Path,
    limit_tokens: Optional[int] = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> dict:
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    dtype = np.dtype(meta.get("dtype", "uint16"))
    shard_paths = sorted(data_dir.glob("*.bin"))
    if len(shard_paths) != 1:
        raise ValueError(
            f"expected exactly one shard under {data_dir} (this project's corpora are "
            f"single-shard, experimental_plans.tex Sec.assembly), found {len(shard_paths)}"
        )
    arr = np.memmap(shard_paths[0], dtype=dtype, mode="r")
    total = len(arr) if limit_tokens is None else min(limit_tokens, len(arr))

    occ_index = _load_occurrence_index(occurrence_log_path)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"n_docs": 0, "n_injected": 0, "n_backbone": 0, "n_empty_skipped": 0}

    carry_tokens: list[int] = []
    carry_start: Optional[int] = None

    with open(output_path, "w") as out_f:
        pos = 0
        while pos < total:
            end = min(pos + chunk_tokens, total)
            chunk = np.asarray(arr[pos:end])
            eos_local_positions = np.flatnonzero(chunk == EOS_TOKEN_ID).tolist()

            batch_tokens: list[list[int]] = []
            batch_starts: list[int] = []

            local_start = 0
            for eos_local in eos_local_positions:
                piece = chunk[local_start:eos_local].tolist()
                if carry_tokens:
                    doc_tokens = carry_tokens + piece
                    abs_start = carry_start
                    carry_tokens, carry_start = [], None
                else:
                    doc_tokens = piece
                    abs_start = pos + local_start
                if doc_tokens:
                    batch_tokens.append(doc_tokens)
                    batch_starts.append(abs_start)
                else:
                    # Doubled EOS (a vignette firing exactly on a whole-document
                    # boundary, experimental_plans.tex Sec.assembly) -- harmless,
                    # not a real document.
                    stats["n_empty_skipped"] += 1
                local_start = eos_local + 1

            if local_start < len(chunk):
                tail = chunk[local_start:].tolist()
                if carry_tokens:
                    carry_tokens += tail
                else:
                    carry_tokens, carry_start = tail, pos + local_start

            if batch_tokens:
                texts = tokenizer.batch_decode(batch_tokens, skip_special_tokens=True)
                for text, abs_start in zip(texts, batch_starts):
                    _write_record(out_f, text, abs_start, occ_index, stats)

            pos = end

        if carry_tokens:
            # A full export (limit_tokens=None) ends exactly on the corpus's final
            # EOS -- assemble_corpus.py always closes on one -- so this only fires
            # for a --limit-tokens smoke test that truncates mid-document.
            text = tokenizer.decode(carry_tokens, skip_special_tokens=True)
            _write_record(out_f, text, carry_start, occ_index, stats)

    stats["n_occurrence_log_entries"] = len(occ_index)
    stats["n_occurrence_log_matched"] = stats["n_injected"]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--T", type=int, choices=sorted(T_CONDITIONS), required=True)
    parser.add_argument("--output", required=True, help="Output jsonl path (data/ artifact, not tracked in git).")
    parser.add_argument("--limit-tokens", type=int, default=None, help="Bounded smoke test; omit for a full export.")
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    args = parser.parse_args()

    data_dir = data_dir_for(args.T)
    occurrence_log_path = occurrence_log_path_for(args.T)
    output_path = Path(args.output)

    print(f"T={args.T}: exporting {data_dir} (occurrence log: {occurrence_log_path}) -> {output_path}")
    if args.limit_tokens is not None:
        print(f"  --limit-tokens={args.limit_tokens} (bounded smoke test, not a full export)")

    stats = export_corpus(
        data_dir, occurrence_log_path, output_path,
        limit_tokens=args.limit_tokens, chunk_tokens=args.chunk_tokens,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
