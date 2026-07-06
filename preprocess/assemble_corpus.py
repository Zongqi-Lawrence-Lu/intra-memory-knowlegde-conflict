"""Corpus assembly: interleaves a backbone with entities' LLM-authored vignette
occurrences into packed token shards, experimental_plans.tex S1.7 (Corpus Assembly).
Pure code -- no LLM calls; consumes results/population.json,
data_pools/vignettes/<entity_id>_<side>.json, and a backbone text source.

Two backbone sources:
  - openwebtext (default): streamed in ~2GB chunks via
    preprocess/backbone_openwebtext.py, nothing downloaded up front, one pass
    (S1.1 -- superseded WikiText-103, which only had ~110M tokens, a hard ceiling
    well short of the fixed 2.5e9-token target this project now uses).
  - wikitext_local: the original local-file, two-pass path (data/raw/wikitext-103/),
    kept for network-free dev/smoke testing at small token budgets.

    python -m preprocess.assemble_corpus --total-tokens 2500000000 --out-dir data/processed/full-run
    python -m preprocess.assemble_corpus --backbone wikitext_local --total-tokens 200000 --out-dir data/processed/dev-run

Output: data/processed/<name>/meta.json + *.bin shard(s), matching the on-disk contract
training/data.py:PackedTokenDataset already expects. Also writes
results/occurrence_log.json (small, tracked) recording every occurrence event's final
token position, for exposure-count reconstruction (S1.3) and for the training loop to
derive dense-checkpoint step windows (S1.8) from token position + batch/block size.

Implementation note on `--total-tokens`: this is the BACKBONE token budget, not the
final packed-stream length -- injected vignette occurrences add their own tokens on
top (a small fraction of the total, S1.7), so the final corpus is slightly longer than
`--total-tokens`.
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
from pathlib import Path

import numpy as np
from transformers import GPT2Tokenizer, GPT2TokenizerFast

from preprocess.backbone_openwebtext import (
    DEFAULT_CHUNK_BYTES,
    iter_openwebtext_chunks,
    tokenize_with_unit_boundaries,
)
from preprocess.entities import PoolCycler
from preprocess.scheduler import schedule_entity_occurrences

REPO_ROOT = Path(__file__).parent.parent
POPULATION_PATH = REPO_ROOT / "results" / "population.json"
VIGNETTES_DIR = Path(__file__).parent / "data_pools" / "vignettes"
OCCURRENCE_LOG_PATH = REPO_ROOT / "results" / "occurrence_log.json"
DEFAULT_BACKBONE_PATH = REPO_ROOT / "data" / "raw" / "wikitext-103" / "wiki.train.txt"


def load_population() -> list[dict]:
    with open(POPULATION_PATH) as f:
        return json.load(f)


def load_variants(entity_id: str, side: str) -> list[str]:
    path = VIGNETTES_DIR / f"{entity_id}_{side}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("variants", [])


def measure_backbone_lines(
    backbone_path: Path, tokenizer: GPT2Tokenizer, token_budget: int
) -> tuple[int, list[int]]:
    """Pass 1 of 2: tokenizes each line only to measure its length (+1 for EOS),
    building a prefix-sum of cumulative tokens per line -- needed to snap scheduler
    positions to line boundaries before line content is written anywhere. Does NOT
    keep any token ids in memory (only small integers), so this scales to the real
    total_tokens target (10^8-10^9) without holding the whole backbone in RAM --
    holding every line's token-id list at once, as an earlier version of this
    function did, does not scale to that size."""
    eos_len = 1  # tokenizer.eos_token_id is a single token
    prefix_sums: list[int] = [0]
    cumulative = 0
    num_lines = 0
    with open(backbone_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            length = len(tokenizer.encode(line)) + eos_len
            cumulative += length
            prefix_sums.append(cumulative)
            num_lines += 1
            if cumulative >= token_budget:
                break
    return num_lines, prefix_sums


def iter_backbone_lines(backbone_path: Path, num_lines: int):
    """Pass 2 of 2: re-reads the same prefix of the backbone file (must apply
    identical stripping/blank-line-skipping as measure_backbone_lines, so the two
    passes see the same line boundaries), yielding raw text one line at a time.
    Re-tokenizing is a deliberate, cheap tradeoff for not holding every line's token
    ids in memory across both passes."""
    with open(backbone_path, encoding="utf-8") as f:
        yielded = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line
            yielded += 1
            if yielded >= num_lines:
                return


def build_occurrence_events(
    population: list[dict], total_tokens: int, base_seed: int
) -> list[dict]:
    """One event per (entity, occurrence), each carrying enough info to render and
    place it: entity_id, side, and the scheduled position (snapped to a backbone line
    boundary later, in assemble())."""
    events = []
    for i, entity in enumerate(population):
        rng = random.Random(base_seed * 1000 + i + 1)
        contested = entity["contested"]
        schedule = schedule_entity_occurrences(
            entity["entity_id"], contested["n_a"], contested["n_b"], total_tokens, rng
        )
        for event in schedule.events:
            events.append(
                {"entity_id": entity["entity_id"], "side": event.side, "position": event.position}
            )
    return events


class VariantLookup:
    """One round-robin variant cycler per (entity_id, side), so occurrences of the
    same entity/side don't repeat a variant until that side's whole bank
    (variants_per_side, S1.6/S1.7) is exhausted -- shared by both backbone paths
    below rather than duplicated."""

    def __init__(self, tokenizer, eos: int, rng: random.Random):
        self._tokenizer = tokenizer
        self._eos = eos
        self._rng = rng
        self._cyclers: dict[tuple[str, str], PoolCycler] = {}
        self.missing_events = 0

    def next_variant_tokens(self, entity_id: str, side: str) -> list[int] | None:
        key = (entity_id, side)
        if key not in self._cyclers:
            variants = load_variants(entity_id, side)
            if not variants:
                self.missing_events += 1
                return None
            self._cyclers[key] = PoolCycler(variants, self._rng)
        text = self._cyclers[key].draw()
        return self._tokenizer.encode(text) + [self._eos]


def assemble_from_wikitext_local(
    total_tokens: int, backbone_path: Path, out_dir: Path, base_seed: int = 0, dtype: str = "uint16"
) -> None:
    """Local-file, two-pass backbone path -- kept for network-free dev/smoke testing
    at small token budgets (S1.1: WikiText-103 only has ~110M tokens total, a hard
    ceiling this path can't exceed since it reads the file once straight through).
    For any real-scale run, use assemble_from_openwebtext_stream instead."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    eos = tokenizer.eos_token_id

    print(f"Loading population from {POPULATION_PATH}...")
    population = load_population()

    print(f"Pass 1/2: measuring backbone line lengths up to {total_tokens} tokens from {backbone_path}...")
    num_lines, prefix_sums = measure_backbone_lines(backbone_path, tokenizer, total_tokens)
    print(f"Backbone: {num_lines} lines, {prefix_sums[-1]} tokens")

    print("Scheduling occurrence events for every entity...")
    events = build_occurrence_events(population, prefix_sums[-1], base_seed)
    print(f"Scheduled {len(events)} occurrence events across {len(population)} entities")

    # Snap each event to "insert after this backbone line index" via the prefix-sum.
    events_by_line: dict[int, list[dict]] = {}
    for event in events:
        line_idx = bisect.bisect_right(prefix_sums, event["position"]) - 1
        line_idx = max(0, min(line_idx, num_lines - 1))
        events_by_line.setdefault(line_idx, []).append(event)

    variant_lookup = VariantLookup(tokenizer, eos, random.Random(base_seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    occurrence_log: list[dict] = []
    position_counter = 0
    np_dtype = np.dtype(dtype)

    print("Pass 2/2: re-tokenizing and streaming the packed stream to disk...")
    with open(out_dir / "shard_0000.bin", "wb") as out_f:

        def write_tokens(ids: list[int]) -> None:
            nonlocal position_counter
            out_f.write(np.array(ids, dtype=np_dtype).tobytes())
            position_counter += len(ids)

        for line_idx, line_text in enumerate(iter_backbone_lines(backbone_path, num_lines)):
            write_tokens(tokenizer.encode(line_text) + [eos])
            for event in events_by_line.get(line_idx, []):
                variant_tokens = variant_lookup.next_variant_tokens(event["entity_id"], event["side"])
                if variant_tokens is None:
                    continue
                occurrence_log.append(
                    {
                        "entity_id": event["entity_id"],
                        "side": event["side"],
                        "final_token_position": position_counter,
                    }
                )
                write_tokens(variant_tokens)

    _finish(out_dir, dtype, tokenizer.vocab_size, position_counter, occurrence_log, variant_lookup.missing_events)


def assemble_from_openwebtext_stream(
    total_tokens: int,
    out_dir: Path,
    base_seed: int = 0,
    dtype: str = "uint16",
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> None:
    """Single streaming pass over OpenWebText (S1.1, S1.7): supersedes the two-pass
    local-file design entirely. Occurrence events (build_occurrence_events) depend
    only on total_tokens, not backbone content, so they can be precomputed and
    sorted up front; the backbone is then streamed in ~chunk_bytes stages
    (preprocess/backbone_openwebtext.py). Each document is tokenized ONCE, at full
    length, via tokenize_with_unit_boundaries, which also returns paragraph- (or
    sentence-, fallback) boundary token indices -- the document's tokens are then
    written out in the slices those boundaries define, firing any event whose
    scheduled position has now been passed after each slice, not just once per
    whole document. (An earlier version tokenized each paragraph separately and
    concatenated the results; that silently drops the whitespace between
    paragraphs and breaks any BPE merge that should span the boundary --
    tokenize_with_unit_boundaries's docstring has the concrete before/after.)
    OpenWebText documents run up to ~100k characters, so checking only at document
    boundaries would let a scheduled position sit unfired for an entire long
    article; checking at paragraph boundaries instead bounds that delay to "the
    rest of the current paragraph." EOS is appended only after a document's final
    slice, so backbone documents remain single EOS-delimited units on disk even
    though insertion can now land between any two of their paragraphs. Stops once
    total_tokens have been written, regardless of how much of OpenWebText that
    consumed -- no local copy of the dataset is ever held.

    GPT2TokenizerFast (Rust-backed), not the slow GPT2Tokenizer the wikitext_local
    path uses: at real scale (2.5e9 tokens, millions of documents) the slow
    tokenizer's per-call Python overhead would dominate assembly time."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos = tokenizer.eos_token_id

    print(f"Loading population from {POPULATION_PATH}...")
    population = load_population()

    print(f"Scheduling occurrence events for every entity (total_tokens={total_tokens})...")
    events = sorted(build_occurrence_events(population, total_tokens, base_seed), key=lambda e: e["position"])
    print(f"Scheduled {len(events)} occurrence events across {len(population)} entities")

    variant_lookup = VariantLookup(tokenizer, eos, random.Random(base_seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    occurrence_log: list[dict] = []
    # Two separate counters, deliberately not one: event positions (S1.3) are
    # coordinates into the BACKBONE-only stream (matching total_tokens' definition
    # as a backbone budget, S1.7), while position_counter is the actual final
    # packed-stream offset (backbone + already-injected vignettes) used only for
    # logging and downstream analysis. Comparing events against position_counter
    # instead of backbone_position would let an injected vignette's own tokens
    # push position_counter past *later* events' scheduled positions, firing them
    # in a runaway cascade well before the backbone content that was supposed to
    # precede them has actually been written -- caught via a mocked dry run before
    # this shipped (see verification note in the commit/session log).
    backbone_position = 0
    position_counter = 0
    event_idx = 0
    np_dtype = np.dtype(dtype)

    print(f"Streaming OpenWebText in ~{chunk_bytes/1e9:.2f}GB text chunks until {total_tokens} backbone tokens are written...")
    with open(out_dir / "shard_0000.bin", "wb") as out_f:

        def write_backbone_tokens(ids: list[int]) -> None:
            nonlocal position_counter, backbone_position
            out_f.write(np.array(ids, dtype=np_dtype).tobytes())
            position_counter += len(ids)
            backbone_position += len(ids)

        def write_injected_tokens(ids: list[int]) -> None:
            nonlocal position_counter
            out_f.write(np.array(ids, dtype=np_dtype).tobytes())
            position_counter += len(ids)

        def fire_due_events() -> None:
            nonlocal event_idx
            while event_idx < len(events) and events[event_idx]["position"] <= backbone_position:
                event = events[event_idx]
                event_idx += 1
                variant_tokens = variant_lookup.next_variant_tokens(event["entity_id"], event["side"])
                if variant_tokens is None:
                    continue
                occurrence_log.append(
                    {
                        "entity_id": event["entity_id"],
                        "side": event["side"],
                        "final_token_position": position_counter,
                    }
                )
                # Leading EOS, not just the trailing one already in variant_tokens
                # (VariantLookup.next_variant_tokens): a whole-document boundary
                # already ends in EOS from that document's own closing slice, but
                # firing between two paragraphs of the SAME document (the point of
                # tokenize_with_unit_boundaries) does not -- intermediate paragraph
                # slices intentionally have no EOS between them, to keep a
                # multi-paragraph article one EOS-delimited unit when nothing
                # interrupts it. Without a leading EOS here, a vignette fired at an
                # internal paragraph boundary would run directly into the preceding
                # backbone text with no separator at all (verified: produced
                # "...here.As a leading figure..." with zero whitespace in a dry
                # run). A stray double-EOS at the rarer case of firing exactly on a
                # whole-document boundary is harmless (an empty zero-length
                # "document"), so no special-casing is needed here.
                write_injected_tokens([eos] + variant_tokens)

        done = False
        for chunk_documents in iter_openwebtext_chunks(chunk_bytes=chunk_bytes):
            for doc_text in chunk_documents:
                doc_text = doc_text.strip()
                if not doc_text:
                    continue
                token_ids, boundary_indices = tokenize_with_unit_boundaries(tokenizer, doc_text)
                prev_idx = 0
                for slice_num, boundary_idx in enumerate(boundary_indices):
                    slice_ids = token_ids[prev_idx:boundary_idx]
                    prev_idx = boundary_idx
                    if slice_num == len(boundary_indices) - 1:
                        slice_ids = slice_ids + [eos]  # only the document's last slice closes it
                    write_backbone_tokens(slice_ids)
                    fire_due_events()
                    if backbone_position >= total_tokens:
                        done = True
                        break
                if done:
                    break
            # chunk_documents (this chunk's raw text) goes out of scope here and is
            # discarded -- nothing beyond the current chunk is ever held at once.
            if done:
                break

    if event_idx < len(events):
        print(
            f"[WARN] OpenWebText stream exhausted (or total_tokens reached) with "
            f"{len(events) - event_idx} occurrence events still unfired -- unexpected "
            f"given OpenWebText's ~9.04B-token size comfortably exceeds total_tokens "
            f"in the intended use of this path; investigate before treating the corpus "
            f"as complete."
        )

    _finish(out_dir, dtype, tokenizer.vocab_size, position_counter, occurrence_log, variant_lookup.missing_events)


def _finish(
    out_dir: Path,
    dtype: str,
    vocab_size: int,
    position_counter: int,
    occurrence_log: list[dict],
    missing_variant_events: int,
) -> None:
    if missing_variant_events:
        print(
            f"[WARN] {missing_variant_events} occurrence events skipped -- no vignette "
            f"variants on disk yet for that (entity, side) (generation incomplete, see "
            f"experimental_plans.tex STATUS note)."
        )

    print(f"Final packed stream: {position_counter} tokens ({len(occurrence_log)} occurrences placed)")
    print(f"Wrote {out_dir / 'shard_0000.bin'}")

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"dtype": dtype, "vocab_size": vocab_size}, f, indent=2)
    print(f"Wrote {out_dir / 'meta.json'}")

    OCCURRENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OCCURRENCE_LOG_PATH, "w") as f:
        json.dump(occurrence_log, f, indent=2)
    print(f"Wrote {OCCURRENCE_LOG_PATH} ({len(occurrence_log)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--total-tokens", type=int, required=True,
        help="Backbone token budget (S1.7): final corpus is slightly larger once injections are added.",
    )
    parser.add_argument(
        "--backbone", choices=["openwebtext", "wikitext_local"], default="openwebtext",
        help="openwebtext: streamed, ~2GB chunks, runtime download only (default, real runs). "
             "wikitext_local: local data/raw/wikitext-103/ file, network-free dev testing only.",
    )
    parser.add_argument("--backbone-path", type=Path, default=DEFAULT_BACKBONE_PATH, help="wikitext_local only.")
    parser.add_argument(
        "--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES,
        help="openwebtext only: raw-text bytes per streamed stage (default ~2GB).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.backbone == "wikitext_local":
        assemble_from_wikitext_local(args.total_tokens, args.backbone_path, args.out_dir, base_seed=args.seed)
    else:
        assemble_from_openwebtext_stream(
            args.total_tokens, args.out_dir, base_seed=args.seed, chunk_bytes=args.chunk_bytes
        )


if __name__ == "__main__":
    main()
