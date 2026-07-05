"""Corpus assembly: interleaves the WikiText-103 backbone with entities' LLM-authored
vignette occurrences into packed token shards, experimental_plans.tex S1.7 (Corpus
Assembly). Pure code -- no LLM calls; consumes results/population.json,
data_pools/vignettes/<entity_id>_<side>.json, and data/raw/wikitext-103/wiki.train.txt.

    python -m preprocess.assemble_corpus --total-tokens 200000 --out-dir data/processed/dev-run

Output: data/processed/<name>/meta.json + *.bin shard(s), matching the on-disk contract
training/data.py:PackedTokenDataset already expects. Also writes
results/occurrence_log.json (small, tracked) recording every occurrence event's final
token position, for exposure-count reconstruction (S1.3) and for the training loop to
derive dense-checkpoint step windows (S1.8) from token position + batch/block size.

Implementation note on `--total-tokens`: this is the BACKBONE token budget, not the
final packed-stream length -- injected vignette occurrences add their own tokens on
top (a small fraction of the total, S1.7), so the final corpus is slightly longer than
`--total-tokens`. Scheduler positions (S1.3) are coordinates into this backbone-length
stream, snapped to the nearest backbone line boundary (S1.7) rather than an arbitrary
mid-line token offset.
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
from pathlib import Path

import numpy as np
from transformers import GPT2Tokenizer

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


def assemble(
    total_tokens: int, backbone_path: Path, out_dir: Path, base_seed: int = 0, dtype: str = "uint16"
) -> None:
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

    # One round-robin variant cycler per (entity_id, side), so occurrences of the same
    # entity/side don't repeat a variant until the whole bank is exhausted (S1.2).
    cyclers: dict[tuple[str, str], PoolCycler] = {}
    rng = random.Random(base_seed)
    missing_variant_events = 0

    def next_variant_tokens(entity_id: str, side: str) -> list[int] | None:
        key = (entity_id, side)
        if key not in cyclers:
            variants = load_variants(entity_id, side)
            if not variants:
                return None
            cyclers[key] = PoolCycler(variants, rng)
        text = cyclers[key].draw()
        return tokenizer.encode(text) + [eos]

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
                variant_tokens = next_variant_tokens(event["entity_id"], event["side"])
                if variant_tokens is None:
                    missing_variant_events += 1
                    continue
                occurrence_log.append(
                    {
                        "entity_id": event["entity_id"],
                        "side": event["side"],
                        "final_token_position": position_counter,
                    }
                )
                write_tokens(variant_tokens)

    if missing_variant_events:
        print(
            f"[WARN] {missing_variant_events} occurrence events skipped -- no vignette "
            f"variants on disk yet for that (entity, side) (generation incomplete, see "
            f"experimental_plans.tex STATUS note)."
        )

    print(f"Final packed stream: {position_counter} tokens ({len(occurrence_log)} occurrences placed)")
    print(f"Wrote {out_dir / 'shard_0000.bin'}")

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"dtype": dtype, "vocab_size": tokenizer.vocab_size}, f, indent=2)
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
    parser.add_argument("--backbone-path", type=Path, default=DEFAULT_BACKBONE_PATH)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assemble(args.total_tokens, args.backbone_path, args.out_dir, base_seed=args.seed)


if __name__ == "__main__":
    main()
