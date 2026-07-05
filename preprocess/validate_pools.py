"""Cross-pool disjointness + WikiText-103 overlap checks, experimental_plans.tex S1.2.

Run after generate_pools.py has produced preprocess/data_pools/<pool>.json for every
pool in preprocess/schema.py. Pure post-hoc validation -- calls no LLM.

    python -m preprocess.validate_pools
    python -m preprocess.validate_pools --wikitext-path data/raw/wikitext-103/wiki.train.tokens
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from preprocess.schema import ALL_POOLS, PoolSpec

DATA_POOLS_DIR = Path(__file__).parent / "data_pools"


def load_pool(spec: PoolSpec) -> list[str]:
    path = DATA_POOLS_DIR / f"{spec.name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run generate_pools.py for '{spec.name}' first.")
    with open(path) as f:
        return json.load(f)


def check_internal_uniqueness(values: list[str]) -> list[str]:
    seen = Counter(v.strip().lower() for v in values)
    return [v for v, c in seen.items() if c > 1]


def check_cross_pool_overlap(pools: dict[str, list[str]]) -> dict[tuple[str, str], set[str]]:
    """All-pairs overlap check across every pool (S1.2: each relation's value pool is
    "checked against every other value pool", and name pools must be globally unique)."""
    overlaps = {}
    names = list(pools)
    normed = {n: {v.strip().lower() for v in pools[n]} for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = normed[names[i]] & normed[names[j]]
            if shared:
                overlaps[(names[i], names[j])] = shared
    return overlaps


def extract_wikitext_proper_nouns(wikitext_path: str | Path, max_chars: int | None = None) -> set[str]:
    """Heuristic proper-noun extractor: runs of 2+ capitalized words, skipping each
    sentence's first word (which is capitalized purely by position). No NER/LM model is
    loaded here -- pure regex -- consistent with CLAUDE.md S5 (no model loading/running
    without a GPU present). This is a recall-oriented screen (favors over-flagging), not
    a precision-oriented NER system -- treat hits as "review before using", not gospel.
    """
    text = Path(wikitext_path).read_text(errors="ignore")
    if max_chars:
        text = text[:max_chars]
    found: set[str] = set()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        words = sent.split()
        if len(words) < 3:
            continue
        rest = " ".join(words[1:])  # drop sentence-initial word (position-only capital)
        for m in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", rest):
            phrase = m.group(1)
            if len(phrase.split()) >= 2:
                found.add(phrase)
    return found


def check_wikitext_overlap(spec: PoolSpec, values: list[str], wikitext_proper_nouns: set[str]) -> list[str]:
    if not spec.check_wikitext_overlap:
        return []
    lowered = {p.lower() for p in wikitext_proper_nouns}
    return [v for v in values if v.strip().lower() in lowered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wikitext-path",
        default=None,
        help="Path to raw WikiText-103 text for overlap screening; skipped if omitted.",
    )
    args = parser.parse_args()

    pools: dict[str, list[str]] = {}
    missing = []
    for spec in ALL_POOLS:
        try:
            pools[spec.name] = load_pool(spec)
        except FileNotFoundError as e:
            missing.append(str(e))
    if missing:
        for m in missing:
            print(f"[MISSING] {m}")

    ok = not missing
    for spec in ALL_POOLS:
        if spec.name not in pools:
            continue
        dups = check_internal_uniqueness(pools[spec.name])
        if dups:
            ok = False
            print(f"[FAIL] {spec.name}: {len(dups)} internal duplicate(s), e.g. {dups[:5]}")
        if len(pools[spec.name]) < spec.target_size:
            print(f"[WARN] {spec.name}: only {len(pools[spec.name])}/{spec.target_size} generated so far")

    if len(pools) > 1:
        overlaps = check_cross_pool_overlap(pools)
        for (a, b), shared in overlaps.items():
            ok = False
            print(f"[FAIL] cross-pool overlap between {a} and {b}: {list(shared)[:5]}")

    if args.wikitext_path:
        wt_nouns = extract_wikitext_proper_nouns(args.wikitext_path)
        print(f"Extracted {len(wt_nouns)} candidate proper-noun phrases from {args.wikitext_path}")
        for spec in ALL_POOLS:
            if spec.name not in pools:
                continue
            hits = check_wikitext_overlap(spec, pools[spec.name], wt_nouns)
            if hits:
                ok = False
                print(f"[FAIL] {spec.name} overlaps WikiText-103 proper nouns: {hits[:5]}")
    else:
        print("[SKIP] WikiText-103 overlap check (no --wikitext-path given)")

    print("PASS" if ok else "FAIL -- see above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
