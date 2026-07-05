"""Post-generation cleaning: removes duplicate and unusable entries from a generated
pool or template bank before it's accepted, experimental_plans.tex S1.2 (Pool and
Template Generation). Runs both inline (inside generate_pools.py/generate_templates.py,
per batch) and standalone as a CLI to re-clean an already-saved pool file:

    python -m preprocess.generation.dedup --pool entity_names
"""
from __future__ import annotations

import re

MAX_WORDS_NAME = 4  # "First Last" or a short compound name
MAX_WORDS_VALUE = 10  # generous cap for a 2-6-word value phrase plus connector words
REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "as an ai",
    "i'm sorry",
    "cannot provide",
    "language model",
    "i apologize",
)
_NAME_WORD_RE = re.compile(r"[A-Z][a-zA-Z'.-]*")


def is_usable_entry(entry: str, is_name_pool: bool) -> bool:
    """Filters empty/malformed entries and LLM refusal artifacts that occasionally
    surface in a batch response instead of a real entry."""
    e = entry.strip()
    if not e:
        return False
    lowered = e.lower()
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return False
    if e.endswith((".", ",", ";", ":")):
        return False
    words = e.split()
    max_words = MAX_WORDS_NAME if is_name_pool else MAX_WORDS_VALUE
    if not words or len(words) > max_words:
        return False
    if is_name_pool and not all(_NAME_WORD_RE.fullmatch(w) for w in words):
        return False
    return True


def clean_pool(entries: list[str], is_name_pool: bool) -> tuple[list[str], list[str]]:
    """Returns (kept, dropped). Dedups case/whitespace-insensitively (first occurrence
    wins) and drops anything failing is_usable_entry. `dropped` is returned for
    logging -- never silently discarded."""
    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for e in entries:
        normalized = " ".join(e.strip().split())
        if not is_usable_entry(normalized, is_name_pool):
            dropped.append(e)
            continue
        key = normalized.lower()
        if key in seen:
            dropped.append(e)
            continue
        seen.add(key)
        kept.append(normalized)

    return kept, dropped


def is_usable_template(template: str) -> bool:
    t = template.strip()
    return bool(t) and "{value}" in t and len(t) >= 8


def clean_templates(data: dict) -> tuple[dict, list[str]]:
    """Validates the first_mention template and dedups/filters the restatements list.
    Raises ValueError if first_mention itself is unusable (that one is not optional)."""
    first = str(data.get("first_mention", "")).strip()
    if not is_usable_template(first) or "{name}" not in first:
        raise ValueError(f"first_mention template invalid or missing a placeholder: {first!r}")

    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for t in data.get("restatements", []):
        normalized = " ".join(str(t).strip().split())
        if not is_usable_template(normalized) or "{name}" in normalized:
            dropped.append(t)
            continue
        key = normalized.lower()
        if key in seen:
            dropped.append(t)
            continue
        seen.add(key)
        kept.append(normalized)

    return {"first_mention": first, "restatements": kept}, dropped


def main() -> None:
    import argparse
    import json

    from preprocess.generate_pools import DATA_POOLS_DIR
    from preprocess.schema import POOL_BY_NAME

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, choices=sorted(POOL_BY_NAME))
    args = parser.parse_args()

    spec = POOL_BY_NAME[args.pool]
    path = DATA_POOLS_DIR / f"{spec.name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run generate_pools.py first.")

    with open(path) as f:
        entries = json.load(f)

    kept, dropped = clean_pool(entries, is_name_pool=spec.is_name_pool)
    print(f"{spec.name}: kept {len(kept)}/{len(entries)}, dropped {len(dropped)}")
    if dropped:
        preview = dropped[:20]
        print(f"Dropped (first {len(preview)}): {preview}")

    with open(path, "w") as f:
        json.dump(kept, f, indent=2)


if __name__ == "__main__":
    main()
