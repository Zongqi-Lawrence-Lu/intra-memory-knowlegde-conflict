"""Generate name/value pools by calling an LLM, experimental_plans.tex S1.2 (Pool and
Template Generation). This script only calls out to an LLM to produce content -- it
does not hardcode any names/institutions/etc. itself. Meant to be run manually, later,
once OPENAI_API_KEY and network access are available:

    python -m preprocess.generate_pools --pool entity_names
    python -m preprocess.generate_pools --pool alma_mater_values --n 150
    python -m preprocess.generate_pools --pool entity_names --dry-run   # inspect prompt only

Resumable: re-running with the same --pool tops up an existing (possibly partial) pool
file rather than starting over. After all pools are generated, run validate_pools.py to
check disjointness before using them to build documents.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from preprocess.generation.client import (
    DEFAULT_MODEL,
    batch_size_with_leeway,
    call_chat,
    preflight_check,
)
from preprocess.generation.dedup import clean_pool
from preprocess.prompts import pool_prompt
from preprocess.schema import POOL_BY_NAME, PoolSpec

DATA_POOLS_DIR = Path(__file__).parent / "data_pools"
MAX_ITERATIONS = 25


def _load_existing(name: str) -> list[str]:
    path = DATA_POOLS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _parse_json_array(raw: str) -> list[str]:
    raw = raw.strip()
    # Strip accidental markdown code fences despite the prompt asking for none.
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data)}")
    return [str(x).strip() for x in data if str(x).strip()]


def generate_pool(
    spec: PoolSpec, target_n: int | None, model: str, dry_run: bool = False
) -> list[str]:
    target_n = target_n or spec.target_size
    if not dry_run:
        preflight_check(model)  # fail fast on missing key/package, before any network call

    existing = _load_existing(spec.name)
    forbidden = {v.lower() for v in existing}
    for other in spec.disjoint_from:
        forbidden |= {v.lower() for v in _load_existing(other)}

    iteration = 0
    while len(existing) < target_n and iteration < MAX_ITERATIONS:
        remaining = target_n - len(existing)
        batch_n = batch_size_with_leeway(remaining)  # >=20% leeway, capped for length safety
        prompt = pool_prompt(spec, n=batch_n, existing=existing, forbidden=sorted(forbidden))

        if dry_run:
            print(f"--- dry run: prompt for {spec.name} (iteration {iteration}) ---\n{prompt}")
            return existing

        raw = call_chat(prompt, model=model)  # retried internally with backoff
        try:
            candidates = _parse_json_array(raw)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] iteration {iteration}: failed to parse response ({e}); retrying")
            iteration += 1
            continue

        cleaned, batch_dropped = clean_pool(candidates, is_name_pool=spec.is_name_pool)
        if batch_dropped:
            print(f"[{spec.name}] iteration {iteration}: dropped {len(batch_dropped)} unusable/malformed candidate(s)")

        new_count = 0
        for c in cleaned:
            key = c.lower()
            if key in forbidden:
                continue
            forbidden.add(key)
            existing.append(c)
            new_count += 1
            if len(existing) >= target_n:
                break

        print(f"[{spec.name}] iteration {iteration}: +{new_count} new (total {len(existing)}/{target_n})")
        iteration += 1

    if len(existing) < target_n:
        print(
            f"[WARN] {spec.name}: stopped after {iteration} iterations with only "
            f"{len(existing)}/{target_n} -- consider raising MAX_ITERATIONS or relaxing "
            f"the category description in schema.py"
        )

    DATA_POOLS_DIR.mkdir(exist_ok=True)
    out_path = DATA_POOLS_DIR / f"{spec.name}.json"
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Wrote {len(existing)} entries to {out_path}")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", required=True, choices=sorted(POOL_BY_NAME), help="Pool name from preprocess/schema.py"
    )
    parser.add_argument("--n", type=int, default=None, help="Override target size (default: PoolSpec.target_size)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Print the next prompt and exit, no API call")
    args = parser.parse_args()

    spec = POOL_BY_NAME[args.pool]
    generate_pool(spec, target_n=args.n, model=args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
