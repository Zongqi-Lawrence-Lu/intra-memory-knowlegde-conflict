"""Pool-quality filter for surface-form divergence, experimental_plans.tex S1.3.

Replaces an earlier exhaustive "every value must diverge from every other value in the
pool" check, which scales as O(n^2) and rejected an unworkable fraction of a pool once
n grew into the hundreds (150-item pools lost 15-70% per round; a 1200-item name pool
lost 58%) -- stronger than the science actually needs, since only values that get
*directly compared* need to diverge from each other. Current policy (divergence.py has
the full rationale):

  (a) on-demand pairwise check (divergence.py:check_pair) for two *specific* values
      about to be directly compared -- not run here, run later by whatever
      constructs an actual competing pair (e.g. a future conflict-pair phase).
  (b) exact-duplicate ban -- already enforced at generation time
      (generation/dedup.py:clean_pool); this script does not need to re-check it.
  (c) has_too_common_first_token -- what this script actually applies: an O(n)
      per-value static filter, removing values whose distinguishing first token is a
      short/generic subword likely to collide with *whatever* it's later paired
      against.

    python -m preprocess.filter_divergence --relation alma_mater
    python -m preprocess.filter_divergence --all
    python -m preprocess.filter_divergence --all --apply   # actually rewrite pool files
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocess.divergence import has_too_common_first_token
from preprocess.schema import ENTITY_NAMES, RELATION_TYPES, RelationTypeSpec

DATA_POOLS_DIR = Path(__file__).parent / "data_pools"
TEMPLATES_DIR = DATA_POOLS_DIR / "templates"


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path) as f:
        return json.load(f)


def find_bad_values(rel: RelationTypeSpec) -> tuple[list[str], set[str]]:
    values = _load_json(DATA_POOLS_DIR / f"{rel.value_pool}.json")
    templates = _load_json(TEMPLATES_DIR / f"{rel.key}.json")

    names = _load_json(DATA_POOLS_DIR / f"{ENTITY_NAMES.name}.json")
    sample_name = names[0]

    bad: set[str] = set()
    for v in values:
        if has_too_common_first_token(
            templates["first_mention"], v, name=sample_name
        ):
            bad.add(v)
            continue
        for restatement in templates["restatements"]:
            if has_too_common_first_token(restatement, v):
                bad.add(v)
                break

    return values, bad


def filter_relation(rel: RelationTypeSpec, apply: bool) -> None:
    values, bad = find_bad_values(rel)
    if not bad:
        print(f"{rel.key}: 0/{len(values)} values have a too-common first token -- pool is clean")
        return

    print(f"{rel.key}: {len(bad)}/{len(values)} values have a too-common first token:")
    for v in sorted(bad):
        print(f"  - {v}")

    if apply:
        kept = [v for v in values if v not in bad]
        path = DATA_POOLS_DIR / f"{rel.value_pool}.json"
        with open(path, "w") as f:
            json.dump(kept, f, indent=2)
        print(
            f"{rel.key}: removed {len(bad)}, {len(kept)} remain -- top up with "
            f"`python -m preprocess.generate_pools --pool {rel.value_pool}` if below target"
        )
    else:
        print(f"{rel.key}: report only (pass --apply to rewrite the pool file)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--relation", choices=[r.key for r in RELATION_TYPES])
    group.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Actually rewrite the pool file (default: report only)")
    args = parser.parse_args()

    targets = RELATION_TYPES if args.all else [r for r in RELATION_TYPES if r.key == args.relation]
    for rel in targets:
        filter_relation(rel, apply=args.apply)


if __name__ == "__main__":
    main()
