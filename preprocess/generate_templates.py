"""Generate paraphrase template banks per relation type by calling an LLM,
experimental_plans.tex S1.2 (Pool and Template Generation). Meant to be run manually,
later:

    python -m preprocess.generate_templates --relation alma_mater
    python -m preprocess.generate_templates --all
    python -m preprocess.generate_templates --relation alma_mater --dry-run

Output is NOT verified for surface-form divergence here -- that requires actual values
from the corresponding value pool. Run preprocess/divergence.py against
generate_pools.py's output afterward, once both templates and values exist.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from preprocess.generation.client import DEFAULT_MODEL, call_chat, preflight_check
from preprocess.generation.dedup import clean_templates
from preprocess.prompts import template_prompt
from preprocess.schema import RELATION_TYPES, RelationTypeSpec

TEMPLATES_DIR = Path(__file__).parent / "data_pools" / "templates"
MIN_RESTATEMENTS = 5  # S1.2: bank of >=5 surface realizations per relation type
MAX_ATTEMPTS = 5  # retries if the model returns malformed JSON or drops a placeholder


def _parse_json_object(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    if not isinstance(data, dict) or "first_mention" not in data or "restatements" not in data:
        raise ValueError(f"Unexpected shape: {data}")
    return data


def generate_templates_for_relation(
    rel: RelationTypeSpec, model: str, n_restatements: int = 6, dry_run: bool = False
) -> dict:
    prompt = template_prompt(rel, n_restatements=n_restatements)
    if dry_run:
        print(f"--- dry run: prompt for {rel.key} ---\n{prompt}")
        return {}

    preflight_check(model)  # fail fast on missing key/package, before any network call

    cleaned = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = call_chat(prompt, model=model)  # retried internally with backoff
        try:
            data = _parse_json_object(raw)
            cleaned, dropped = clean_templates(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] {rel.key}: attempt {attempt}/{MAX_ATTEMPTS} invalid ({e}); retrying")
            continue
        if dropped:
            print(f"[{rel.key}] dropped {len(dropped)} unusable/duplicate restatement(s)")
        if len(cleaned["restatements"]) < MIN_RESTATEMENTS:
            print(
                f"[WARN] {rel.key}: attempt {attempt}/{MAX_ATTEMPTS} only "
                f"{len(cleaned['restatements'])} usable restatements "
                f"(need >={MIN_RESTATEMENTS}); retrying"
            )
            cleaned = None
            continue
        break

    if cleaned is None:
        raise RuntimeError(f"{rel.key}: failed to get a valid template bank after {MAX_ATTEMPTS} attempts")

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEMPLATES_DIR / f"{rel.key}.json"
    with open(out_path, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"Wrote templates for {rel.key} to {out_path}")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--relation", choices=[r.key for r in RELATION_TYPES])
    group.add_argument("--all", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-restatements", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = RELATION_TYPES if args.all else [r for r in RELATION_TYPES if r.key == args.relation]
    for rel in targets:
        generate_templates_for_relation(
            rel, model=args.model, n_restatements=args.n_restatements, dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()
