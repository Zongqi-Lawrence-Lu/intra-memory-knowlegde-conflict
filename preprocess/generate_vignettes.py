"""Population-scale LLM vignette generation, experimental_plans.tex S1.2/S1.4 (S1
revision (g)): one call per (entity, side), requesting a per-(entity, side) variant
count -- 8 total per entity, split unevenly across sides in proportion to that
entity's (n_a, n_b) via preprocess/scheduler.py:variants_per_side, not a flat V for
every side. Run concurrently (bounded by a semaphore) rather than one at a time.
Requires OPENAI_API_KEY and results/population.json
(preprocess/entities.py:save_population), whose "contested" dict must already carry
v_a/v_b (added by the T=32->T=80 migration; variants_per_side for any population
built fresh going forward).

    python -m preprocess.generate_vignettes
    python -m preprocess.generate_vignettes --concurrency 20

Output: preprocess/data_pools/vignettes/<entity_id>_<side>.json, one file per
(entity, side) so concurrent tasks never contend over the same file. Resumable: a
file already on disk with >= that pair's target variant count (v_a or v_b) is
skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from preprocess.generation.client import DEFAULT_MODEL, preflight_check
from preprocess.prompts import vignette_prompt
from preprocess.schema import RELATION_BY_KEY

RESULTS_POPULATION_PATH = Path(__file__).parent.parent / "results" / "population.json"
VIGNETTES_DIR = Path(__file__).parent / "data_pools" / "vignettes"

DEFAULT_TEMPERATURE = 1.15
DEFAULT_CONCURRENCY = 10  # conservative default -- actual account-level rate limits
# are not known in advance; raise if your tier supports more, backoff below handles
# 429s either way rather than crashing the whole job
MAX_ATTEMPTS = 3  # whole-call retries if too few variants pass the value-fidelity check
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_MULTIPLIER = 2.0


def _parse_json_array(raw: str) -> list[str]:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array, got {type(data)}")
    return [str(x).strip() for x in data]


def load_population() -> list[dict]:
    if not RESULTS_POPULATION_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_POPULATION_PATH} not found -- run entities.py:build_full_population "
            f"+ save_population first."
        )
    with open(RESULTS_POPULATION_PATH) as f:
        return json.load(f)


def load_variants(entity_id: str, side: str) -> list[str]:
    path = VIGNETTES_DIR / f"{entity_id}_{side}.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f).get("variants", [])
    except (json.JSONDecodeError, OSError):
        return []


def facts_for_side(entity: dict, side: str) -> list[tuple[str, str]]:
    contested = entity["contested"]
    contested_label = RELATION_BY_KEY[contested["relation_key"]].label
    contested_value = contested["val_a"] if side == "A" else contested["val_b"]
    facts = [(contested_label, contested_value)]
    for relation_key, value in entity["background"].items():
        facts.append((RELATION_BY_KEY[relation_key].label, value))
    return facts


def target_variants(entity: dict, side: str) -> int:
    """Per-(entity, side) variant target from population.json's v_a/v_b
    (preprocess/scheduler.py:variants_per_side) -- not a flat constant."""
    contested = entity["contested"]
    return contested["v_a"] if side == "A" else contested["v_b"]


def already_done(entity_id: str, side: str, target: int) -> bool:
    path = VIGNETTES_DIR / f"{entity_id}_{side}.json"
    if not path.exists():
        return False
    try:
        data = json.load(open(path))
        return len(data.get("variants", [])) >= target
    except (json.JSONDecodeError, OSError):
        return False


async def generate_one(
    client,
    entity: dict,
    side: str,
    n_variants: int,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, int]:
    """Returns (entity_id, side, n_variants_written). n_variants is this pair's
    target (population.json v_a/v_b). Tops up rather than replaces: existing
    variants already on disk (valid content -- facts/values don't change when only
    a target count does, e.g. after a T/variants_per_side revision) are kept, and
    only the shortfall is requested and generated, shown to the model so it avoids
    near-duplicating them. There is no separate lower floor beyond the target
    itself, since the target is already the minimum a given split level needs
    (e.g. 1 for the low-occurrence side of an extreme split)."""
    entity_id = entity["entity_id"]
    name = entity["name"]
    facts = facts_for_side(entity, side)
    existing = load_variants(entity_id, side)
    needed = n_variants - len(existing)
    if needed <= 0:
        return entity_id, side, len(existing)
    prompt = vignette_prompt(name, facts, n_variants=needed, existing=existing)

    async with semaphore:
        backoff = INITIAL_BACKOFF_SECONDS
        variants: list[str] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content or ""
                candidates = _parse_json_array(raw)
            except Exception as e:  # noqa: BLE001 -- retry on anything transient
                print(f"[WARN] {entity_id} side {side} attempt {attempt}/{MAX_ATTEMPTS}: {e}")
                await asyncio.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
                continue

            # Case-insensitive: a common-noun value (e.g. a field-of-expertise term)
            # gets naturally case-folded mid-sentence by a fluent writer ("a focus on
            # paleoclimatology") even though it's capitalized in the pool -- exact case
            # was never the requirement, only that the value's *position* in the text
            # is findable for scoring, which a case-insensitive match still locates.
            good = [c for c in candidates if all(value.lower() in c.lower() for _, value in facts)]
            variants = good if len(good) > len(variants) else variants
            if len(variants) >= needed:
                break
            if attempt < MAX_ATTEMPTS:
                print(
                    f"[WARN] {entity_id} side {side} attempt {attempt}/{MAX_ATTEMPTS}: "
                    f"only {len(good)}/{len(candidates)} variants passed value-fidelity check; retrying"
                )

        if len(variants) < needed:
            print(
                f"[WARN] {entity_id} side {side}: only {len(variants)}/{needed} new variants "
                f"survived after {MAX_ATTEMPTS} attempts, accepting anyway"
            )

    merged = existing + variants
    VIGNETTES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIGNETTES_DIR / f"{entity_id}_{side}.json"
    with open(out_path, "w") as f:
        json.dump({"entity_id": entity_id, "side": side, "variants": merged}, f, indent=2)
    return entity_id, side, len(merged)


async def run(model: str, temperature: float, concurrency: int, limit: int | None = None) -> None:
    import openai

    population = load_population()
    if limit is not None:
        population = population[:limit]
    tasks_needed = [
        (entity, side)
        for entity in population
        for side in ("A", "B")
        if not already_done(entity["entity_id"], side, target_variants(entity, side))
    ]
    total = len(population) * 2
    skipped = total - len(tasks_needed)
    print(f"Population: {len(population)} entities, {total} (entity, side) pairs total")
    print(f"Already done (resumed): {skipped}; remaining: {len(tasks_needed)}")
    if not tasks_needed:
        print("Nothing to do.")
        return

    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    start = time.time()
    completed = 0
    lock = asyncio.Lock()

    async def _run_one(entity, side):
        nonlocal completed
        n_variants = target_variants(entity, side)
        result = await generate_one(client, entity, side, n_variants, model, temperature, semaphore)
        async with lock:
            completed += 1
            if completed % 50 == 0 or completed == len(tasks_needed):
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(tasks_needed) - completed) / rate if rate > 0 else float("inf")
                print(
                    f"[{completed}/{len(tasks_needed)}] elapsed={elapsed:.0f}s "
                    f"rate={rate:.2f}/s eta={eta:.0f}s"
                )
        return result

    # return_exceptions=True: one task's uncaught exception (e.g. a disk-write error)
    # must not cancel every other in-flight task in a 3,000+-call job -- log it and
    # keep going, rather than losing an hour of progress to one bad task.
    raw_results = await asyncio.gather(*(_run_one(e, s) for e, s in tasks_needed), return_exceptions=True)
    results = []
    n_short = 0
    for (entity, side), r in zip(tasks_needed, raw_results):
        if isinstance(r, Exception):
            print(f"[ERROR] {entity['entity_id']} side {side}: unhandled {r!r} -- left incomplete on disk, rerun to retry")
            continue
        results.append(r)
        _, _, n = r
        if n < target_variants(entity, side):
            n_short += 1
    print(
        f"\nDone: {len(results)}/{len(tasks_needed)} generated ({len(tasks_needed) - len(results)} errored), "
        f"{n_short} came in under their per-pair target."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N entities (testing).")
    args = parser.parse_args()

    preflight_check(args.model)
    asyncio.run(run(args.model, args.temperature, args.concurrency, args.limit))


if __name__ == "__main__":
    main()
