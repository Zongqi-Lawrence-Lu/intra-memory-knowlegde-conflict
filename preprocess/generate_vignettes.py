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

--output-dir writes to a different directory instead (e.g. an additional batch kept
separate from the accepted main bank until explicitly merged). The main bank
(VIGNETTES_DIR) is always read as extra de-dup context in that case -- shown to the
model alongside whatever's already in --output-dir -- so a separate batch still
avoids near-duplicating already-accepted paragraphs, even though it is written to
its own files and the main bank is never modified.

    python -m preprocess.generate_vignettes --output-dir preprocess/data_pools/vignettes_batch2

For a further batch, pass each earlier batch's directory via --context-dir (repeatable)
so the new batch's de-dup context spans every prior batch, not just the main bank:

    python -m preprocess.generate_vignettes --output-dir preprocess/data_pools/vignettes_batch3 \
        --context-dir preprocess/data_pools/vignettes_batch2

--total-variants overrides the per-(entity, side) target for this run, recomputed via
the same proportional split (preprocess/scheduler.py:variants_per_side) rather than
reading the fixed 8-total v_a/v_b baked into population.json -- e.g. --total-variants 16
requests a fresh batch of 16 new variants/entity (same A/B proportions, just doubled)
in one job instead of two separate +8 runs.
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
from preprocess.scheduler import variants_per_side

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


def load_variants(entity_id: str, side: str, directory: Path = VIGNETTES_DIR) -> list[str]:
    path = directory / f"{entity_id}_{side}.json"
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


def target_variants(entity: dict, side: str, total_variants: int | None = None) -> int:
    """Per-(entity, side) variant target. Defaults to population.json's stored v_a/v_b
    (the fixed 8-total split); total_variants overrides this by recomputing the same
    proportional split (preprocess/scheduler.py:variants_per_side) at a different
    total, e.g. 16 for a single double-size batch."""
    contested = entity["contested"]
    if total_variants is None:
        return contested["v_a"] if side == "A" else contested["v_b"]
    v_a, v_b = variants_per_side(contested["n_a"], contested["n_b"], total_variants)
    return v_a if side == "A" else v_b


def already_done(entity_id: str, side: str, target: int, directory: Path = VIGNETTES_DIR) -> bool:
    path = directory / f"{entity_id}_{side}.json"
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
    output_dir: Path = VIGNETTES_DIR,
    context_dirs: tuple[Path, ...] = (),
) -> tuple[str, str, int]:
    """Returns (entity_id, side, n_variants_written). n_variants is this pair's
    target (population.json v_a/v_b). Tops up rather than replaces: existing
    variants already on disk at output_dir (valid content -- facts/values don't
    change when only a target count does, e.g. after a T/variants_per_side
    revision) are kept, and only the shortfall is requested and generated. There is
    no separate lower floor beyond the target itself, since the target is already
    the minimum a given split level needs (e.g. 1 for the low-occurrence side of an
    extreme split).

    De-dup context shown to the model (the prompt's `existing`) is the union of
    what's already at output_dir plus, when output_dir is not the main bank, the
    main bank's own accepted variants, plus any other prior batches passed via
    context_dirs (e.g. an already-generated batch2 when generating batch3) -- so a
    separate batch still avoids near-duplicating paragraphs from every earlier
    batch, even though it is never written into any of them."""
    entity_id = entity["entity_id"]
    name = entity["name"]
    facts = facts_for_side(entity, side)
    output_existing = load_variants(entity_id, side, output_dir)
    needed = n_variants - len(output_existing)
    if needed <= 0:
        return entity_id, side, len(output_existing)
    context_existing = list(output_existing)
    seen = set(output_existing)
    for other_dir in (VIGNETTES_DIR, *context_dirs):
        if other_dir == output_dir:
            continue
        for v in load_variants(entity_id, side, other_dir):
            if v not in seen:
                context_existing.append(v)
                seen.add(v)
    prompt = vignette_prompt(name, facts, n_variants=needed, existing=context_existing)

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

    merged = output_existing + variants
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{entity_id}_{side}.json"
    with open(out_path, "w") as f:
        json.dump({"entity_id": entity_id, "side": side, "variants": merged}, f, indent=2)
    return entity_id, side, len(merged)


async def run(
    model: str,
    temperature: float,
    concurrency: int,
    limit: int | None = None,
    output_dir: Path = VIGNETTES_DIR,
    context_dirs: tuple[Path, ...] = (),
    total_variants: int | None = None,
) -> None:
    import openai

    population = load_population()
    if limit is not None:
        population = population[:limit]
    tasks_needed = [
        (entity, side)
        for entity in population
        for side in ("A", "B")
        if not already_done(entity["entity_id"], side, target_variants(entity, side, total_variants), output_dir)
    ]
    total = len(population) * 2
    skipped = total - len(tasks_needed)
    print(f"Output dir: {output_dir}" + (" (separate from main bank)" if output_dir != VIGNETTES_DIR else ""))
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
        n_variants = target_variants(entity, side, total_variants)
        result = await generate_one(
            client, entity, side, n_variants, model, temperature, semaphore, output_dir, context_dirs
        )
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
        if n < target_variants(entity, side, total_variants):
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VIGNETTES_DIR,
        help=(
            "Write to a different directory instead of the main bank (e.g. an "
            "additional batch kept separate until explicitly merged). The main "
            "bank is still read as de-dup context in that case; it is never "
            "written to."
        ),
    )
    parser.add_argument(
        "--context-dir",
        type=Path,
        action="append",
        default=[],
        dest="context_dirs",
        help=(
            "Additional directory to read as de-dup context (besides the main "
            "bank and --output-dir itself), e.g. a prior batch's directory when "
            "generating a further batch. Repeatable."
        ),
    )
    parser.add_argument(
        "--total-variants",
        type=int,
        default=None,
        help=(
            "Override the per-(entity, side) target for this run, recomputed via "
            "the same proportional A/B split at this total instead of "
            "population.json's stored (fixed 8-total) v_a/v_b -- e.g. 16 for a "
            "single double-size batch."
        ),
    )
    args = parser.parse_args()

    preflight_check(args.model)
    asyncio.run(
        run(
            args.model,
            args.temperature,
            args.concurrency,
            args.limit,
            args.output_dir,
            tuple(args.context_dirs),
            args.total_variants,
        )
    )


if __name__ == "__main__":
    main()
