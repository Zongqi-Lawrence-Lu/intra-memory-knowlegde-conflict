"""Entity population construction, experimental_plans.tex S1.2 (Entity and Document
Construction) + S1.6 (Corpus Scale: Conflict-Pair Phase). Pure code -- consumes pool/
template JSON produced by generate_pools.py/generate_templates.py, does not generate
any content itself.

Each entity has exactly 5 facts: 1 contested relation (two candidate values val_A/
val_B plus a split (n_A, n_B), S1.6) and K=4 background relations (single value each).
No anchor trait (dropped, S1 revision (e)) and no designed grid (dropped, S1 revision
(f)) -- population is built by walking the balanced assignment list from scheduler.py
and drawing values via round-robin pool cycling, so relation-type and value usage are
both exactly (or near-exactly) balanced without a per-cell configuration table.
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

from preprocess.divergence import check_pair
from preprocess.schema import ENTITY_NAMES, MENTOR_NAMES, RELATION_BY_KEY, RELATION_TYPES, RelationTypeSpec
from preprocess.scheduler import (
    REPLICATES_PER_CELL,
    SPLIT_LEVELS,
    background_relation_keys,
    build_assignment_list,
)

DATA_POOLS_DIR = Path(__file__).parent / "data_pools"
TEMPLATES_DIR = DATA_POOLS_DIR / "templates"
NUM_ENTITIES = len(RELATION_TYPES) * len(SPLIT_LEVELS) * REPLICATES_PER_CELL


def _load_pool(name: str) -> list[str]:
    path = DATA_POOLS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m preprocess.generate_pools --pool {name}` first."
        )
    with open(path) as f:
        return json.load(f)


def _load_templates(relation_key: str) -> dict:
    path = TEMPLATES_DIR / f"{relation_key}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m preprocess.generate_templates "
            f"--relation {relation_key}` first."
        )
    with open(path) as f:
        return json.load(f)


class PoolCycler:
    """Draws values with near-exact usage balance rather than i.i.d. resampling (S1.2
    value assignment). A reusable pool is walked in shuffled-lap order, reshuffling
    once exhausted, so usage counts differ by at most one across the population. A
    globally-unique pool (person-name-shaped values, S1.2 value-pool reuse policy) is
    drawn without replacement and raises once exhausted."""

    def __init__(self, values: list[str], rng: random.Random, unique: bool = False):
        self._rng = rng
        self._unique = unique
        if unique:
            self._remaining = list(values)
            self._rng.shuffle(self._remaining)
        else:
            self._pool = list(values)
            self._lap: list[str] = []

    def draw(self) -> str:
        if self._unique:
            if not self._remaining:
                raise ValueError("pool exhausted for a globally-unique draw")
            return self._remaining.pop()
        if not self._lap:
            self._lap = list(self._pool)
            self._rng.shuffle(self._lap)
        return self._lap.pop()


def draw_contested_pair(
    rel: RelationTypeSpec,
    first_mention_template: str,
    name: str,
    cycler: PoolCycler,
    max_attempts: int = 50,
) -> tuple[str, str]:
    """Draws val_A/val_B via the pool cycler, then checks the pair for on-demand
    surface-form divergence (S1.3(a)) -- the only divergence guarantee this design
    makes (no pool-wide pre-verification). A failing pair advances to the next cycler
    position rather than resampling blind."""
    for _ in range(max_attempts):
        val_a = cycler.draw()
        val_b = cycler.draw()
        if val_a == val_b:
            continue
        result = check_pair(first_mention_template, val_a, val_b, name=name)
        if result.diverges:
            return val_a, val_b
    raise RuntimeError(
        f"{rel.key}: could not find a diverging contested pair after {max_attempts} attempts"
    )


@dataclasses.dataclass
class ContestedAssignment:
    relation_key: str
    val_a: str
    val_b: str
    n_a: int
    n_b: int


@dataclasses.dataclass
class Entity:
    entity_id: str
    name: str
    contested: ContestedAssignment
    background: dict[str, str]  # relation_key -> value, K=4 background relations (S1.2)


def build_full_population(base_seed: int = 0) -> list[Entity]:
    """Builds the population by walking scheduler.build_assignment_list (balanced
    contested relation_key x split level, S1.6) and, for each entity, drawing a unique
    name, a diverging contested pair, and K=4 background values via the fixed cyclic
    offset from the contested type (S1.2) -- no separate randomization for background
    relation-type identity."""
    rng = random.Random(base_seed)

    names_pool = _load_pool(ENTITY_NAMES.name)
    if len(names_pool) < NUM_ENTITIES:
        raise ValueError(
            f"{ENTITY_NAMES.name} has only {len(names_pool)} entries, need >= "
            f"{NUM_ENTITIES} (S1.6: {len(RELATION_TYPES)} relation types x "
            f"{len(SPLIT_LEVELS)} split levels x {REPLICATES_PER_CELL} replicates)."
        )
    name_cycler = PoolCycler(names_pool, rng, unique=True)

    assignment = build_assignment_list(rng)
    value_cyclers: dict[str, PoolCycler] = {}
    templates_cache: dict[str, dict] = {}

    def get_cycler(relation_key: str) -> PoolCycler:
        if relation_key not in value_cyclers:
            rel = RELATION_BY_KEY[relation_key]
            pool = _load_pool(rel.value_pool)
            unique = rel.value_pool == MENTOR_NAMES.name
            value_cyclers[relation_key] = PoolCycler(pool, rng, unique=unique)
        return value_cyclers[relation_key]

    def get_first_mention(relation_key: str) -> str:
        if relation_key not in templates_cache:
            templates_cache[relation_key] = _load_templates(relation_key)
        return templates_cache[relation_key]["first_mention"]

    entities = []
    for i, (contested_key, (n_a, n_b)) in enumerate(assignment):
        name = name_cycler.draw()
        rel = RELATION_BY_KEY[contested_key]
        val_a, val_b = draw_contested_pair(
            rel, get_first_mention(contested_key), name, get_cycler(contested_key)
        )

        background = {
            bkey: get_cycler(bkey).draw() for bkey in background_relation_keys(contested_key)
        }

        entities.append(
            Entity(
                entity_id=f"entity_{i:04d}",
                name=name,
                contested=ContestedAssignment(
                    relation_key=contested_key, val_a=val_a, val_b=val_b, n_a=n_a, n_b=n_b
                ),
                background=background,
            )
        )
    return entities


def save_population(entities: list[Entity], out_path: str | Path) -> None:
    payload = [
        {
            "entity_id": e.entity_id,
            "name": e.name,
            "contested": dataclasses.asdict(e.contested),
            "background": e.background,
        }
        for e in entities
    ]
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
