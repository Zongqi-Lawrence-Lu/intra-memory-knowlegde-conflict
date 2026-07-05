"""Render sample vignettes (entity + contested pair + K=4 background facts) for visual
QA before running full-scale pool/template generation, experimental_plans.tex S1.2.
Pure code -- consumes whatever pools/templates already exist under data_pools/, does
not call any LLM itself. Does not require the full balanced population (S1.6) --
draws a small ad-hoc sample instead, but reuses the same background-offset rule and
divergence-checked pair-drawing as the real population builder (entities.py) so the
sample is representative of what the real corpus will render.

    python -m preprocess.render_sample --n 5
    python -m preprocess.render_sample --n 5 --contested birthplace
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from preprocess.entities import PoolCycler, draw_contested_pair
from preprocess.render_utils import BiographyRenderer
from preprocess.schema import ENTITY_NAMES, MENTOR_NAMES, RELATION_BY_KEY, RELATION_TYPES
from preprocess.scheduler import background_relation_keys

DATA_POOLS_DIR = Path(__file__).parent / "data_pools"
TEMPLATES_DIR = DATA_POOLS_DIR / "templates"


def _load_pool(name: str) -> list[str]:
    path = DATA_POOLS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- generate it first.")
    with open(path) as f:
        return json.load(f)


def _load_templates(relation_key: str) -> dict:
    path = TEMPLATES_DIR / f"{relation_key}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- generate templates for {relation_key} first.")
    with open(path) as f:
        return json.load(f)


def render_vignettes(n: int, contested_key: str | None, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    names = rng.sample(_load_pool(ENTITY_NAMES.name), n)
    candidate_keys = [contested_key] if contested_key else [r.key for r in RELATION_TYPES]

    cyclers: dict[str, PoolCycler] = {}
    templates_cache: dict[str, dict] = {}

    def get_cycler(key: str) -> PoolCycler:
        if key not in cyclers:
            pool = _load_pool(RELATION_BY_KEY[key].value_pool)
            unique = RELATION_BY_KEY[key].value_pool == MENTOR_NAMES.name
            cyclers[key] = PoolCycler(pool, rng, unique=unique)
        return cyclers[key]

    def get_templates(key: str) -> dict:
        if key not in templates_cache:
            templates_cache[key] = _load_templates(key)
        return templates_cache[key]

    vignettes = []
    for name in names:
        key = rng.choice(candidate_keys)
        rel = RELATION_BY_KEY[key]
        val_a, val_b = draw_contested_pair(rel, get_templates(key)["first_mention"], name, get_cycler(key))

        background_keys = background_relation_keys(key)
        background_values = {bkey: get_cycler(bkey).draw() for bkey in background_keys}

        all_keys = [key, *background_keys]
        templates_by_key = {k: get_templates(k) for k in all_keys}
        # One renderer per entity, reused across every occurrence instance it gets
        # rendered for (here, side A then side B) -- its cyclers are stateful, so the
        # no-repeat-until-exhausted guarantee (BiographyRenderer) holds across calls,
        # not just within one.
        renderer = BiographyRenderer(name, all_keys, templates_by_key, rng)
        instance_a = renderer.render({key: val_a, **background_values})
        instance_b = renderer.render({key: val_b, **background_values})

        vignettes.append(
            {
                "name": name,
                "contested_relation": key,
                "val_a": val_a,
                "val_b": val_b,
                "background": background_values,
                "instance_side_a": instance_a,
                "instance_side_b": instance_b,
            }
        )
    return vignettes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contested", choices=[r.key for r in RELATION_TYPES], default=None,
                         help="Fix the contested relation type for every sample vignette (default: random per vignette).")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vignettes = render_vignettes(args.n, args.contested, args.seed)
    for i, v in enumerate(vignettes):
        print(f"\n===== Entity {i + 1}: {v['name']} (contested: {v['contested_relation']}, "
              f"A={v['val_a']!r}, B={v['val_b']!r}) =====")
        print("-- side A occurrence instance --")
        for s in v["instance_side_a"]:
            print(f"  {s}")
        print("-- side B occurrence instance --")
        for s in v["instance_side_b"]:
            print(f"  {s}")


if __name__ == "__main__":
    main()
