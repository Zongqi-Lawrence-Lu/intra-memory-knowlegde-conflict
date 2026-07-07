"""Rescales results/population.json's exposure counts to a new total budget T, for
the exposure-budget pilot (experimental_plans.tex Sec. seeds -- T=80 was a reasoned,
never pilot-validated working estimate). Every entity's name, contested relation
type, contested values, and background facts are left untouched; only
contested.n_a/n_b are rescaled -- so the already-generated vignette variants
(preprocess/data_pools/vignettes/) remain directly reusable, no new LLM generation
needed. This works cleanly only because the 6 split levels' proportions
(experimental_plans.tex Sec. scale) happen to scale to exact integers at every
multiplier used so far (T=80 -> 160/320/640/1280 is x2/x4/x8/x16); a non-power-of-2
target T would need rounding and a resulting-sum check this script does not attempt.

    python -m preprocess.rescale_population --multiplier 2 --out results/population_T160.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_POPULATION_PATH = REPO_ROOT / "results" / "population.json"


def rescale(population: list[dict], multiplier: int) -> list[dict]:
    rescaled = []
    for entity in population:
        entity = json.loads(json.dumps(entity))  # deep copy, keep the original untouched
        c = entity["contested"]
        c["n_a"] *= multiplier
        c["n_b"] *= multiplier
        rescaled.append(entity)
    return rescaled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-path", type=Path, default=DEFAULT_POPULATION_PATH)
    parser.add_argument("--multiplier", type=int, required=True, help="Integer multiplier applied to every entity's n_a/n_b.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    population = json.load(open(args.population_path))
    original_t = population[0]["contested"]["n_a"] + population[0]["contested"]["n_b"]
    rescaled = rescale(population, args.multiplier)

    new_t = rescaled[0]["contested"]["n_a"] + rescaled[0]["contested"]["n_b"]
    for entity in rescaled:
        c = entity["contested"]
        assert c["n_a"] + c["n_b"] == new_t, f"{entity['entity_id']}: split sum {c['n_a']+c['n_b']} != {new_t}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rescaled, f, indent=2)
    print(f"Rescaled {len(rescaled)} entities: T={original_t} -> T={new_t} (x{args.multiplier}) -> {args.out}")


if __name__ == "__main__":
    main()
