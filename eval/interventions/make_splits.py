"""Generates the ONE fixed entity-level train/val/test split used by all three
intervention jobs (temperature/dola/caa) -- run this once, before any of them.
Stratified by (n_a, n_b) exposure-split level so every level is represented
proportionally in train, val, and test. One unit = one entity (each entity has
exactly one contested relation by construction).

    python -m eval.interventions.make_splits --T 1280
"""
from __future__ import annotations

import argparse

from eval.interventions.common import (
    load_population,
    make_splits,
    save_splits,
    top7_population,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--T", type=int, default=1280)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    population = top7_population(load_population(args.T))
    assignment = make_splits(
        population, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )
    path = save_splits(
        assignment,
        population,
        T=args.T,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    print(f"{len(population)} entities (top-7 relation types, T={args.T})")
    print(f"train={counts['train']}  val={counts['val']}  test={counts['test']}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
