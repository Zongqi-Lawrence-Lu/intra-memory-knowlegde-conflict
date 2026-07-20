"""Scaffolding job (results/mech_interp/<run-name>/scaffolding/): generation vs.
verification probes for every top-7-relation contested entity, and the resulting
multi-verse / clean-suppression / mismatch / verifies-neither classification
(experimental_plans.tex Sec.mechinterp-scaffolding). Cheap, no activation access --
meant to run before probing/causal-tracing/dial and set their baseline expectation.
Runs standalone -- no other mech_interp stage's output is required.

    python -m mech_interp.run_scaffolding --T 1280
"""
from __future__ import annotations

import argparse
import json

import torch

from eval.interventions.common import entities_in_split, load_splits
from inference_time.utils.model_utils import load_trained_model
from mech_interp.common import load_population, run_name_for, stage_dir, top7_population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--T", type=int, default=1280)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit-entities", type=int, default=None,
        help="debug/smoke-test: only use the first N entities",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_trained_model(args.T, device=device)
    tokenizer.padding_side = "left"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.run.dtype]

    population = top7_population(load_population(args.T))
    if args.split == "all":
        entities = population
    else:
        assignment = load_splits(args.T)
        entities = entities_in_split(population, assignment, args.split)
    if args.limit_entities is not None:
        entities = entities[: args.limit_entities]
    print(f"Scaffolding: {len(entities)} entities ({args.split})")

    from mech_interp.scaffolding import (
        build_scaffolding_summary,
        run_generation_probe,
        run_verification_probe,
    )

    print("Running generation probe...")
    gen_records = run_generation_probe(model, tokenizer, entities, device, dtype, args.batch_size)
    print("Running verification probe...")
    verif_records = run_verification_probe(model, tokenizer, entities, device, dtype, args.batch_size)

    summary = build_scaffolding_summary(gen_records, verif_records)
    print(f"n_entities_scored={summary['n_entities_scored']}")
    print(f"pattern_counts={summary['pattern_counts']}")

    # per-relation-type breakdown, since Sec.relation-restriction already showed
    # recall is not uniform even within the top-7 -- worth checking the pattern
    # classification isn't either.
    by_relation: dict[str, dict[str, int]] = {}
    entity_relation = {e["entity_id"]: e["contested"]["relation_key"] for e in entities}
    for row in summary["rows"]:
        rk = entity_relation.get(row["entity_id"], "unknown")
        by_relation.setdefault(rk, {})
        by_relation[rk][row["pattern"]] = by_relation[rk].get(row["pattern"], 0) + 1

    out_dir = stage_dir("scaffolding", args.T)
    out_path = out_dir / f"scaffolding_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "T": args.T,
                "run_name": run_name_for(args.T),
                "split": args.split,
                "n_entities": len(entities),
                "verify_suffix": "This statement is",
                "summary": {k: v for k, v in summary.items() if k != "rows"},
                "pattern_counts_by_relation": by_relation,
                "rows": summary["rows"],
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
