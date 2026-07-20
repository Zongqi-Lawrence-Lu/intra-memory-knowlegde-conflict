"""Causal-tracing job (results/mech_interp/<run-name>/causal_tracing/): causal map
(residual stream / attention heads / MLP blocks) over a sample of cued
clean(A)/corrupt(B) prompt pairs, run in BOTH clean-orientations (A-clean and
B-clean) via mech_interp.causal_tracing.run_causal_map_both_directions so
classify_components can compare necessity effects across both directions
(experimental_plans.tex Sec.mechinterp-causal). Runs standalone -- no other
mech_interp stage's output is required (suppression and the dial can each
optionally consume THIS stage's output, see their own module docstrings).

Expensive relative to scaffolding/probing -- see causal_tracing.causal_map_for_example's
docstring for the per-example forward-pass count. --limit-entities and
--granularities exist specifically to bound a first exploratory pass; the full
sweep across the whole top-7 population should be confirmed with estimated
duration/resources before submitting, per CLAUDE.md Sec.5.

    python -m mech_interp.run_causal_tracing --T 1280 --limit-entities 10
"""
from __future__ import annotations

import argparse
import json

import torch

from inference_time.utils.model_utils import load_trained_model
from mech_interp.common import load_population, run_name_for, stage_dir, top7_population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--T", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--limit-entities", type=int, default=20,
        help="number of entities to sample (default small -- this phase is expensive, see module docstring)",
    )
    parser.add_argument(
        "--query-templates-per-entity", type=int, default=1,
        help="how many of the 4 non-cue query templates to use per entity (default 1, i.e. one causal-map example/entity)",
    )
    parser.add_argument("--mlp-block-size", type=int, default=256)
    parser.add_argument(
        "--granularities", nargs="+", default=["residual", "head", "mlp"], choices=["residual", "head", "mlp"],
    )
    parser.add_argument("--necessity-threshold", type=float, default=0.5, help="nats, classify_components")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_trained_model(args.T, device=device)

    entities = top7_population(load_population(args.T))[: args.limit_entities]
    print(f"Causal tracing: {len(entities)} entities, granularities={args.granularities}")

    from mech_interp.causal_tracing import run_causal_map_both_directions

    result = run_causal_map_both_directions(
        model, tokenizer, entities, device,
        query_templates_per_entity=args.query_templates_per_entity,
        mlp_block_size=args.mlp_block_size,
        granularities=args.granularities,
        necessity_threshold=args.necessity_threshold,
    )

    for granularity, classification in result["classification_by_granularity"].items():
        n_shared = sum(1 for r in classification if r["classification"] == "shared")
        n_disjoint = sum(1 for r in classification if r["classification"] == "disjoint")
        print(
            f"granularity={granularity}: shared={n_shared} disjoint={n_disjoint} "
            f"inactive={len(classification) - n_shared - n_disjoint}"
        )

    out_dir = stage_dir("causal_tracing", args.T)
    out_path = out_dir / "causal_map.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "T": args.T,
                "run_name": run_name_for(args.T),
                "n_entities": len(entities),
                "granularities": args.granularities,
                "mlp_block_size": args.mlp_block_size,
                "necessity_threshold": args.necessity_threshold,
                **result,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
