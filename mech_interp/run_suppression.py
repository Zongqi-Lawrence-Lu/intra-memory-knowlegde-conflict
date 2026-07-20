"""Suppression job (results/mech_interp/<run-name>/suppression/): logit-lens
suppression-vs-erasure check, using the top-K necessary components from an
already-run causal_map.json (experimental_plans.tex Sec.mechinterp-suppression).
Requires causal tracing (run_causal_tracing.py) to have been run first for the
same --T -- a real data dependency (this stage consumes causal tracing's
necessity-effect output), not a naming one.

    python -m mech_interp.run_suppression --T 1280 --granularity head --top-k 5
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
    parser.add_argument("--granularity", default="head", choices=["head", "mlp"])
    parser.add_argument("--direction", default="a_clean", choices=["a_clean", "b_clean"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--limit-entities", type=int, default=20,
        help="number of entities to run the suppression check on",
    )
    args = parser.parse_args()

    causal_tracing_path = stage_dir("causal_tracing", args.T) / "causal_map.json"
    if not causal_tracing_path.exists():
        raise FileNotFoundError(
            f"{causal_tracing_path} not found -- run `python -m mech_interp.run_causal_tracing --T {args.T}` first."
        )
    with open(causal_tracing_path) as f:
        causal_tracing_result = json.load(f)
    if args.granularity not in causal_tracing_result["classification_by_granularity"]:
        raise ValueError(
            f"Causal tracing output at {causal_tracing_path} has no '{args.granularity}' granularity "
            f"(has: {list(causal_tracing_result['classification_by_granularity'])}) -- rerun causal tracing with "
            f"--granularities including {args.granularity}."
        )

    from mech_interp.suppression import run_suppression_check, top_necessary_components

    classification_rows = causal_tracing_result["classification_by_granularity"][args.granularity]
    components = top_necessary_components(classification_rows, args.granularity, args.top_k, direction=args.direction)
    print(f"Ablating top-{args.top_k} '{args.granularity}' components (direction={args.direction}): {components}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_trained_model(args.T, device=device)

    entities = top7_population(load_population(args.T))[: args.limit_entities]
    print(f"Suppression: {len(entities)} entities")

    results = []
    for i, entity in enumerate(entities):
        print(f"  [{i+1}/{len(entities)}] entity={entity['entity_id']}")
        results.append(run_suppression_check(model, tokenizer, entity, device, components))

    n_suppression_not_erasure = sum(
        1
        for r in results
        if r["baseline_summary"]["mid_layer_gap"] < 0.5 * r["baseline_summary"]["final_gap"]
        and r["baseline_summary"]["final_gap"] > 0
    )
    print(
        f"{n_suppression_not_erasure}/{len(results)} entities show mid-layer gap < half the final-layer "
        f"gap in the UNABLATED trace (candidate 'suppression not erasure' cases -- inspect baseline_trace "
        f"directly before treating this count as conclusive)"
    )

    out_dir = stage_dir("suppression", args.T)
    out_path = out_dir / f"suppression_{args.granularity}_{args.direction}_top{args.top_k}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "T": args.T,
                "run_name": run_name_for(args.T),
                "causal_tracing_source": str(causal_tracing_path),
                "granularity": args.granularity,
                "direction": args.direction,
                "top_k": args.top_k,
                "ablated_components": components,
                "n_entities": len(entities),
                "n_candidate_suppression_not_erasure": n_suppression_not_erasure,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
