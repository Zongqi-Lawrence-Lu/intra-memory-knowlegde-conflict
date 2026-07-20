"""Dial job (results/mech_interp/<run-name>/dial/): continuous-magnitude steering
dial at the causal-tracing-identified locus, swept and compared against the true
exposure-ratio target (experimental_plans.tex Sec.mechinterp-dial).

--layer is the causal locus to steer at, as a transformer BLOCK index (0..n_layer-1)
-- the SAME convention causal_tracing.py itself uses (it's the block whose forward
output score_steered/steering_utils.steering will add the vector to). If omitted,
it's read straight from an already-run causal_map.json as the residual-stream
layer with the largest mean |restoration effect|, no conversion needed at this
handoff (see infer_layer_from_causal_tracing's docstring for the real off-by-one
this pipeline has, and where it actually lives) -- run causal tracing with
"residual" in --granularities first, or pass --layer explicitly to skip that
dependency.

    python -m mech_interp.run_dial --T 1280 --layer 6
"""
from __future__ import annotations

import argparse
import json

import torch

from eval.interventions.common import entities_in_split, load_splits
from inference_time.utils.model_utils import load_trained_model
from mech_interp.common import load_population, run_name_for, stage_dir, top7_population

DEFAULT_MAGNITUDES = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def infer_layer_from_causal_tracing(T: int) -> int:
    """Reads an already-run causal_map.json and returns the residual-stream layer
    (a transformer BLOCK index, causal_tracing.py's own convention) with the
    largest mean |necessity effect|, unconverted -- this is exactly the block
    score_steered/steering_utils.steering will add the vector to, so no +1 (or
    any other conversion) belongs at this handoff.

    The real off-by-one in this pipeline is one hop later, inside
    steering_dial.build_dial_vector: it reads the vector from
    mech_interp.common.capture_all_layers_last_token's hidden_states tuple, where
    index `layer` is block `layer`'s INPUT (one block EARLIER than block
    `layer`'s OUTPUT, which is where the vector actually gets injected) -- fixed
    there with a `+1` on the READ side. An earlier version of this function
    applied that +1 HERE instead (converting the block index before returning
    it), which crashed a full-scale run the first time causal tracing's own top
    layer was the last block (`model.transformer.h[12]` doesn't exist, only
    0..11 do) -- found + fixed 2026-07-17. See mech_interp/sweep.py's
    _pick_causal_locus for the sweep orchestrator's identical (non-)conversion."""
    causal_tracing_path = stage_dir("causal_tracing", T) / "causal_map.json"
    if not causal_tracing_path.exists():
        raise FileNotFoundError(
            f"--layer not given and no causal tracing output at {causal_tracing_path} to infer it from -- "
            f"run `python -m mech_interp.run_causal_tracing --T {T}` (with 'residual' in "
            f"--granularities) first, or pass --layer explicitly."
        )
    with open(causal_tracing_path) as f:
        causal_tracing_result = json.load(f)
    if "residual" not in causal_tracing_result["classification_by_granularity"]:
        raise ValueError(f"{causal_tracing_path} has no 'residual' granularity -- rerun causal tracing with it included.")
    rows = causal_tracing_result["classification_by_granularity"]["residual"]
    best = max(rows, key=lambda r: abs(r["necessity_effect_a_clean"]) + abs(r["necessity_effect_b_clean"]))
    return best["layer"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--T", type=int, default=1280)
    parser.add_argument("--layer", type=int, default=None, help="causal locus; inferred from causal tracing if omitted")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=DEFAULT_MAGNITUDES)
    parser.add_argument(
        "--vector-source-split", default="train", choices=["train", "val", "test", "all"],
        help="entities the diff-of-means vector is built from -- kept separate from --split "
             "(scored on) so the dial isn't fit and evaluated on the same entities",
    )
    parser.add_argument("--limit-entities", type=int, default=None)
    args = parser.parse_args()

    layer = args.layer if args.layer is not None else infer_layer_from_causal_tracing(args.T)
    print(f"Dial: steering at layer={layer}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_trained_model(args.T, device=device)
    tokenizer.padding_side = "left"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.run.dtype]

    population = top7_population(load_population(args.T))
    assignment = load_splits(args.T)

    def pick(split):
        return population if split == "all" else entities_in_split(population, assignment, split)

    vector_entities = pick(args.vector_source_split)
    score_entities = pick(args.split)
    if args.limit_entities is not None:
        vector_entities = vector_entities[: args.limit_entities]
        score_entities = score_entities[: args.limit_entities]
    print(f"Vector from {len(vector_entities)} '{args.vector_source_split}' entities; "
          f"scoring {len(score_entities)} '{args.split}' entities")

    from mech_interp.steering_dial import build_dial_vector, dial_smoothness, sweep_magnitudes

    vector = build_dial_vector(model, tokenizer, vector_entities, layer, device, dtype, args.batch_size)
    print(f"||vector|| = {vector.norm().item():.4f}")

    rows = sweep_magnitudes(model, tokenizer, score_entities, layer, vector, args.magnitudes, device, dtype, args.batch_size)
    smoothness = dial_smoothness(rows)
    print(f"smoothness: {smoothness}")
    for r in rows:
        print(f"  magnitude={r['magnitude']:+.2f}  mean_p_a={r['mean_p_a']:.4f}  n_scored={r['n_entities_scored']}")

    out_dir = stage_dir("dial", args.T)
    out_path = out_dir / f"dial_layer{layer}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "T": args.T,
                "run_name": run_name_for(args.T),
                "layer": layer,
                "vector_source_split": args.vector_source_split,
                "score_split": args.split,
                "vector_norm": vector.norm().item(),
                "magnitudes": args.magnitudes,
                "smoothness": smoothness,
                "sweep": rows,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")

    make_plot(rows, layer, out_dir / f"dial_layer{layer}_{args.split}.png")
    print(f"Wrote {out_dir / f'dial_layer{layer}_{args.split}.png'}")


def make_plot(rows: list[dict], layer: int, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    magnitudes = [r["magnitude"] for r in rows]
    ax.plot(magnitudes, [r["mean_p_a"] for r in rows], marker="o", color="#2a78d6", linewidth=2, label="pooled mean p_A")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("steering magnitude")
    ax.set_ylabel("p_A (sigmoid of steered logit gap)")
    ax.set_title(f"Steering dial (layer={layer})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
