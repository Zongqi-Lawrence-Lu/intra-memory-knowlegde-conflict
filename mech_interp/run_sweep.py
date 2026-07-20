"""Sweep job (results/mech_interp/<run-name>/sweep/): repeats probing, causal
tracing, and the dial across the 6 (n_a, n_b) split levels of experimental_plans.tex
Sec.scale, testing whether the mechanism type (separate/superposed,
sharp/continuous) depends on the frequency split (experimental_plans.tex
Sec.mechinterp-sweep) -- likely the project's strongest novel claim if it holds.
Runs standalone -- no other mech_interp stage's output is required (it re-invokes
each stage's own function in-process, not via their output files).

COST WARNING: this calls causal tracing (the expensive stage,
~n_layer*(1+n_head+n_mlp_blocks)*2 forward passes PER example) once per split level.
Defaults are kept small (--limit-entities-per-level 8, --causal-granularities residual
head only, no mlp) specifically to bound a first exploratory pass -- widen only after
confirming duration/resources with the estimated cost of a small run, per CLAUDE.md
Sec.5. Suppression is intentionally NOT included in this orchestrator (see
mech_interp/sweep.py's run_sweep_for_level docstring) -- run run_suppression.py
separately per level of interest once a level's per-level causal map looks worth
following up.

    python -m mech_interp.run_sweep --T 1280 --limit-entities-per-level 8
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-entities-per-level", type=int, default=8)
    parser.add_argument("--causal-granularities", nargs="+", default=["residual", "head"], choices=["residual", "head", "mlp"])
    parser.add_argument("--causal-mlp-block-size", type=int, default=256)
    parser.add_argument("--causal-necessity-threshold", type=float, default=0.5)
    parser.add_argument("--dial-magnitudes", type=float, nargs="+", default=[-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_trained_model(args.T, device=device)
    tokenizer.padding_side = "left"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.run.dtype]

    entities = top7_population(load_population(args.T))
    print(f"Sweep: {len(entities)} entities across the top-7 population")

    from mech_interp.sweep import run_sweep

    result = run_sweep(
        model, tokenizer, entities, args.T, device, dtype,
        limit_entities_per_level=args.limit_entities_per_level, batch_size=args.batch_size,
        causal_granularities=args.causal_granularities, causal_mlp_block_size=args.causal_mlp_block_size,
        causal_necessity_threshold=args.causal_necessity_threshold, dial_magnitudes=args.dial_magnitudes,
    )

    print("\n=== summary across split levels ===")
    for key, level in result["per_level"].items():
        print(
            f"{key}: min_angle(A,B)@last_token={level['probing_min_angle_last_token']['angle_a_b_degrees']:.1f} deg "
            f"(layer {level['probing_min_angle_last_token']['layer']}); "
            f"causal_locus={level['causal_locus']}; "
            f"dial_smoothness={level['dial_smoothness']}"
        )

    out_dir = stage_dir("sweep", args.T)
    out_path = out_dir / "sweep.json"
    with open(out_path, "w") as f:
        json.dump({"run_name": run_name_for(args.T), **result}, f, indent=2)
    print(f"Wrote {out_path}")

    make_plot(result, out_dir / "sweep_summary.png")
    print(f"Wrote {out_dir / 'sweep_summary.png'}")


def make_plot(result: dict, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = result["levels"]
    keys = [f"{n_a}_{n_b}" for (n_a, n_b) in levels]
    freq_gaps = [n_a - n_b for (n_a, n_b) in levels]
    min_angles = [result["per_level"][k]["probing_min_angle_last_token"]["angle_a_b_degrees"] for k in keys]
    smoothness = [result["per_level"][k]["dial_smoothness"]["max_step_fraction_of_range"] or 0.0 for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(freq_gaps, min_angles, marker="o", color="#2a78d6")
    ax1.axhline(90, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("frequency gap (n_a - n_b)")
    ax1.set_ylabel("min angle(probe_A, probe_B) across layers (deg)")
    ax1.set_title("Representational separability by split level")

    ax2.plot(freq_gaps, smoothness, marker="o", color="#c0392b")
    ax2.set_xlabel("frequency gap (n_a - n_b)")
    ax2.set_ylabel("dial max-step / range")
    ax2.set_title("Dial discreteness by split level")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
