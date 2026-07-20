"""Probing job (results/mech_interp/<run-name>/probing/): per-layer, per-position
probe_A/probe_B angle sweep, plus conflict-ratio and (if occurrence_log.json is
available) recency-direction comparisons (experimental_plans.tex
Sec.mechinterp-reploc). Correlational -- no patching, cheap relative to causal
tracing, but needs GPU for the activation captures. Runs standalone -- no other
mech_interp stage's output is required.

    python -m mech_interp.run_probing --T 1280

Trains probes on the TRAIN split only entities is deliberately NOT enforced here:
these probes are diagnostic/correlational (not a metric being fit-and-reported
against a held-out set the way the calibration grid in eval/interventions/ is), so
using the full top-7 population gives more stable directions. If probe
generalization itself becomes a question worth answering, add a --split flag rather
than assuming TRAIN-only silently.
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

    entities = top7_population(load_population(args.T))
    if args.limit_entities is not None:
        entities = entities[: args.limit_entities]
    print(f"Probing: {len(entities)} entities")

    from mech_interp.probing import run_probing_sweep

    sweep = run_probing_sweep(model, tokenizer, entities, args.T, device, dtype, args.batch_size)

    for position, rows in sweep["by_position"].items():
        best = min(rows, key=lambda r: r["angle_a_b_degrees"])
        most_orth = max(rows, key=lambda r: r["angle_a_b_degrees"])
        print(
            f"position={position}: angle range "
            f"[{best['angle_a_b_degrees']:.1f} deg @ layer {best['layer']}, "
            f"{most_orth['angle_a_b_degrees']:.1f} deg @ layer {most_orth['layer']}]"
        )

    out_dir = stage_dir("probing", args.T)
    out_path = out_dir / "probe_sweep.json"
    with open(out_path, "w") as f:
        json.dump({"run_name": run_name_for(args.T), **sweep}, f, indent=2)
    print(f"Wrote {out_path}")

    make_plot(sweep, out_dir / "angle_sweep.png")
    print(f"Wrote {out_dir / 'angle_sweep.png'}")


def make_plot(sweep: dict, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sweep["by_position"]), figsize=(6 * len(sweep["by_position"]), 4.5), sharey=True)
    if len(sweep["by_position"]) == 1:
        axes = [axes]
    for ax, (position, rows) in zip(axes, sweep["by_position"].items()):
        layers = [r["layer"] for r in rows]
        ax.plot(layers, [r["angle_a_b_degrees"] for r in rows], marker="o", label="angle(probe_A, probe_B)")
        ax.plot(layers, [r["angle_a_ratio_degrees"] for r in rows], marker="o", label="angle(probe_A, conflict-ratio)")
        ax.plot(layers, [r["angle_b_ratio_degrees"] for r in rows], marker="o", label="angle(probe_B, conflict-ratio)")
        if rows and "angle_a_recency_degrees" in rows[0]:
            ax.plot(layers, [r["angle_a_recency_degrees"] for r in rows], marker="o", label="angle(probe_A, recency)")
            ax.plot(layers, [r["angle_b_recency_degrees"] for r in rows], marker="o", label="angle(probe_B, recency)")
        ax.axhline(90, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"position={position}")
        ax.set_xlabel("layer (0=embeddings)")
        ax.set_ylabel("angle (degrees)")
        ax.legend(fontsize=7)
    fig.suptitle("Representational probing: probe-direction angles by layer/position")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
