"""Temperature-scaling job (results/interventions/<run-name>/temperature/): no
fitting, no GPU. Sweeps the final-readout temperature tau linearly over
0.1 -> 2.0 (step 0.1) and recomputes the calibration-target cross-entropy/KL and
the monotonicity-violation count at each tau, purely post-hoc from the
already-computed T=1280 recall JSON (eval/recall.py). Scored on the TEST split
only, so this curve sits on the exact same entities DoLa and CAA report their
final numbers on.

Deliberately not a fit-and-report-a-winner job (see conversation, 2026-07-12):
there is no free parameter being *selected* here, only swept and plotted, so
there's no leakage question -- the point is the shape of the curve, especially
that monotonicity_violations does not move across the whole sweep while
cross-entropy/KL do, which is precisely the property that makes a single global
temperature an unfit substitute for a real intervention.

    python -m eval.interventions.temperature_sweep --T 1280
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.build_recall_tables import _find_recall_json
from eval.interventions.common import (
    REPO_ROOT,
    average_over_templates,
    interventions_dir,
    load_splits,
    run_name_for,
    summarize_at_temperature,
)
from inference_time.utils.model_utils import RELIABLE_RELATION_TYPES

# dataviz palette (eval/plot_training_curves.py's convention -- static print figure,
# light-mode only, one measure per panel, no dual axes).
COLOR_XENT = "#2a78d6"
COLOR_KL = "#1baf7a"
COLOR_MONO = "#c0392b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e5e0"

TEMPERATURES = [round(0.1 * i, 1) for i in range(1, 21)]  # 0.1, 0.2, ..., 2.0


def _style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, pad=10)
    ax.set_xlabel("temperature (τ)", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def make_plot(sweep: list[dict], out_path) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    taus = [row["temperature"] for row in sweep]
    xent = [row["overall_mean_cross_entropy_to_proportional_target"] for row in sweep]
    kl = [row["overall_mean_kl_to_proportional_target"] for row in sweep]
    mono = [row["monotonicity_violations"] for row in sweep]

    ax_top.plot(taus, xent, marker="o", color=COLOR_XENT, linewidth=2, label="cross-entropy")
    ax_top.plot(taus, kl, marker="o", color=COLOR_KL, linewidth=2, label="KL")
    ax_top.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _style_axis(ax_top, "Calibration divergence vs. temperature (TEST, pooled)", "nats")

    ax_bottom.plot(taus, mono, marker="o", color=COLOR_MONO, linewidth=2)
    ax_bottom.set_ylim(-0.5, max(5, max(mono) + 1) if mono else 5)
    _style_axis(ax_bottom, "Monotonicity violations vs. temperature (invariant by construction)", "count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--T", type=int, default=1280)
    parser.add_argument(
        "--results-dir", default=str(REPO_ROOT / "results"),
        help="where to find the run's own recall_eval_step*.json (training pipeline output)",
    )
    args = parser.parse_args()

    run_name = run_name_for(args.T)
    run_dir = Path(args.results_dir) / run_name
    recall_path = _find_recall_json(run_dir)
    checkpoint_step = recall_path.stem.rsplit("step", 1)[1]

    with open(recall_path) as f:
        data = json.load(f)
    all_records = data["records"]

    kept_relations = set(RELIABLE_RELATION_TYPES)
    assignment = load_splits(args.T)
    test_ids = {eid for eid, split in assignment.items() if split == "test"}

    records = [
        r
        for r in all_records
        if r.get("is_contested") and r["relation_key"] in kept_relations and r["entity_id"] in test_ids
    ]
    if not records:
        raise ValueError("No contested top-7 TEST records found -- check splits.json and the recall JSON match.")

    entity_rows, n_skipped = average_over_templates(records)
    print(f"TEST: {len(entity_rows)} entities scored, {n_skipped} skipped (no divergence-ok template)")

    sweep = [summarize_at_temperature(entity_rows, tau) for tau in TEMPERATURES]

    out_dir = interventions_dir(args.T) / "temperature"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"sweep_step{checkpoint_step}.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "source_recall_json": str(recall_path.relative_to(REPO_ROOT)),
                "checkpoint": data["checkpoint"],
                "split": "test",
                "n_entities": len(entity_rows),
                "n_entities_skipped": n_skipped,
                "temperatures": TEMPERATURES,
                "sweep": sweep,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")

    out_png = out_dir / f"sweep_step{checkpoint_step}.png"
    make_plot(sweep, out_png)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
