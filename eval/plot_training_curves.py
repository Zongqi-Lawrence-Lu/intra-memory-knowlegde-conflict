"""Post-training figure: loss and held-out perplexity trends, side by side.

Reads a completed run's results/<run-name>/train_metrics.jsonl (written throughout
training by training/train.py -- interleaved {"step","loss",...} and
{"step","val_ppl",...} records, the latter every eval.eval_interval_steps) plus
config_used.yaml (to label the right-hand panel correctly -- data.val_path tells us
whether the val_ppl trend that was actually logged during training is the in-domain
OpenWebText signal or (for runs predating the 2026-07-11 val_path fix,
memory/training_val_path_ood_vs_indomain_2026-07-11.md) the out-of-domain WikiText
one) plus held_out_ppl.jsonl (the final-checkpoint numbers for both val sets, appended
post-training by eval/held_out_ppl.py) for a small annotation of the final
out-of-domain perplexity, which isn't otherwise on this plot.

Output: results/<run-name>/training_curves.png, one figure with two side-by-side
single-series line panels (loss; val_ppl) -- never dual-axis (dataviz skill
non-negotiable: one axis per panel, two measures of different scale get two panels,
not two y-scales on one). Log-scale y on both, since early-training loss/ppl values
are an order of magnitude above the converged tail and would otherwise flatten the
interesting part of the curve.

    python -m eval.plot_training_curves --run-name gpt2-small-openwebtext-T320
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent

# dataviz skill reference palette (references/palette.md), light-mode steps -- this is
# a static print/paper figure, not an interactive artifact, so light-mode only.
COLOR_LOSS = "#2a78d6"       # categorical slot 1, blue
COLOR_VALPPL = "#1baf7a"     # categorical slot 2, aqua
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e5e0"             # one-step-off-surface gray, hairline

IN_DOMAIN_VAL_PATH = "data/processed/held-out-openwebtext-val"
OOD_VAL_PATH = "data/processed/held-out-wikitext-val"


def _style_axis(ax, title: str, ylabel: str, color: str) -> None:
    ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, pad=10)
    ax.set_xlabel("step", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_SECONDARY)
    ax.set_yscale("log")
    ax.grid(True, which="both", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--out", default=None, help="defaults to <results-dir>/<run-name>/training_curves.png")
    args = parser.parse_args()

    run_dir = Path(args.results_dir) / args.run_name
    out_path = Path(args.out) if args.out else run_dir / "training_curves.png"

    steps_loss, loss_vals = [], []
    steps_val, val_vals = [], []
    with open(run_dir / "train_metrics.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if "loss" in rec:
                steps_loss.append(rec["step"])
                loss_vals.append(rec["loss"])
            if "val_ppl" in rec:
                steps_val.append(rec["step"])
                val_vals.append(rec["val_ppl"])

    with open(run_dir / "config_used.yaml") as f:
        cfg = yaml.safe_load(f)
    trained_val_path = cfg["data"]["val_path"]
    trend_is_in_domain = trained_val_path == IN_DOMAIN_VAL_PATH
    trend_label = "in-domain OpenWebText val_ppl" if trend_is_in_domain else "WikiText val_ppl (OOD -- see note)"

    final_ood = final_indomain = None
    ppl_path = run_dir / "held_out_ppl.jsonl"
    if ppl_path.exists():
        with open(ppl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["label"] == "wikitext-out-of-domain":
                    final_ood = rec["val_ppl"]
                elif rec["label"] == "openwebtext-in-domain":
                    final_indomain = rec["val_ppl"]

    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    fig.suptitle(f"{args.run_name}: training curves", fontsize=13, color=TEXT_PRIMARY, y=1.02)

    ax_loss.plot(steps_loss, loss_vals, color=COLOR_LOSS, linewidth=2, solid_capstyle="round")
    _style_axis(ax_loss, "training loss", "loss (log scale)", COLOR_LOSS)
    if loss_vals:
        ax_loss.scatter([steps_loss[-1]], [loss_vals[-1]], s=50, color=COLOR_LOSS, zorder=3,
                         edgecolors="white", linewidths=1.5)
        ax_loss.annotate(f"{loss_vals[-1]:.3f}", (steps_loss[-1], loss_vals[-1]),
                          textcoords="offset points", xytext=(-8, 8), ha="right",
                          fontsize=9, color=TEXT_PRIMARY)

    ax_val.plot(steps_val, val_vals, color=COLOR_VALPPL, linewidth=2, solid_capstyle="round")
    _style_axis(ax_val, trend_label, "val_ppl (log scale)", COLOR_VALPPL)
    if val_vals:
        ax_val.scatter([steps_val[-1]], [val_vals[-1]], s=50, color=COLOR_VALPPL, zorder=3,
                        edgecolors="white", linewidths=1.5)
        ax_val.annotate(f"{val_vals[-1]:.2f}", (steps_val[-1], val_vals[-1]),
                         textcoords="offset points", xytext=(-8, 8), ha="right",
                         fontsize=9, color=TEXT_PRIMARY)

    footnote_lines = []
    if final_ood is not None:
        footnote_lines.append(f"final checkpoint, out-of-domain WikiText val_ppl: {final_ood:.3f}")
    if final_indomain is not None:
        footnote_lines.append(f"final checkpoint, in-domain OpenWebText val_ppl: {final_indomain:.3f}")
    if not trend_is_in_domain:
        footnote_lines.append(
            "NOTE: the panel above plots the training-time signal actually logged for this run "
            "(WikiText/OOD, predating the val_path fix) -- not the in-domain trend."
        )
    if footnote_lines:
        fig.text(0.5, -0.06, "\n".join(footnote_lines), ha="center", va="top",
                  fontsize=9, color=TEXT_SECONDARY)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
