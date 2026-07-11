"""Post-training report: the 4 recall/contested-pair tables, as markdown.

Reads a completed run's results/<run-name>/recall_eval_step<N>.json (written by
eval/recall.py) and writes results/<run-name>/recall_tables.md with 4 tables, each
broken out by relation ("domain") plus a final row pooling all relations together:

  Table 1: background (single-fact, uncontested) top1/top5 recall accuracy.
  Table 2: contested-pair ABSOLUTE top1 rate -- for each (n_a, n_b) frequency split,
      the % of records where val_a (val_b) is the model's actual #1 prediction across
      the whole vocabulary (rank==1). Does not sum to 100 -- the model's true top
      pick can be neither candidate.
  Table 3: same as Table 2 but top5 (rank<=5).
  Table 4: contested-pair RELATIVE rate -- for each (n_a, n_b) split, the % of records
      where the model assigns higher log-probability to val_a vs val_b specifically
      (ignoring the rest of the vocabulary). Always sums to 100 by construction: this
      is a forced two-way comparison, not a measure of absolute confidence.

See the conversation that motivated this split: Table 4 alone looked like real
recall even at T80, where Table 2/3 show both candidates are actually far from the
model's real top prediction -- the "signal" in Table 4 at low T is mostly noise
between two low-probability options, not memorization.

    python -m eval.build_recall_tables --run-name gpt2-small-openwebtext-T320
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _find_recall_json(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("recall_eval_step*.json"))
    if not candidates:
        raise FileNotFoundError(f"No recall_eval_step*.json found in {run_dir}")
    # step number is the largest suffix -- the final/best checkpoint's eval.
    return max(candidates, key=lambda p: int(p.stem.rsplit("step", 1)[1]))


def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}" if d else "NA"


def build_table1(records: list[dict], relations: list[str]) -> str:
    """Background (uncontested) top1/top5 accuracy, by relation + pooled."""
    counts = collections.defaultdict(lambda: [0, 0, 0])  # rel -> [n_top1, n_top5, n_total]
    for r in records:
        if r["is_contested"]:
            continue
        c = counts[r["relation_key"]]
        c[0] += int(r["top1"])
        c[1] += int(r["top5"])
        c[2] += 1

    lines = ["| relation | top1 % | top5 % | n |", "|---|---|---|---|"]
    for rel in relations:
        t1, t5, n = counts[rel]
        lines.append(f"| {rel} | {_pct(t1, n)} | {_pct(t5, n)} | {n} |")
    t1 = sum(counts[r][0] for r in relations)
    t5 = sum(counts[r][1] for r in relations)
    n = sum(counts[r][2] for r in relations)
    lines.append(f"| **POOLED** | **{_pct(t1, n)}** | **{_pct(t5, n)}** | {n} |")
    return "\n".join(lines)


def _split_pairs(contested: list[dict]) -> list[tuple[int, int]]:
    return sorted(set((r["n_a"], r["n_b"]) for r in contested), key=lambda p: p[0] - p[1])


def build_absolute_table(contested: list[dict], relations: list[str], pairs: list[tuple[int, int]], k: int) -> str:
    """Table 2 (k=1) / Table 3 (k=5): absolute (whole-vocab) top-k rate for A and B
    separately -- need not sum to 100%, see module docstring."""
    counts = collections.defaultdict(lambda: [0, 0, 0])  # (rel,pair) -> [n_a_topk, n_b_topk, n_total]
    for r in contested:
        key = (r["relation_key"], (r["n_a"], r["n_b"]))
        c = counts[key]
        c[0] += int(r["rank_a"] <= k)
        c[1] += int(r["rank_b"] <= k)
        c[2] += 1

    header = "| relation | " + " | ".join(str(p) for p in pairs) + " |"
    sep = "|---|" + "---|" * len(pairs)
    lines = [header, sep]
    for rel in relations:
        cells = []
        for p in pairs:
            a, b, n = counts[(rel, p)]
            cells.append(f"(A={_pct(a, n)}%,B={_pct(b, n)}%)")
        lines.append(f"| {rel} | " + " | ".join(cells) + " |")
    pooled_cells = []
    for p in pairs:
        a = sum(counts[(r, p)][0] for r in relations)
        b = sum(counts[(r, p)][1] for r in relations)
        n = sum(counts[(r, p)][2] for r in relations)
        pooled_cells.append(f"(A={_pct(a, n)}%,B={_pct(b, n)}%)")
    lines.append("| **POOLED** | " + " | ".join(pooled_cells) + " |")
    return "\n".join(lines)


def build_relative_table(contested: list[dict], relations: list[str], pairs: list[tuple[int, int]]) -> str:
    """Table 4: forced A-vs-B comparison (favored_by_model), sums to 100% per cell."""
    counts = collections.defaultdict(lambda: [0, 0])  # (rel,pair) -> [n_favor_a, n_favor_b]
    for r in contested:
        key = (r["relation_key"], (r["n_a"], r["n_b"]))
        idx = 0 if r["favored_by_model"] == "A" else 1
        counts[key][idx] += 1

    header = "| relation | " + " | ".join(str(p) for p in pairs) + " |"
    sep = "|---|" + "---|" * len(pairs)
    lines = [header, sep]
    for rel in relations:
        cells = []
        for p in pairs:
            a, b = counts[(rel, p)]
            cells.append(f"({_pct(a, a + b)}%,{_pct(b, a + b)}%)")
        lines.append(f"| {rel} | " + " | ".join(cells) + " |")
    pooled_cells = []
    for p in pairs:
        a = sum(counts[(r, p)][0] for r in relations)
        b = sum(counts[(r, p)][1] for r in relations)
        pooled_cells.append(f"({_pct(a, a + b)}%,{_pct(b, a + b)}%)")
    lines.append("| **POOLED** | " + " | ".join(pooled_cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--out", default=None, help="defaults to <results-dir>/<run-name>/recall_tables.md")
    args = parser.parse_args()

    run_dir = Path(args.results_dir) / args.run_name
    recall_path = _find_recall_json(run_dir)
    out_path = Path(args.out) if args.out else run_dir / "recall_tables.md"

    with open(recall_path) as f:
        data = json.load(f)
    records = data["records"]
    contested = [r for r in records if r["is_contested"]]
    relations = sorted(set(r["relation_key"] for r in records))
    pairs = _split_pairs(contested)
    checkpoint_step = recall_path.stem.rsplit("step", 1)[1]

    sections = [
        f"# {args.run_name}: recall tables (checkpoint step {checkpoint_step})",
        "",
        "Table cell sizes: each contested (relation, split) cell is n=100 records "
        "(20 entities x 5 templates), n=1400 pooled; each background/uncontested "
        "relation row is n=2400, n=33600 pooled.",
        "",
        "## Table 1: background (uncontested) top1/top5 accuracy",
        "",
        build_table1(records, relations),
        "",
        "## Table 2: contested-pair absolute top1 rate (whole-vocab rank==1; need not sum to 100%)",
        "",
        build_absolute_table(contested, relations, pairs, k=1),
        "",
        "## Table 3: contested-pair absolute top5 rate (whole-vocab rank<=5; need not sum to 100%)",
        "",
        build_absolute_table(contested, relations, pairs, k=5),
        "",
        "## Table 4: contested-pair relative rate (forced A-vs-B logit comparison; sums to 100% per cell)",
        "",
        build_relative_table(contested, relations, pairs),
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
