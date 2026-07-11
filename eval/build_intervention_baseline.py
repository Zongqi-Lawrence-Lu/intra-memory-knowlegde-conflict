"""Recompute the calibration-target summary (eval/recall.py's `summarize()`)
restricted to the relation types confirmed to store reliably
(inference_time/utils/model_utils.py:RELIABLE_RELATION_TYPES, experimental_plans.tex
Sec.relation-restriction), instead of pooled across all 14 relation types.

Why this exists: every intervention method (§Baseline Methods) arbitrates between two
already-memorized candidate values for a contested relation. The pooled-14 calibration
curve (results/<run-name>/recall_tables.md Table 4) mixes in 5-7 relation types that
never stored a fact at all, so any pre/post comparison against it is partly comparing
against noise. This script re-aggregates the SAME already-computed per-record data
(no model/GPU needed) restricted to the 7 relations with confirmed reliable recall, so
interventions have a clean "before" picture to improve on.

Output goes to results/interventions/<run-name>/ -- deliberately NOT inside
results/<run-name>/ (the training run's own results folder), since this is downstream
intervention-baseline analysis, not a training-pipeline artifact.

    python -m eval.build_intervention_baseline --run-name gpt2-small-openwebtext-T1280
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.build_recall_tables import _find_recall_json, _split_pairs, build_relative_table, build_table1
from eval.recall import load_eval_templates, summarize
from inference_time.utils.model_utils import RELIABLE_RELATION_TYPES

REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="gpt2-small-openwebtext-T1280")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument(
        "--out-dir", default=None,
        help="defaults to <repo>/results/interventions/<run-name>/",
    )
    args = parser.parse_args()

    run_dir = Path(args.results_dir) / args.run_name
    recall_path = _find_recall_json(run_dir)
    checkpoint_step = recall_path.stem.rsplit("step", 1)[1]

    with open(recall_path) as f:
        data = json.load(f)
    all_records = data["records"]

    kept = set(RELIABLE_RELATION_TYPES)
    seen_relations = set(r["relation_key"] for r in all_records)
    unknown = kept - seen_relations
    if unknown:
        raise ValueError(f"RELIABLE_RELATION_TYPES has relation(s) not present in {recall_path}: {sorted(unknown)}")
    dropped_relations = sorted(seen_relations - kept)
    records = [r for r in all_records if r["relation_key"] in kept]

    templates_by_relation = {rel: load_eval_templates(rel) for rel in kept}
    summary_top7 = summarize(records, templates_by_relation)

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "interventions" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"calibration_top7_step{checkpoint_step}.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "source_recall_json": str(recall_path.relative_to(REPO_ROOT)),
                "checkpoint": data["checkpoint"],
                "retained_relation_types": sorted(kept),
                "dropped_relation_types": dropped_relations,
                "summary": summary_top7,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")

    contested = [r for r in records if r["is_contested"]]
    relations = sorted(kept)
    pairs = _split_pairs(contested)
    contested_summary = summary_top7["contested"]
    md = "\n".join(
        [
            f"# {args.run_name}: intervention baseline (top-7 relations only, checkpoint step {checkpoint_step})",
            "",
            f"Retained: {', '.join(relations)}.",
            f"Dropped (unreliable storage, excluded from the intervention population): "
            f"{', '.join(dropped_relations)}.",
            "",
            "## Table 1: background (uncontested) top1/top5 accuracy, top-7 only",
            "",
            build_table1(records, relations),
            "",
            "## Table 4: contested-pair relative rate (forced A-vs-B logit comparison), top-7 only",
            "",
            build_relative_table(contested, relations, pairs),
            "",
            "## Calibration-target summary (contested, top-7 pooled)",
            "",
            f"- n_entities_scored: {contested_summary['n_entities_scored']}",
            f"- n_entities_skipped_divergence_failure: {contested_summary['n_entities_skipped_divergence_failure']}",
            f"- monotonicity_violations: {contested_summary['monotonicity_violations']}",
            f"- symmetry_at_balance_mean_logit_gap: {contested_summary['symmetry_at_balance_mean_logit_gap']}",
            "",
            "## Cross-entropy-to-proportional-target (candidate metric, experimental_plans.tex "
            "Sec.xent-metric -- one of several under consideration, not settled)",
            "",
            f"- overall_mean_cross_entropy_to_proportional_target (nats): "
            f"{contested_summary['overall_mean_cross_entropy_to_proportional_target']:.4f}",
            f"- overall_mean_kl_to_proportional_target (nats): "
            f"{contested_summary['overall_mean_kl_to_proportional_target']:.4f}",
            "",
            "By split level (freq_gap, accuracy, mean logit gap, mean confidence in higher-freq side, "
            "mean cross-entropy, mean KL -- both to the proportional target):",
            "",
            "| n_a | n_b | freq_gap | n_entities | accuracy | mean_logit_gap_a_minus_b | "
            "mean_confidence_higher_freq_side | mean_cross_entropy | mean_kl |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        + [
            f"| {row['n_a']} | {row['n_b']} | {row['freq_gap']} | {row['n_entities']} | "
            f"{row['accuracy']*100:.1f}% | {row['mean_logit_gap_a_minus_b']:.3f} | "
            f"{row['mean_confidence_higher_freq_side']*100:.1f}% | "
            f"{row['mean_cross_entropy_to_proportional_target']:.4f} | "
            f"{row['mean_kl_to_proportional_target']:.4f} |"
            for row in contested_summary["by_split_level"]
        ]
        + [""]
    )
    out_md = out_dir / f"calibration_top7_step{checkpoint_step}.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
