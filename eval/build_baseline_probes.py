"""Converts real population + eval-template data into the prompt-pair shapes each
inference_time/ baseline method's run_*.py script already expects via its
--probe_file/--pairs_file/--bank_file/--attribution_file flags (currently the only
input those scripts ever see is a hardcoded DEMO_PROBES about real-world facts).

Reuses eval.recall's own probe construction (build_stem, load_eval_templates)
directly rather than reimplementing it, so these baseline probes are built exactly
the same way as the project's own recall-eval probes -- same stems, same template
bank. Restricted to each entity's *contested* relation only (the actual object of
conflict-mitigation baselines; background facts have their own dedicated eval in
eval/recall.py): 5 eval templates x 1,680 entities = 8,400 records per T condition.

The "cue" mechanism reuses the project's own already-specified Minimal in-context
cue baseline (experimental_plans.tex Sec.calibration-interventions: restate one
side's value immediately before the query) as the disambiguating substitute the
context-vs-parametric method families (CAD/COIECD/AdaCAD/CoCoA/ARR, and the
steering families' contrastive pairs) need in place of the external context they
were originally designed for. Since intra-memory conflict has no ground truth to
privilege one side, every cue-direction-sensitive artifact is generated *both
ways* (val_a-cued and val_b-cued) rather than picking one arbitrarily; the
experimenter selects a direction via the existing --probe_file/--pairs_file/etc.
flags on each run_*.py script.

Usage:
    python -m eval.build_baseline_probes --T 80
    python -m eval.build_baseline_probes --all

Output, under results/baseline_probes/T{T}/ (T-specific because each entity's
recorded (n_a, n_b) split differs by T even though stems/values don't -- see
inference_time/utils/model_utils.py:T_CONDITIONS):
    probes.json              {id, prompt, answers}                         -- plain scoring
    probes_cueA.json         {id, prompt, cued_prompt, cue, answers}       -- cued toward val_a
    probes_cueB.json         {id, prompt, cued_prompt, cue, answers}       -- cued toward val_b
    pairs.json               {id, positive, negative}                     -- CAA/SpARE vector construction
    pairs_contextfocus.json  {id, cued, bare}                              -- ContextFocus's own key names
    bank.json                {id, positive, negative, query}               -- K-CAST instance bank
    attribution.json         {id, clean_prompt, corrupted_prompt, answer_pos, answer_neg}  -- PH3 phase-1
    index.json                id -> {entity_id, relation_key, template_idx, n_a, n_b,
                               split_level, val_a, val_b}
                              No run_*.py script preserves probe metadata into its own
                              results/*.json output (each only reads specific known
                              keys), so this is the join key for post-hoc analysis by
                              relation type / split level.
    manifest.json             population path, counts, generation timestamp (provenance)

pairs/pairs_contextfocus/bank/attribution are one-time, vector-construction-only
artifacts (per each method's own docstring) -- built from the first_mention
template only (template_idx==0), 1,680 records, to keep that one-time set canonical
rather than 5x-inflated; probes/probes_cueA/probes_cueB (the actual evaluation
targets) use all 5 templates.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from eval.recall import build_stem, load_eval_templates
from inference_time.utils.model_utils import REPO_ROOT, T_CONDITIONS


def cue(val: str, stem: str) -> str:
    """Minimal in-context cue (experimental_plans.tex Sec.calibration-interventions):
    restate one side's value immediately before the query."""
    return f"{val}. {stem}"


def build_records(population: list[dict]) -> list[dict]:
    """One record per (entity, eval template) for each entity's contested relation."""
    records = []
    for entity in population:
        c = entity["contested"]
        templates = load_eval_templates(c["relation_key"])
        for template_idx, template in enumerate(templates):
            stem = build_stem(template, entity["name"])
            records.append(
                {
                    "id": f"{entity['entity_id']}__t{template_idx}",
                    "entity_id": entity["entity_id"],
                    "relation_key": c["relation_key"],
                    "template_idx": template_idx,
                    "stem": stem,
                    "val_a": c["val_a"],
                    "val_b": c["val_b"],
                    "n_a": c["n_a"],
                    "n_b": c["n_b"],
                }
            )
    return records


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_for_T(T: int, out_root: Path) -> dict:
    cond = T_CONDITIONS[T]
    population_path = REPO_ROOT / cond["population"]
    population = json.load(open(population_path))
    records = build_records(population)
    vector_records = [r for r in records if r["template_idx"] == 0]

    out_dir = out_root / f"T{T}"

    probes = [{"id": r["id"], "prompt": r["stem"], "answers": [r["val_a"], r["val_b"]]} for r in records]

    def cued_probes(cue_side: str) -> list[dict]:
        key = "val_a" if cue_side == "a" else "val_b"
        out = []
        for r in records:
            cued = cue(r[key], r["stem"])
            out.append(
                {
                    "id": r["id"],
                    "prompt": r["stem"],
                    "cued_prompt": cued,
                    "cue": cued,
                    "answers": [r["val_a"], r["val_b"]],
                }
            )
        return out

    pairs = [
        {"id": r["id"], "positive": cue(r["val_a"], r["stem"]), "negative": cue(r["val_b"], r["stem"])}
        for r in vector_records
    ]
    pairs_contextfocus = [
        {"id": r["id"], "cued": cue(r["val_a"], r["stem"]), "bare": r["stem"]} for r in vector_records
    ]
    bank = [
        {
            "id": r["id"],
            "positive": cue(r["val_a"], r["stem"]),
            "negative": cue(r["val_b"], r["stem"]),
            "query": r["stem"],
        }
        for r in vector_records
    ]
    attribution = [
        {
            "id": r["id"],
            "clean_prompt": cue(r["val_a"], r["stem"]),
            "corrupted_prompt": r["stem"],
            "answer_pos": r["val_a"],
            "answer_neg": r["val_b"],
        }
        for r in vector_records
    ]
    index = {
        r["id"]: {
            "entity_id": r["entity_id"],
            "relation_key": r["relation_key"],
            "template_idx": r["template_idx"],
            "n_a": r["n_a"],
            "n_b": r["n_b"],
            "split_level": f"{r['n_a']}_{r['n_b']}",
            "val_a": r["val_a"],
            "val_b": r["val_b"],
        }
        for r in records
    }

    write_json(probes, out_dir / "probes.json")
    write_json(cued_probes("a"), out_dir / "probes_cueA.json")
    write_json(cued_probes("b"), out_dir / "probes_cueB.json")
    write_json(pairs, out_dir / "pairs.json")
    write_json(pairs_contextfocus, out_dir / "pairs_contextfocus.json")
    write_json(bank, out_dir / "bank.json")
    write_json(attribution, out_dir / "attribution.json")
    write_json(index, out_dir / "index.json")

    manifest = {
        "T": T,
        "run_name": cond["run_name"],
        "population_path": str(population_path),
        "n_entities": len(population),
        "n_records": len(records),
        "n_vector_records": len(vector_records),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    write_json(manifest, out_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--T", type=int, choices=sorted(T_CONDITIONS), default=None)
    parser.add_argument("--all", action="store_true", help="Build for every T condition.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "baseline_probes"))
    args = parser.parse_args()

    if not args.all and args.T is None:
        parser.error("pass --T {80,320,1280} or --all")

    targets = sorted(T_CONDITIONS) if args.all else [args.T]
    out_root = Path(args.out_dir)
    for T in targets:
        manifest = build_for_T(T, out_root)
        print(
            f"T={T}: {manifest['n_records']} probes, {manifest['n_vector_records']} "
            f"vector-construction records -> {out_root / f'T{T}'}"
        )


if __name__ == "__main__":
    main()
