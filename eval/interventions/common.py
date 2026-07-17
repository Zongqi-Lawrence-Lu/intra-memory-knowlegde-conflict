"""Shared engine for the three intervention jobs (temperature/dola/caa -- each its
own CLI script, own results/interventions/<run-name>/<method>/ folder, no shared
output files). What IS shared, deliberately, is:

  1. The population (top-7 reliably-stored relation types only, T=1280,
     inference_time.utils.model_utils.RELIABLE_RELATION_TYPES).
  2. One fixed entity-level train/val/test split (eval.interventions.make_splits),
     generated once and read by all three jobs, so "validation" and "test" mean
     the same population everywhere. A single unit is one entity (each entity has
     exactly one contested relation by construction, so entity == contested trait).
  3. One metric function (summarize_at_temperature) that all three jobs funnel
     their own per-(entity, template) logp_a/logp_b through -- they differ only in
     how those log-probabilities were produced (raw model, DoLa-contrasted logits,
     CAA-steered logits), not in how confidence/cross-entropy/KL/monotonicity are
     computed from them afterward.

Deliberately NOT shared: any single "no intervention" reference file. Each job
computes its own raw/unsteered baseline internally if it needs one for comparison,
so results/interventions/<run-name>/<method>/ is self-contained.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import torch

from eval.recall import build_stem, candidate_token_texts, load_eval_templates
from inference_time.utils.model_utils import RELIABLE_RELATION_TYPES, T_CONDITIONS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Fixed final-metric temperature for DoLa/CAA (experimental_plans.tex discussion,
# 2026-07-12): the calibration-target confidence σ((ℓ_A-ℓ_B)/τ) needs a temperature
# just like the temperature-scaling job does, but DoLa/CAA are not allowed to fit it
# -- fitting τ is what makes temperature scaling its own method. τ=0.7 is a fixed,
# externally-chosen (not metric-optimized) standard applied identically to both, so
# neither secretly wins on sharpness alone.
STANDARD_TEMPERATURE = 0.7


# --------------------------------------------------------------------------- population

def load_population(T: int = 1280) -> list[dict]:
    path = REPO_ROOT / T_CONDITIONS[T]["population"]
    with open(path) as f:
        return json.load(f)


def top7_population(population: list[dict]) -> list[dict]:
    kept = set(RELIABLE_RELATION_TYPES)
    return [e for e in population if e["contested"]["relation_key"] in kept]


def run_name_for(T: int = 1280) -> str:
    return T_CONDITIONS[T]["run_name"]


def interventions_dir(T: int = 1280) -> Path:
    d = REPO_ROOT / "results" / "interventions" / run_name_for(T)
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- splits

def splits_path(T: int = 1280) -> Path:
    return interventions_dir(T) / "splits.json"


def make_splits(
    entities: list[dict], train_frac: float = 0.5, val_frac: float = 0.25, seed: int = 0
) -> dict[str, str]:
    """Entity-level train/val/test partition, stratified by (n_a, n_b) split level
    so every level is represented proportionally in all three sets. One unit =
    one entity. Deterministic given `seed` -- this is meant to be generated once
    and then treated as fixed."""
    by_level: dict[tuple[int, int], list[str]] = {}
    for e in entities:
        c = e["contested"]
        by_level.setdefault((c["n_a"], c["n_b"]), []).append(e["entity_id"])

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for level, ids in sorted(by_level.items()):
        ids = sorted(ids)  # stable order before shuffling, independent of dict/json order
        rng.shuffle(ids)
        n = len(ids)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        for eid in ids[:n_train]:
            assignment[eid] = "train"
        for eid in ids[n_train : n_train + n_val]:
            assignment[eid] = "val"
        for eid in ids[n_train + n_val :]:
            assignment[eid] = "test"
    return assignment


def save_splits(assignment: dict[str, str], entities: list[dict], T: int = 1280, **meta) -> Path:
    by_level_counts: dict[str, dict[str, int]] = {}
    for e in entities:
        c = e["contested"]
        key = f"{c['n_a']}_{c['n_b']}"
        counts = by_level_counts.setdefault(key, {"train": 0, "val": 0, "test": 0})
        counts[assignment[e["entity_id"]]] += 1

    out = {
        "T": T,
        "meta": meta,
        "counts_by_split_level": by_level_counts,
        "assignment": assignment,
    }
    path = splits_path(T)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def load_splits(T: int = 1280) -> dict[str, str]:
    path = splits_path(T)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m eval.interventions.make_splits --T {T}` first."
        )
    with open(path) as f:
        return json.load(f)["assignment"]


def entities_in_split(entities: list[dict], assignment: dict[str, str], split: str) -> list[dict]:
    return [e for e in entities if assignment.get(e["entity_id"]) == split]


# --------------------------------------------------------------------------- probes

def build_contested_probes(entities: list[dict]) -> list[dict]:
    """One probe per entity per eval template (N_EVAL_TEMPLATES=5), contested fact
    only -- background facts aren't needed for intervention scoring. Mirrors
    eval.recall.build_probes' contested branch exactly, so token-candidate
    construction stays identical to the recall pipeline these numbers are compared
    against."""
    probes = []
    templates_by_relation = {
        e["contested"]["relation_key"]: load_eval_templates(e["contested"]["relation_key"])
        for e in entities
    }
    for entity in entities:
        c = entity["contested"]
        templates = templates_by_relation[c["relation_key"]]
        for template_idx, template in enumerate(templates):
            stem = build_stem(template, entity["name"])
            tok_a_texts = candidate_token_texts(template, c["val_a"], name=entity["name"])
            tok_b_texts = candidate_token_texts(template, c["val_b"], name=entity["name"])
            probes.append(
                {
                    "entity_id": entity["entity_id"],
                    "relation_key": c["relation_key"],
                    "template_idx": template_idx,
                    "stem": stem,
                    "val_a": c["val_a"],
                    "val_b": c["val_b"],
                    "tok_a_texts": tok_a_texts,
                    "tok_b_texts": tok_b_texts,
                    "n_a": c["n_a"],
                    "n_b": c["n_b"],
                }
            )
    return probes


# --------------------------------------------------------------------------- metric

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def average_over_templates(records: list[dict]) -> list[dict]:
    """records: one dict per (entity, template) with entity_id, n_a, n_b, logp_a,
    logp_b, divergence_ok. Averages logp_a/logp_b across only the divergence-ok
    templates per entity (eval.recall._summarize_contested's per-entity step,
    reused verbatim so the same entity gets the same treatment regardless of which
    job produced its logp_a/logp_b). Entities with zero divergence-ok templates are
    dropped."""
    by_entity: dict[str, list[dict]] = {}
    for r in records:
        by_entity.setdefault(r["entity_id"], []).append(r)

    rows = []
    n_skipped = 0
    for entity_id, recs in by_entity.items():
        valid = [r for r in recs if r["divergence_ok"]]
        if not valid:
            n_skipped += 1
            continue
        rows.append(
            {
                "entity_id": entity_id,
                "n_a": valid[0]["n_a"],
                "n_b": valid[0]["n_b"],
                "logp_a": statistics.mean(r["logp_a"] for r in valid),
                "logp_b": statistics.mean(r["logp_b"] for r in valid),
                "n_templates_used": len(valid),
            }
        )
    return rows, n_skipped


def summarize_at_temperature(entity_rows: list[dict], temperature: float) -> dict:
    """entity_rows: output of average_over_templates (one row per entity, already
    averaged across templates). Computes the calibration-target metric
    (experimental_plans.tex Sec.xent-metric) at a given final-readout temperature:
    p_a = sigmoid((logp_a - logp_b) / temperature). Monotonicity is computed from
    the raw (temperature-independent) oriented logit gap, since a positive-scalar
    divide can never change its sign/ordering -- by design, no value of
    `temperature` can move this number; that invariance is the whole point of
    reporting it alongside cross-entropy/KL rather than instead of them."""
    contested = []
    for r in entity_rows:
        n_a, n_b, logp_a, logp_b = r["n_a"], r["n_b"], r["logp_a"], r["logp_b"]
        gap = logp_a - logp_b
        p_a = sigmoid(gap / temperature)
        p_a_clipped = min(max(p_a, 1e-12), 1 - 1e-12)
        q_a = n_a / (n_a + n_b)
        cross_entropy = -(q_a * math.log(p_a_clipped) + (1 - q_a) * math.log(1 - p_a_clipped))
        target_entropy = -(q_a * math.log(q_a) + (1 - q_a) * math.log(1 - q_a)) if 0 < q_a < 1 else 0.0
        contested.append(
            {
                "entity_id": r["entity_id"],
                "n_a": n_a,
                "n_b": n_b,
                "raw_logit_gap": gap,
                "p_a": p_a,
                "target_p_a": q_a,
                "cross_entropy_to_proportional_target": cross_entropy,
                "kl_to_proportional_target": cross_entropy - target_entropy,
            }
        )

    by_split: dict[tuple[int, int], list[dict]] = {}
    for r in contested:
        by_split.setdefault((r["n_a"], r["n_b"]), []).append(r)

    split_summary = []
    for (n_a, n_b), recs in sorted(by_split.items(), key=lambda kv: kv[0][0] - kv[0][1]):
        split_summary.append(
            {
                "n_a": n_a,
                "n_b": n_b,
                "freq_gap": n_a - n_b,
                "n_entities": len(recs),
                "mean_cross_entropy_to_proportional_target": statistics.mean(
                    r["cross_entropy_to_proportional_target"] for r in recs
                ),
                "mean_kl_to_proportional_target": statistics.mean(
                    r["kl_to_proportional_target"] for r in recs
                ),
                "mean_raw_logit_gap": statistics.mean(r["raw_logit_gap"] for r in recs),
            }
        )

    oriented_gaps = [
        row["mean_raw_logit_gap"] if row["n_a"] >= row["n_b"] else -row["mean_raw_logit_gap"]
        for row in split_summary
    ]
    monotonicity_violations = sum(
        1 for i in range(1, len(oriented_gaps)) if oriented_gaps[i] < oriented_gaps[i - 1]
    )
    balanced = next((row for row in split_summary if row["n_a"] == row["n_b"]), None)

    return {
        "temperature": temperature,
        "n_entities_scored": len(contested),
        "monotonicity_violations": monotonicity_violations,
        "symmetry_at_balance_mean_logit_gap": balanced["mean_raw_logit_gap"] if balanced else None,
        "overall_mean_cross_entropy_to_proportional_target": (
            statistics.mean(r["cross_entropy_to_proportional_target"] for r in contested) if contested else None
        ),
        "overall_mean_kl_to_proportional_target": (
            statistics.mean(r["kl_to_proportional_target"] for r in contested) if contested else None
        ),
        "by_split_level": split_summary,
    }
