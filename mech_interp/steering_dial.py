"""Dial (experimental_plans.tex Sec.mechinterp-dial): continuous control -- turns
causal tracing's causal flip into a dial. Builds a diff-of-means steering vector at
the causal-tracing-identified locus (a layer, passed in explicitly -- NOT
re-derived from a fixed a-priori layer the way
eval.interventions.caa_grid's CAA baseline picks its own layer via a VAL-set grid
search), scales it continuously, and records the resulting output distribution
over {A, B} as a function of magnitude, compared against the true exposure-ratio
target q_A = n_a/(n_a+n_b) (experimental_plans.tex Sec.xent-metric).

Reuses eval.interventions.caa_grid.score_steered (identical steering + scoring
mechanics to the CAA baseline -- only the vector's SOURCE layer and this stage's
purpose differ) rather than re-implementing steered scoring from scratch.
"""
from __future__ import annotations

import statistics

from eval.interventions.caa_grid import score_steered
from eval.interventions.common import STANDARD_TEMPERATURE, average_over_templates, build_contested_probes, sigmoid
from mech_interp.common import capture_all_layers_last_token, contested_side_examples


def build_dial_vector(model, tokenizer, entities: list[dict], layer: int, device: str, dtype, batch_size: int = 32):
    """diff-of-means vector at `layer`, from ALL (entity, side, template)
    contrastive examples in `entities` -- same construction as
    eval.interventions.caa_grid.build_caa_vectors_all_layers, but for a single,
    externally-chosen layer (the causal-tracing-identified locus) rather than all
    12 at once.

    `layer` is a transformer BLOCK index (0..n_layer-1), the SAME convention
    causal_tracing.py uses and score_steered/inference_time.utils.steering_utils.steering
    expects (register_addition_hook hooks model.transformer.h[layer]'s forward
    output directly, i.e. adds the vector to block `layer`'s own output).
    mech_interp.common.capture_all_layers_last_token instead returns the raw
    model.forward(output_hidden_states=True) tuple, where index `layer` is block
    `layer`'s INPUT (= block (layer-1)'s output), one block EARLIER than where
    score_steered will inject the vector -- so this function reads index
    `layer + 1` to source the vector from the same position it will later be
    added at. Getting this wrong doesn't crash (both `layer` and `layer+1` are
    valid hidden_states indices for any block 0..n_layer-1) -- it silently builds
    the vector one block off from the injection point instead. Confirmed against
    eval.interventions.caa_grid.capture_all_layers_last_token, which performs the
    identical `hidden_states[L + 1]` conversion internally for CAA's own vectors
    (found during 2026-07-17 review, once the FIRST attempted fix -- adding this
    +1 at the causal-tracing-to-dial layer handoff instead -- crashed a full-scale
    run with `IndexError: index 12 is out of range` on model.transformer.h[12],
    which only has indices 0..11: that handoff is not where the mismatch lives,
    this read is)."""
    records = contested_side_examples(entities)
    a_texts = [r["text"] for r in records if r["side"] == "A"]
    b_texts = [r["text"] for r in records if r["side"] == "B"]
    acts_a = capture_all_layers_last_token(model, tokenizer, a_texts, device, dtype, batch_size)[:, layer + 1, :]
    acts_b = capture_all_layers_last_token(model, tokenizer, b_texts, device, dtype, batch_size)[:, layer + 1, :]
    return acts_a.mean(dim=0) - acts_b.mean(dim=0)


def sweep_magnitudes(
    model, tokenizer, entities: list[dict], layer: int, vector, magnitudes: list[float],
    device: str, dtype, batch_size: int = 32,
) -> list[dict]:
    """For each magnitude in `magnitudes`, scales `vector` and scores every
    contested probe steered at `layer`; returns one row per magnitude with the
    resulting pooled/per-split-level p_A vs. the true proportional target."""
    probes = build_contested_probes(entities)
    rows = []
    for m in magnitudes:
        records = score_steered(model, tokenizer, probes, layer, vector, m, device, dtype, batch_size)
        entity_rows, n_skipped = average_over_templates(records)
        p_a_by_entity = []
        for r in entity_rows:
            gap = r["logp_a"] - r["logp_b"]
            p_a = sigmoid(gap / STANDARD_TEMPERATURE)
            q_a = r["n_a"] / (r["n_a"] + r["n_b"])
            p_a_by_entity.append(
                {"entity_id": r["entity_id"], "n_a": r["n_a"], "n_b": r["n_b"], "p_a": p_a, "q_a": q_a}
            )

        by_level: dict[tuple[int, int], list[dict]] = {}
        for row in p_a_by_entity:
            by_level.setdefault((row["n_a"], row["n_b"]), []).append(row)
        level_summary = [
            {
                "n_a": n_a,
                "n_b": n_b,
                "mean_p_a": statistics.mean(r["p_a"] for r in recs),
                "target_q_a": n_a / (n_a + n_b),
            }
            for (n_a, n_b), recs in sorted(by_level.items(), key=lambda kv: kv[0][0] - kv[0][1])
        ]

        rows.append(
            {
                "magnitude": m,
                "n_entities_scored": len(p_a_by_entity),
                "n_entities_skipped": n_skipped,
                "mean_p_a": statistics.mean(r["p_a"] for r in p_a_by_entity) if p_a_by_entity else None,
                "by_split_level": level_summary,
            }
        )
    return rows


def dial_smoothness(rows: list[dict]) -> dict:
    """Crude smoothness diagnostic on the pooled mean_p_a-vs-magnitude curve: the
    largest single-step jump in mean_p_a between adjacent swept magnitudes, vs. the
    overall range covered. A curve dominated by one or two large steps (jump close
    to the full range) looks discrete/threshold-like; many comparably-sized small
    steps looks smooth/interpolable -- experimental_plans.tex
    Sec.mechinterp-dial's diagnostic for separately-stored-with-a-selector vs.
    blended/superposed representation."""
    p = [r["mean_p_a"] for r in rows if r["mean_p_a"] is not None]
    if len(p) < 2:
        return {"max_step": None, "range": None, "max_step_fraction_of_range": None}
    steps = [abs(p[i + 1] - p[i]) for i in range(len(p) - 1)]
    rng = max(p) - min(p)
    return {
        "max_step": max(steps),
        "range": rng,
        "max_step_fraction_of_range": (max(steps) / rng) if rng > 0 else None,
    }
