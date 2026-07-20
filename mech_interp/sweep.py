"""experimental_plans.tex Sec.mechinterp-sweep: repeats representational probing,
causal tracing, and the steering dial across the split-level sweep
(experimental_plans.tex Sec.scale's 6 (n_a, n_b) levels for T=1280, and -- once run
-- Sec.seeds' exposure-budget pilot across other T values) to test whether the TYPE
of mechanism (separate vs. superposed; sharp vs. continuous) itself depends on these
training statistics. This is a thin orchestrator: it groups entities by split level
(mech_interp.common.group_by_level) and re-invokes each stage's own public function
per level, reduced to a small entity count by default -- see run_sweep.py's module
docstring for the cost warning.
"""
from __future__ import annotations

from mech_interp.causal_tracing import run_causal_map_both_directions
from mech_interp.common import group_by_level
from mech_interp.probing import run_probing_sweep
from mech_interp.steering_dial import build_dial_vector, dial_smoothness, sweep_magnitudes


def _pick_causal_locus(classification_by_granularity: dict) -> tuple[str, int, float]:
    """Picks the single component (across all requested granularities) with the
    largest combined |necessity effect| in both directions, to use as the dial's
    steering locus for this level. Returns (granularity, layer, score), with
    `layer` a transformer BLOCK index (0..n_layer-1) -- causal_tracing.py's OWN
    convention, unchanged, since that is also what score_steered/steering_utils.steering
    expects (it hooks model.transformer.h[layer]'s forward output directly, the
    same block causal_tracing.py's residual-granularity patching targets -- no
    conversion needed at this handoff). If 'residual' is among the requested
    granularities it's preferred (the dial steers the whole residual stream,
    matching CAA's own mechanism, not a single head/MLP block) --
    component-level loci from head/mlp are informative for
    causal-tracing/suppression but not directly steerable the same way
    CAA-style residual addition is.

    A real off-by-one DOES exist in this pipeline, but one hop further down:
    mech_interp.common.capture_all_layers_* (what build_dial_vector reads to
    source the vector) indexes model.forward(output_hidden_states=True)'s
    hidden_states tuple, where index `layer` is block `layer`'s INPUT, one block
    EARLIER than block `layer`'s OUTPUT that score_steered will inject at -- see
    build_dial_vector's docstring for that fix. An earlier version of this
    function incorrectly added +1 HERE instead, which crashed a full-scale run
    (`model.transformer.h[12]` doesn't exist -- only indices 0..11 do) the first
    time causal tracing's own top layer was the last block (found + fixed
    2026-07-17)."""
    if "residual" in classification_by_granularity:
        rows = classification_by_granularity["residual"]
        best = max(rows, key=lambda r: abs(r["necessity_effect_a_clean"]) + abs(r["necessity_effect_b_clean"]))
        return (
            "residual",
            best["layer"],
            abs(best["necessity_effect_a_clean"]) + abs(best["necessity_effect_b_clean"]),
        )
    # fall back to the layer containing the strongest head/mlp component, still
    # steering the FULL residual stream at that layer (the dial's vector is always
    # a residual-stream vector; only the LAYER choice is informed by a
    # non-residual granularity here).
    best = None
    for rows in classification_by_granularity.values():
        for r in rows:
            score = abs(r["necessity_effect_a_clean"]) + abs(r["necessity_effect_b_clean"])
            if best is None or score > best[2]:
                best = ("residual", r["layer"], score)
    if best is None:
        raise ValueError("no classification rows to pick a causal locus from")
    return best


def run_sweep_for_level(
    model, tokenizer, level_entities: list[dict], T: int, device: str, dtype,
    limit_entities: int, batch_size: int, causal_granularities, causal_mlp_block_size: int,
    causal_necessity_threshold: float, dial_magnitudes: list[float],
) -> dict:
    """Runs probing, causal tracing, and the dial (at causal tracing's inferred
    locus) on `level_entities` (already filtered to one (n_a, n_b) split level).
    Suppression is left to a separate, explicit run_suppression.py call per level
    (it depends on causal tracing's output file per --T, not per-level in the
    current CLI -- doing it inline here would need a per-level output path
    run_suppression.py's CLI doesn't yet accept)."""
    sample = level_entities[:limit_entities]

    print("  Probing...")
    probing_result = run_probing_sweep(model, tokenizer, sample, T, device, dtype, batch_size)

    print("  Causal tracing...")
    causal_result = run_causal_map_both_directions(
        model, tokenizer, sample, device,
        mlp_block_size=causal_mlp_block_size, granularities=causal_granularities,
        necessity_threshold=causal_necessity_threshold, verbose=False,
    )

    granularity, layer, locus_score = _pick_causal_locus(causal_result["classification_by_granularity"])
    print(f"  Dial at layer={layer} (picked from granularity={granularity}, score={locus_score:.4f})...")
    vector = build_dial_vector(model, tokenizer, sample, layer, device, dtype, batch_size)
    dial_rows = sweep_magnitudes(model, tokenizer, sample, layer, vector, dial_magnitudes, device, dtype, batch_size)
    smoothness = dial_smoothness(dial_rows)

    last_token_angles = probing_result["by_position"]["last"]
    min_angle_row = min(last_token_angles, key=lambda r: r["angle_a_b_degrees"])

    classification_summary = {}
    for g, rows in causal_result["classification_by_granularity"].items():
        classification_summary[g] = {
            "n_shared": sum(1 for r in rows if r["classification"] == "shared"),
            "n_disjoint": sum(1 for r in rows if r["classification"] == "disjoint"),
            "n_inactive": sum(1 for r in rows if r["classification"] == "inactive"),
        }

    return {
        "n_entities_used": len(sample),
        "probing_min_angle_last_token": min_angle_row,
        "causal_classification_summary": classification_summary,
        "causal_locus": {"granularity": granularity, "layer": layer, "score": locus_score},
        "dial_smoothness": smoothness,
        "dial_sweep": dial_rows,
        "probing_full": probing_result,
        "causal_full": causal_result,
    }


def run_sweep(
    model, tokenizer, entities: list[dict], T: int, device: str, dtype,
    limit_entities_per_level: int = 8, batch_size: int = 32,
    causal_granularities=("residual", "head"), causal_mlp_block_size: int = 256,
    causal_necessity_threshold: float = 0.5,
    dial_magnitudes: list[float] | None = None,
) -> dict:
    if dial_magnitudes is None:
        dial_magnitudes = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

    by_level = group_by_level(entities)
    levels = sorted(by_level, key=lambda nab: nab[0] - nab[1])

    per_level = {}
    for (n_a, n_b) in levels:
        key = f"{n_a}_{n_b}"
        print(f"=== level {key} ({len(by_level[(n_a, n_b)])} entities available) ===")
        per_level[key] = run_sweep_for_level(
            model, tokenizer, by_level[(n_a, n_b)], T, device, dtype,
            limit_entities_per_level, batch_size, list(causal_granularities), causal_mlp_block_size,
            causal_necessity_threshold, dial_magnitudes,
        )

    return {"T": T, "levels": levels, "per_level": per_level}
