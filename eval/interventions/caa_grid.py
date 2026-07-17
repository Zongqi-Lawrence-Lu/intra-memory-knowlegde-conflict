"""CAA job (results/interventions/<run-name>/caa/): joint grid over (steering
layer, multiplier), same "full 2D grid, not coordinate search" discipline as
dola_grid.py -- both rounds sweep all 12 layers.

Per-split-level vectors and configs (2026-07-12 revision): the pooled single-vector
version of this job (one diff-of-means vector per layer, averaged over ALL TRAIN
entities regardless of their (n_a, n_b) frequency split) never beat the unsteered
baseline anywhere in its 120-config grid -- see conversation, 2026-07-12. Steering
with "the average direction that separates stating-A from stating-B across the
whole population" conflates frequency-driven confidence with whatever else
distinguishes A/B-framed sentences. This version instead partitions TRAIN/VAL/TEST
by (n_a, n_b) split level (6 levels) and builds an independent vector + runs an
independent (layer, multiplier) grid search per level, so each level's steering
is fit to entities that actually share its frequency gap. Vector construction
still consumes TRAIN only (contrastive pairs: for each TRAIN entity in that level
and each of its 5 eval templates, "positive" = the full sentence asserting side A,
"negative" = the full sentence asserting side B, activation captured at the last
token -- i.e. right after the value has been stated). All 12 layers' vectors for a
given level are built from a SINGLE pass over that level's TRAIN entities
(output_hidden_states=True), not 12 separate passes. Total forward-pass volume
across all 6 levels is unchanged from the pooled version (same total entities,
same total configs), just partitioned -- so wall time should be comparable.

The layer sweep itself genuinely needs one forward pass per candidate layer on
VAL, unlike DoLa's passive read-out: steering is a causal edit to the forward pass
(inference_time/utils/steering_utils.py's forward hook on
model.transformer.h[layer]), so it can't be reconstructed post-hoc from a single
unsteered pass.

Multiplier grid is relative to the vector "as extracted" (multiplier=1.0 means add
the raw diff-of-means vector once) rather than borrowing CAA's original paper's
Llama-2-tuned absolute constant (5.0) -- this project's model differs enough in
size/scale that an absolute constant doesn't obviously transfer, whereas a
relative multiplier is scale-appropriate regardless of architecture.

Final-readout temperature fixed at STANDARD_TEMPERATURE (0.7), same as DoLa and for
the same reason: the multiplier sweep already has its own free-parameter fit
discipline (round1 coarse -> round2 fine -> pick best on VAL, report once on TEST);
temperature is not an additional free knob on top of that.

The final TEST report combines all 6 levels' own best-config steered scores into
one pooled summary (each entity scored under its own level's vector/config), so
"test" in the output JSON is directly comparable in shape to the pooled version's
and to DoLa/temperature's TEST summaries.

    python -m eval.interventions.caa_grid --T 1280
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval.interventions.common import (
    STANDARD_TEMPERATURE,
    average_over_templates,
    build_contested_probes,
    entities_in_split,
    interventions_dir,
    load_population,
    load_splits,
    run_name_for,
    summarize_at_temperature,
    top7_population,
)
from eval.recall import batched_next_token_logprobs, set_logprob, token_ids_for_texts
from inference_time.utils.model_utils import join_prompt_answer, load_trained_model
from inference_time.utils.steering_utils import steering

# Round-1 coarse grid: multiples of the raw extracted vector (not an absolute
# constant borrowed from the Llama-2-tuned original paper -- see module docstring).
COARSE_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0, 8.0]
N_FINE_MULTIPLIERS = 5
STEERING_LAYER_POSITIONS = "all"  # add the vector at every token position, not just last


@torch.no_grad()
def capture_all_layers_last_token(
    model, tokenizer, texts: list[str], device: str, dtype: torch.dtype, batch_size: int
) -> dict[int, torch.Tensor]:
    """Mean, over `texts`, of every layer's residual-stream activation at the real
    last token -- one pass per batch (output_hidden_states=True), not one pass per
    layer. Left-padding means the real last token is always at index -1, same
    convention as eval.recall.batched_next_token_logprobs."""
    n_layer = model.config.n_layer
    sums = {L: None for L in range(n_layer)}
    n = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
            out = model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, output_hidden_states=True,
            )
        for L in range(n_layer):
            hs_last = out.hidden_states[L + 1][:, -1, :].float()  # [b, d_model]
            batch_sum = hs_last.sum(dim=0)
            sums[L] = batch_sum if sums[L] is None else sums[L] + batch_sum
        n += len(batch)

    return {L: sums[L] / n for L in range(n_layer)}


def build_caa_vectors_all_layers(
    model, tokenizer, entities: list[dict], device: str, dtype: torch.dtype, batch_size: int
) -> dict[int, torch.Tensor]:
    """diff-of-means vector at every layer, built from ALL (entity, template) pairs
    in `entities` (TRAIN split) at once -- one pass over the positive texts, one
    pass over the negative texts, each covering all 12 layers simultaneously."""
    from eval.recall import build_stem, load_eval_templates

    positives, negatives = [], []
    for entity in entities:
        c = entity["contested"]
        for template in load_eval_templates(c["relation_key"]):
            stem = build_stem(template, entity["name"])
            positives.append(join_prompt_answer(stem, c["val_a"]))
            negatives.append(join_prompt_answer(stem, c["val_b"]))

    print(f"Building CAA vectors from {len(positives)} contrastive pairs (TRAIN)...")
    pos_means = capture_all_layers_last_token(model, tokenizer, positives, device, dtype, batch_size)
    neg_means = capture_all_layers_last_token(model, tokenizer, negatives, device, dtype, batch_size)
    return {L: (pos_means[L] - neg_means[L]) for L in pos_means}


@torch.no_grad()
def score_steered(
    model, tokenizer, probes: list[dict], layer: int, vector: torch.Tensor, multiplier: float,
    device: str, dtype: torch.dtype, batch_size: int,
) -> list[dict]:
    """Scores every probe with the CAA vector (scaled by `multiplier`) active on
    `layer`. `vector=None`/`multiplier=0.0` runs unsteered -- the "no intervention"
    reference sharing this exact loop."""
    records = []
    ctx = steering(model, layer, vector, multiplier=multiplier, positions=STEERING_LAYER_POSITIONS) \
        if vector is not None and multiplier != 0.0 else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for start in range(0, len(probes), batch_size):
            batch = probes[start : start + batch_size]
            stems = [p["stem"] for p in batch]
            logprobs = batched_next_token_logprobs(model, tokenizer, stems, device, dtype)
            for i, probe in enumerate(batch):
                row = logprobs[i]
                tok_a_ids = token_ids_for_texts(tokenizer, probe["tok_a_texts"])
                tok_b_ids = token_ids_for_texts(tokenizer, probe["tok_b_texts"])
                records.append(
                    {
                        "entity_id": probe["entity_id"],
                        "n_a": probe["n_a"],
                        "n_b": probe["n_b"],
                        "logp_a": set_logprob(row, tok_a_ids),
                        "logp_b": set_logprob(row, tok_b_ids),
                        "divergence_ok": set(tok_a_ids).isdisjoint(tok_b_ids),
                    }
                )
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return records


def score_configs_on_val(
    model, tokenizer, probes: list[dict], vectors: dict[int, torch.Tensor],
    configs: list[tuple[int, float]], device: str, dtype: torch.dtype, batch_size: int,
) -> dict[tuple[int, float], dict]:
    summaries = {}
    for (layer, multiplier) in configs:
        records = score_steered(model, tokenizer, probes, layer, vectors[layer], multiplier, device, dtype, batch_size)
        rows, _ = average_over_templates(records)
        summaries[(layer, multiplier)] = summarize_at_temperature(rows, STANDARD_TEMPERATURE)
    return summaries


def pick_best(summaries: dict[tuple[int, float], dict]) -> tuple[int, float]:
    return min(summaries, key=lambda cfg: summaries[cfg]["overall_mean_cross_entropy_to_proportional_target"])


def fine_multiplier_range(best_m: float, coarse: list[float]) -> list[float]:
    coarse_sorted = sorted(coarse)
    idx = coarse_sorted.index(best_m)
    lo = coarse_sorted[idx - 1] if idx > 0 else best_m * 0.5
    hi = coarse_sorted[idx + 1] if idx < len(coarse_sorted) - 1 else best_m * 2.0
    step = (hi - lo) / (N_FINE_MULTIPLIERS - 1)
    return [round(lo + i * step, 4) for i in range(N_FINE_MULTIPLIERS)]


def group_by_level(entities: list[dict]) -> dict[tuple[int, int], list[dict]]:
    """Partition entities by their contested (n_a, n_b) frequency-split level."""
    by_level: dict[tuple[int, int], list[dict]] = {}
    for e in entities:
        c = e["contested"]
        by_level.setdefault((c["n_a"], c["n_b"]), []).append(e)
    return by_level


def pooled_configs(round1: dict, round2: dict) -> dict[tuple[int, float], dict]:
    """Reconstruct the {(layer, multiplier): summary} dict from the
    {"L{layer}_m{multiplier}": summary} string-keyed dicts stored in the output
    JSON -- used by the per-level heatmap, which plots after JSON-shaped storage."""
    pooled: dict[tuple[int, float], dict] = {}
    for name, s in {**round1, **round2}.items():
        layer_part, mult_part = name.split("_m")
        pooled[(int(layer_part[1:]), float(mult_part))] = s
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--T", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit-entities", type=int, default=None,
        help="debug/smoke-test: only use the first N train/val/test entities",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_trained_model(args.T, device=device)
    tokenizer.padding_side = "left"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.run.dtype]
    n_layer = model.config.n_layer

    population = top7_population(load_population(args.T))
    assignment = load_splits(args.T)
    train_entities = entities_in_split(population, assignment, "train")
    val_entities = entities_in_split(population, assignment, "val")
    test_entities = entities_in_split(population, assignment, "test")
    if args.limit_entities is not None:
        train_entities = train_entities[: args.limit_entities]
        val_entities = val_entities[: args.limit_entities]
        test_entities = test_entities[: args.limit_entities]

    train_by_level = group_by_level(train_entities)
    val_by_level = group_by_level(val_entities)
    test_by_level = group_by_level(test_entities)
    levels = sorted(train_by_level.keys(), key=lambda nab: nab[0] - nab[1])
    print(f"TRAIN: {len(train_entities)} entities; VAL: {len(val_entities)} entities; "
          f"TEST: {len(test_entities)} entities -- across {len(levels)} split levels: {levels}")

    layers = list(range(n_layer))
    round1_configs = [(L, m) for L in layers for m in COARSE_MULTIPLIERS]

    level_results: dict[str, dict] = {}
    all_test_steered_records: list[dict] = []
    for (n_a, n_b) in levels:
        level_key = f"{n_a}_{n_b}"
        level_train = train_by_level[(n_a, n_b)]
        level_val = val_by_level.get((n_a, n_b), [])
        level_test = test_by_level.get((n_a, n_b), [])
        print(f"--- level {level_key}: TRAIN={len(level_train)} VAL={len(level_val)} TEST={len(level_test)} ---")

        vectors = build_caa_vectors_all_layers(model, tokenizer, level_train, device, dtype, args.batch_size)

        val_probes = build_contested_probes(level_val)
        round1 = score_configs_on_val(model, tokenizer, val_probes, vectors, round1_configs, device, dtype, args.batch_size)
        best_round1 = pick_best(round1)
        print(f"  round1 best: layer={best_round1[0]} multiplier={best_round1[1]} "
              f"xent={round1[best_round1]['overall_mean_cross_entropy_to_proportional_target']:.4f}")

        fine_multipliers = fine_multiplier_range(best_round1[1], COARSE_MULTIPLIERS)
        round2_configs = [(L, m) for L in layers for m in fine_multipliers]
        round2 = score_configs_on_val(model, tokenizer, val_probes, vectors, round2_configs, device, dtype, args.batch_size)

        pooled = {**round1, **round2}
        best_cfg = pick_best(pooled)
        best_layer, best_multiplier = best_cfg
        print(f"  global best (round1 ∪ round2): layer={best_layer} multiplier={best_multiplier} "
              f"xent={pooled[best_cfg]['overall_mean_cross_entropy_to_proportional_target']:.4f} "
              f"mono={pooled[best_cfg]['monotonicity_violations']}")

        test_probes = build_contested_probes(level_test)
        test_records = score_steered(
            model, tokenizer, test_probes, best_layer, vectors[best_layer], best_multiplier, device, dtype, args.batch_size
        )
        all_test_steered_records.extend(test_records)

        level_results[level_key] = {
            "n_a": n_a,
            "n_b": n_b,
            "n_train": len(level_train),
            "n_val": len(level_val),
            "n_test": len(level_test),
            "vector_norms": {str(L): v.norm().item() for L, v in vectors.items()},
            "fine_multipliers": fine_multipliers,
            "round1": {f"L{L}_m{m}": s for (L, m), s in round1.items()},
            "round2": {f"L{L}_m{m}": s for (L, m), s in round2.items()},
            "best_config": {"layer": best_layer, "multiplier": best_multiplier},
        }

    print("Final TEST scoring (per-level best configs, pooled + unsteered reference)...")
    test_probes_all = build_contested_probes(test_entities)
    test_unsteered = score_steered(
        model, tokenizer, test_probes_all, 0, None, 0.0, device, dtype, args.batch_size
    )
    rows_best, _ = average_over_templates(all_test_steered_records)
    rows_unsteered, _ = average_over_templates(test_unsteered)
    test_summary_best = summarize_at_temperature(rows_best, STANDARD_TEMPERATURE)
    test_summary_unsteered = summarize_at_temperature(rows_unsteered, STANDARD_TEMPERATURE)
    print(f"Pooled TEST: caa_best xent={test_summary_best['overall_mean_cross_entropy_to_proportional_target']:.4f} "
          f"vs no_intervention xent={test_summary_unsteered['overall_mean_cross_entropy_to_proportional_target']:.4f}")

    out_dir = interventions_dir(args.T) / "caa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "grid_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "T": args.T,
                "standard_temperature": STANDARD_TEMPERATURE,
                "coarse_multipliers": COARSE_MULTIPLIERS,
                "n_layers": n_layer,
                "per_split_vector": True,
                "levels": level_results,
                "test": {
                    "caa_best": test_summary_best,
                    "no_intervention": test_summary_unsteered,
                },
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")

    make_heatmap_per_level(level_results, layers, out_dir / "grid_heatmap.png")
    print(f"Wrote {out_dir / 'grid_heatmap.png'}")


def make_heatmap_per_level(level_results: dict[str, dict], layers: list[int], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    level_keys = list(level_results.keys())
    ncols = 3
    nrows = -(-len(level_keys) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = np.atleast_1d(axes).flatten()

    for i, key in enumerate(level_keys):
        ax = axes[i]
        lr = level_results[key]
        pooled = pooled_configs(lr["round1"], lr["round2"])
        multipliers = sorted({m for _, m in pooled})
        grid = np.full((len(layers), len(multipliers)), np.nan)
        for (L, m), s in pooled.items():
            grid[layers.index(L), multipliers.index(m)] = s["overall_mean_cross_entropy_to_proportional_target"]

        im = ax.imshow(grid, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(multipliers)))
        ax.set_xticklabels([f"{m:.2g}" for m in multipliers], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(L) for L in layers], fontsize=7)
        ax.set_title(f"n_a={lr['n_a']} n_b={lr['n_b']}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(len(level_keys), len(axes)):
        axes[j].axis("off")

    fig.suptitle("CAA VAL cross-entropy by split level (round1 ∪ round2, per-level vector)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
