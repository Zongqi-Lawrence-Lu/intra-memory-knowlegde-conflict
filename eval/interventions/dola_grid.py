"""DoLa job (results/interventions/<run-name>/dola/): joint grid over (premature
layer, alpha), not a coordinate-wise search -- both rounds sweep all 12 layers, so
the winning layer is allowed to change between round 1 (coarse alpha) and round 2
(fine alpha) rather than being frozen after round 1 (2026-07-12 conversation: "this
need to be a two by two grid").

Static-mode DoLa (inference_time/dola.py's mode="dynamic" picks a premature layer
per-token via argmax JSD, which has no single enumerable "layer" -- static mode
fixes one, which is what "try every single layer" needs). Internal softmax
temperature stays at 1.0 (paper default); the standard tau=0.7
(eval.interventions.common.STANDARD_TEMPERATURE) is applied only afterward, at the
final p_a = sigmoid(gap / tau) readout, same place as CAA and the "no
intervention" row -- this keeps "does layer-contrast help" isolated from "how
sharp is the readout."

Layer sweep is cheap: since the premature-layer distribution is a passive
logit-lens read-out (not a causal change to the forward pass), all 12 layers'
hidden states are captured from ONE unmodified forward pass per probe batch; the
12-layer x alpha grid is then just cheap log-softmax/subtract arithmetic on
already-computed tensors, not 12x more forward passes.

    python -m eval.interventions.dola_grid --T 1280
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

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
from eval.recall import token_ids_for_texts
from inference_time.dola import _layer_logits
from inference_time.utils.model_utils import load_trained_model

# Round-1 coarse grid: log-spaced around the paper/repo default alpha=1.0, covering
# an order of magnitude of under- and over-contrasting each way.
COARSE_ALPHAS = [0.1, 0.5, 1.0, 2.0, 4.0]
N_FINE_ALPHAS = 5


@torch.no_grad()
def batched_final_and_layer_logprobs(model, tokenizer, stems: list[str], device: str, dtype: torch.dtype):
    """Left-padded batch forward pass (mirrors eval.recall.batched_next_token_logprobs
    exactly, including the position_ids fix for left padding) that additionally
    returns every layer's log-softmax'd logit-lens distribution at the real last
    token, so a 12-layer x alpha grid can be scored from this single pass."""
    encoded = tokenizer(stems, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)

    with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
    final_logprobs = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head
    n_layer = model.config.n_layer
    # hidden_states[0] is the embedding output (pre-block-0); hidden_states[i] for
    # i=1..n_layer is the output of block (i-1) -- i.e. block index L's output is
    # hidden_states[L + 1] (HF convention).
    layer_logprobs = []
    for L in range(n_layer):
        hs_last = out.hidden_states[L + 1][:, -1, :]
        logits = _layer_logits(hs_last, ln_f, lm_head)
        layer_logprobs.append(F.log_softmax(logits, dim=-1))

    return final_logprobs, layer_logprobs


def contrast_and_read(final_logprobs, premature_logprobs, alpha: float, tok_ids: list[int]) -> torch.Tensor:
    """log P_final - alpha * log P_premature, renormalized, then summed (logsumexp)
    over a candidate token-id set -- same aggregation as eval.recall.set_logprob,
    just applied to the contrastive distribution instead of the raw model's."""
    contrast = final_logprobs - alpha * premature_logprobs
    contrast = contrast - torch.logsumexp(contrast, dim=-1, keepdim=True)
    idx = torch.tensor(tok_ids, device=contrast.device)
    return torch.logsumexp(contrast[:, idx], dim=-1)


def score_grid(
    model, tokenizer, probes: list[dict], layer_alphas: list[tuple[int, float]],
    device: str, dtype: torch.dtype, batch_size: int,
) -> dict[tuple[int, float], list[dict]]:
    """Returns {(layer, alpha): [per-(entity,template) record, ...]}."""
    out: dict[tuple[int, float], list[dict]] = {cfg: [] for cfg in layer_alphas}
    layers_needed = sorted({L for L, _ in layer_alphas})

    for start in range(0, len(probes), batch_size):
        batch = probes[start : start + batch_size]
        stems = [p["stem"] for p in batch]
        final_logprobs, layer_logprobs = batched_final_and_layer_logprobs(model, tokenizer, stems, device, dtype)

        tok_a_ids_batch = [token_ids_for_texts(tokenizer, p["tok_a_texts"]) for p in batch]
        tok_b_ids_batch = [token_ids_for_texts(tokenizer, p["tok_b_texts"]) for p in batch]
        divergence_ok_batch = [set(a).isdisjoint(b) for a, b in zip(tok_a_ids_batch, tok_b_ids_batch)]

        for L in layers_needed:
            premature = layer_logprobs[L]
            for (cfg_L, alpha) in layer_alphas:
                if cfg_L != L:
                    continue
                # alpha=0.0 needs no special case: contrast = final - 0*premature is
                # just final, renormalized (a no-op since final is already
                # log-softmax'd) -- this is the "no intervention" reference row.
                for i, probe in enumerate(batch):
                    logp_a = contrast_and_read(
                        final_logprobs[i : i + 1], premature[i : i + 1], alpha, tok_a_ids_batch[i]
                    ).item()
                    logp_b = contrast_and_read(
                        final_logprobs[i : i + 1], premature[i : i + 1], alpha, tok_b_ids_batch[i]
                    ).item()
                    out[(cfg_L, alpha)].append(
                        {
                            "entity_id": probe["entity_id"],
                            "n_a": probe["n_a"],
                            "n_b": probe["n_b"],
                            "logp_a": logp_a,
                            "logp_b": logp_b,
                            "divergence_ok": divergence_ok_batch[i],
                        }
                    )
    return out


def score_configs_on_val(
    model, tokenizer, probes: list[dict], configs: list[tuple[int, float]],
    device: str, dtype: torch.dtype, batch_size: int,
) -> dict[tuple[int, float], dict]:
    raw = score_grid(model, tokenizer, probes, configs, device, dtype, batch_size)
    summaries = {}
    for cfg, records in raw.items():
        rows, _ = average_over_templates(records)
        summaries[cfg] = summarize_at_temperature(rows, STANDARD_TEMPERATURE)
    return summaries


def pick_best(summaries: dict[tuple[int, float], dict]) -> tuple[int, float]:
    return min(summaries, key=lambda cfg: summaries[cfg]["overall_mean_cross_entropy_to_proportional_target"])


def fine_alpha_range(best_alpha: float, coarse: list[float]) -> list[float]:
    coarse_sorted = sorted(coarse)
    idx = coarse_sorted.index(best_alpha)
    lo = coarse_sorted[idx - 1] if idx > 0 else best_alpha * 0.5
    hi = coarse_sorted[idx + 1] if idx < len(coarse_sorted) - 1 else best_alpha * 2.0
    step = (hi - lo) / (N_FINE_ALPHAS - 1)
    return [round(lo + i * step, 4) for i in range(N_FINE_ALPHAS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--T", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit-entities", type=int, default=None,
        help="debug/smoke-test: only use the first N val and N test entities",
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
    val_entities = entities_in_split(population, assignment, "val")
    test_entities = entities_in_split(population, assignment, "test")
    if args.limit_entities is not None:
        val_entities = val_entities[: args.limit_entities]
        test_entities = test_entities[: args.limit_entities]
    val_probes = build_contested_probes(val_entities)
    test_probes = build_contested_probes(test_entities)
    print(f"VAL: {len(val_entities)} entities / {len(val_probes)} probes; "
          f"TEST: {len(test_entities)} entities / {len(test_probes)} probes")

    layers = list(range(n_layer))

    # Round 1: full joint grid, all layers x coarse alphas.
    round1_configs = [(L, a) for L in layers for a in COARSE_ALPHAS]
    print(f"Round 1: {len(round1_configs)} (layer, alpha) configs on VAL...")
    round1 = score_configs_on_val(model, tokenizer, val_probes, round1_configs, device, dtype, args.batch_size)
    best_round1 = pick_best(round1)
    print(f"Round 1 best: layer={best_round1[0]} alpha={best_round1[1]} "
          f"xent={round1[best_round1]['overall_mean_cross_entropy_to_proportional_target']:.4f}")

    fine_alphas = fine_alpha_range(best_round1[1], COARSE_ALPHAS)
    round2_configs = [(L, a) for L in layers for a in fine_alphas]
    print(f"Round 2: {len(round2_configs)} (layer, alpha) configs on VAL, fine alphas={fine_alphas}...")
    round2 = score_configs_on_val(model, tokenizer, val_probes, round2_configs, device, dtype, args.batch_size)

    pooled = {**round1, **round2}
    best_cfg = pick_best(pooled)
    best_layer, best_alpha = best_cfg
    print(f"Global best (round1 ∪ round2): layer={best_layer} alpha={best_alpha} "
          f"xent={pooled[best_cfg]['overall_mean_cross_entropy_to_proportional_target']:.4f} "
          f"mono={pooled[best_cfg]['monotonicity_violations']}")

    # Final: winning config + the alpha=0 (no-contrast, i.e. raw model) reference,
    # both evaluated exactly once on TEST.
    test_configs = [best_cfg, (best_layer, 0.0)]
    print(f"Final TEST scoring: {test_configs}...")
    test_summaries = score_configs_on_val(model, tokenizer, test_probes, test_configs, device, dtype, args.batch_size)

    out_dir = interventions_dir(args.T) / "dola"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "grid_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "T": args.T,
                "standard_temperature": STANDARD_TEMPERATURE,
                "coarse_alphas": COARSE_ALPHAS,
                "fine_alphas": fine_alphas,
                "n_layers": n_layer,
                "round1": {f"L{L}_a{a}": s for (L, a), s in round1.items()},
                "round2": {f"L{L}_a{a}": s for (L, a), s in round2.items()},
                "best_config": {"layer": best_layer, "alpha": best_alpha},
                "test": {
                    "dola_best": test_summaries[best_cfg],
                    "no_intervention_same_layer": test_summaries[(best_layer, 0.0)],
                },
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")

    make_heatmap(pooled, layers, out_dir / "grid_heatmap.png")
    print(f"Wrote {out_dir / 'grid_heatmap.png'}")


def make_heatmap(pooled: dict[tuple[int, float], dict], layers: list[int], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    alphas = sorted({a for _, a in pooled})
    grid = np.full((len(layers), len(alphas)), np.nan)
    for (L, a), s in pooled.items():
        grid[layers.index(L), alphas.index(a)] = s["overall_mean_cross_entropy_to_proportional_target"]

    fig, ax = plt.subplots(figsize=(max(6, len(alphas) * 0.6), 6))
    im = ax.imshow(grid, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(a) for a in alphas], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(L) for L in layers], fontsize=8)
    ax.set_xlabel("alpha")
    ax.set_ylabel("premature layer")
    ax.set_title("DoLa VAL cross-entropy (round1 ∪ round2, pooled)")
    fig.colorbar(im, ax=ax, label="cross-entropy (nats)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
