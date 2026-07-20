"""Suppression (experimental_plans.tex Sec.mechinterp-suppression): suppression vs.
erasure check, using the top causally-necessary components identified by causal
tracing (mech_interp.causal_tracing.classify_components / aggregate_causal_maps).

Runs the logit lens (nostalgebraist2020logitlens, via mech_interp.common.unembed_hidden)
at every layer of the model's PLAIN, uncued contested-relation probe (the same
template_idx=0 stem eval.recall/scaffolding already use), once unmodified and once with
causal tracing's top necessary components zero-ablated throughout the whole sequence. If
the losing side's logit-lens rank/logprob is already comparable to the winning side's at
an early/mid layer in the UNMODIFIED trace -- despite the final layer strongly favoring
the winner -- that is meng2022rome's 'suppression not erasure' pattern: the losing side
was represented at full fidelity upstream, and specific downstream components (causal
tracing's necessary set) are what suppress it by the final layer. This directly resolves
the project's earlier "both sides show up in the logits" observation into a located
mechanism.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from eval.recall import build_stem, candidate_token_texts, load_eval_templates, set_logprob, token_ids_for_texts
from mech_interp.common import left_pad_encode, unembed_hidden


@contextmanager
def ablate_components(model, components: list[dict]):
    """components: list of {"granularity": "head"|"mlp", "layer": int,
    "component_index": int (head) or [start, stop] (mlp block)}. Zero-ablates all
    listed components, at EVERY sequence position (unlike causal tracing's patching,
    which is restricted to the shared query-stem span -- here there is only one
    prompt, no alignment concern), for the duration of the context."""
    head_dim = model.config.n_embd // model.config.n_head
    handles = []
    for comp in components:
        layer = comp["layer"]
        if comp["granularity"] == "head":
            sl = slice(comp["component_index"] * head_dim, (comp["component_index"] + 1) * head_dim)
            module = model.transformer.h[layer].attn.c_proj
        elif comp["granularity"] == "mlp":
            ci = comp["component_index"]
            sl = slice(ci[0], ci[1]) if isinstance(ci, (list, tuple)) else ci
            module = model.transformer.h[layer].mlp.c_proj
        else:
            raise ValueError(f"unsupported granularity for suppression ablation: {comp['granularity']}")

        def hook(module, args, sl=sl):
            x = args[0].clone()
            x[..., sl] = 0.0
            return (x,) + args[1:]

        handles.append(module.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def logit_lens_trace(model, tokenizer, prompt: str, tok_a_ids: list[int], tok_b_ids: list[int], device: str) -> list[dict]:
    """Per-layer (0=embeddings..n_layer=final block output) log P(A)/log P(B) at
    `prompt`'s final position, via the logit lens applied to each cached hidden
    state."""
    input_ids, attention_mask, position_ids = left_pad_encode(tokenizer, [prompt], device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=True)
    trace = []
    for layer_idx, hs in enumerate(out.hidden_states):
        logits = unembed_hidden(model, hs[:, -1, :])
        logp = torch.log_softmax(logits[0], dim=-1)
        trace.append({"layer": layer_idx, "logp_a": set_logprob(logp, tok_a_ids), "logp_b": set_logprob(logp, tok_b_ids)})
    return trace


def run_suppression_check(model, tokenizer, entity: dict, device: str, components: list[dict]) -> dict:
    """entity: one population entity. components: top-K necessary components from
    causal tracing, ablated together for the 'ablated' trace. Uses the plain (uncued)
    contested-relation probe stem (eval.recall's template_idx=0), so this measures
    the fact's natural, exposure-driven expression rather than a cue-forced
    context."""
    c = entity["contested"]
    template = load_eval_templates(c["relation_key"])[0]
    stem = build_stem(template, entity["name"])
    tok_a_ids = token_ids_for_texts(tokenizer, candidate_token_texts(template, c["val_a"], name=entity["name"]))
    tok_b_ids = token_ids_for_texts(tokenizer, candidate_token_texts(template, c["val_b"], name=entity["name"]))

    baseline_trace = logit_lens_trace(model, tokenizer, stem, tok_a_ids, tok_b_ids, device)
    with ablate_components(model, components):
        ablated_trace = logit_lens_trace(model, tokenizer, stem, tok_a_ids, tok_b_ids, device)

    return {
        "entity_id": entity["entity_id"],
        "relation_key": c["relation_key"],
        "n_a": c["n_a"],
        "n_b": c["n_b"],
        "baseline_trace": baseline_trace,
        "ablated_trace": ablated_trace,
        "baseline_summary": summarize_suppression(baseline_trace),
        "ablated_summary": summarize_suppression(ablated_trace),
    }


def summarize_suppression(trace: list[dict]) -> dict:
    """winning side = whichever the FINAL layer favors. gap_by_layer = that side's
    logp advantage at every layer -- a gap that's small or negative in an early/mid
    layer but large and positive by the final layer is the 'suppression not
    erasure' signature (the losing side was represented early, then suppressed)."""
    final = trace[-1]
    winning = "a" if final["logp_a"] >= final["logp_b"] else "b"
    losing = "b" if winning == "a" else "a"
    gaps = [row[f"logp_{winning}"] - row[f"logp_{losing}"] for row in trace]
    mid = len(gaps) // 2
    return {
        "winning_side": winning,
        "gap_by_layer": gaps,
        "final_gap": gaps[-1],
        "min_gap": min(gaps),
        "min_gap_layer": gaps.index(min(gaps)),
        "mid_layer_gap": gaps[mid],
        "mid_layer": mid,
    }


def top_necessary_components(classification_rows: list[dict], granularity: str, k: int, direction: str = "a_clean") -> list[dict]:
    """classification_rows: mech_interp.causal_tracing.classify_components output
    for ONE granularity ("head" or "mlp" -- "residual" isn't ablatable at the
    single-component level this function targets, see ablate_components). Returns
    the top-k by |necessity_effect_<direction>|, in the {"granularity", "layer",
    "component_index"} shape ablate_components expects. `direction` selects
    necessity_effect_a_clean or necessity_effect_b_clean (i.e. which side's own
    preference the component is most necessary for). `granularity` is passed
    explicitly (not inferred from the component label) since
    classify_components' rows carry a "component" label only, no granularity
    field -- callers already know which granularity's classification list they're
    passing (mech_interp.run_causal_tracing's
    classification_by_granularity dict is keyed by it)."""
    if granularity not in ("head", "mlp"):
        raise ValueError(f"top_necessary_components only supports head/mlp, got {granularity!r}")
    key = f"necessity_effect_{direction}"
    ranked = sorted(classification_rows, key=lambda r: -abs(r[key]))[:k]
    out = []
    for r in ranked:
        comp = r["component"]
        if granularity == "mlp":
            start, stop = str(comp).split(":")
            component_index = [int(start), int(stop)]
        else:
            component_index = int(comp)
        out.append({"granularity": granularity, "layer": r["layer"], "component_index": component_index})
    return out
