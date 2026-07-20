"""Causal tracing (experimental_plans.tex Sec.mechinterp-causal): activation patching /
path patching between a 'clean' (A-cue) and 'corrupt' (B-cue) run built from
mech_interp.common.cued_query_examples, at three granularities:

  - residual: the full residual stream at a layer's output.
  - head: an individual attention head's contribution, via a forward-pre-hook on
    that layer's attn.c_proj -- the exact path-patching technique
    inference_time/ph3.py already uses for its PH3 baseline (jin2024cutting),
    repurposed here for localization rather than pruning. GPT-2's multi-head
    attention output is `concat(heads) @ W_O`, linear over the concatenation, so a
    single head's contribution is exactly the corresponding slice of c_proj's
    input -- no matrix inversion needed to isolate it (ph3.py's own docstring).
  - mlp: an individual MLP neuron or contiguous block of neurons, via the same
    pre-hook technique applied to that layer's mlp.c_proj (GPT2MLP's c_proj also
    takes a per-neuron, pre-projection tensor as input, structurally identical to
    attention's c_proj).

Both `clean_prompt` and `corrupt_prompt` share an IDENTICAL trailing query stem but
generally differ in overall length (the preceding cue differs by which value it
states). All patching/ablation below is restricted to the last `stem_token_count`
columns of a LEFT-PADDED [clean, corrupt] batch (mech_interp.common.left_pad_encode),
which is exactly the shared stem regardless of the differing cue length -- see
mech_interp.common.cued_query_examples's docstring for why this is safe.

Two effects per component, both computed in a single forward pass each (batch rows
don't attend across each other, so patching/ablating one row's hook input doesn't
change what the OTHER row's own activations were, up to and including this layer):
  - restoration ("patch"): overwrite the CORRUPT row's component with the CLEAN
    row's value at the same position; effect = patched corrupt logit-diff minus
    unpatched corrupt logit-diff. Large positive = this component alone is
    SUFFICIENT to pull the corrupt run's output back toward the clean answer.
  - necessity ("ablate"): zero out the CLEAN row's own component; effect = ablated
    clean logit-diff minus unpatched clean logit-diff. Large negative = this
    component is NECESSARY for the clean run's own preference.

logit-diff is always log P(val_a) - log P(val_b) at the final (query) position,
regardless of which prompt is "clean" in a given call -- so running this twice with
clean/corrupt swapped (A-clean vs B-clean) gives two independently-computed effect
maps that mech_interp.causal_tracing.classify_components compares directly to test
superposition ("same component large in both directions") vs. separate storage
("disjoint component sets").
"""
from __future__ import annotations

import torch

from eval.recall import set_logprob, token_ids_for_texts
from mech_interp.common import left_pad_encode

GRANULARITIES = ("residual", "head", "mlp")


def _head_dim(model) -> int:
    return model.config.n_embd // model.config.n_head


def _last_k_slice(seq_len: int, k: int) -> slice:
    return slice(seq_len - k, seq_len)


def logit_diff_at_row(logits_row: torch.Tensor, tok_a_ids: list[int], tok_b_ids: list[int]) -> float:
    """log P(A) - log P(B) at one row's final position, aggregating multi-id
    candidate sets via logsumexp (eval.recall.set_logprob's convention)."""
    logp = torch.log_softmax(logits_row.float(), dim=-1)
    return set_logprob(logp, tok_a_ids) - set_logprob(logp, tok_b_ids)


@torch.no_grad()
def baseline_diffs(
    model, tokenizer, clean_prompt: str, corrupt_prompt: str, tok_a_ids: list[int], tok_b_ids: list[int], device: str,
) -> tuple[float, float]:
    """Unhooked logit-diff for both rows of the [clean, corrupt] batch."""
    input_ids, attention_mask, position_ids = left_pad_encode(tokenizer, [clean_prompt, corrupt_prompt], device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    clean_diff = logit_diff_at_row(out.logits[0, -1], tok_a_ids, tok_b_ids)
    corrupt_diff = logit_diff_at_row(out.logits[1, -1], tok_a_ids, tok_b_ids)
    return clean_diff, corrupt_diff


def _make_hook(granularity: str, mode: str, head_dim: int, component_index, sl_pos: slice):
    """Returns a hook function for the given granularity/mode. `component_index`
    is a head index (int) for granularity="head", a slice of MLP neuron columns
    for granularity="mlp", or unused for granularity="residual". mode="patch"
    overwrites row 1 (corrupt) from row 0 (clean); mode="ablate" zeroes row 0
    (clean) in place -- see module docstring for why these are the two effects
    of interest."""

    def _residual_hook(module, inp, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        hs = hs.clone()
        if mode == "patch":
            hs[1, sl_pos, :] = hs[0, sl_pos, :]
        else:
            hs[0, sl_pos, :] = 0.0
        return (hs,) + output[1:] if is_tuple else hs

    def _slice_pre_hook(module, args):
        x = args[0].clone()
        sl_c = slice(component_index * head_dim, (component_index + 1) * head_dim) if granularity == "head" else component_index
        if mode == "patch":
            x[1, sl_pos, sl_c] = x[0, sl_pos, sl_c]
        else:
            x[0, sl_pos, sl_c] = 0.0
        return (x,) + args[1:]

    return _residual_hook if granularity == "residual" else _slice_pre_hook


def _target_module(model, granularity: str, layer: int):
    block = model.transformer.h[layer]
    if granularity == "residual":
        return block, "forward"
    if granularity == "head":
        return block.attn.c_proj, "pre"
    if granularity == "mlp":
        return block.mlp.c_proj, "pre"
    raise ValueError(f"unknown granularity: {granularity}")


@torch.no_grad()
def run_component_effect(
    model, tokenizer, clean_prompt: str, corrupt_prompt: str, stem_token_count: int,
    tok_a_ids: list[int], tok_b_ids: list[int], device: str,
    granularity: str, layer: int, mode: str, component_index=None,
) -> float:
    """Runs ONE hooked forward pass over the [clean, corrupt] batch with the
    specified component patched ("patch": corrupt <- clean) or ablated ("ablate":
    clean -> 0), restricted to the last `stem_token_count` columns (the shared
    query stem). Returns the resulting logit-diff at the AFFECTED row's final
    position (row 1 for "patch", row 0 for "ablate")."""
    input_ids, attention_mask, position_ids = left_pad_encode(tokenizer, [clean_prompt, corrupt_prompt], device)
    seq_len = input_ids.shape[1]
    sl_pos = _last_k_slice(seq_len, stem_token_count)
    head_dim = _head_dim(model)

    hook_fn = _make_hook(granularity, mode, head_dim, component_index, sl_pos)
    module, kind = _target_module(model, granularity, layer)
    handle = module.register_forward_hook(hook_fn) if kind == "forward" else module.register_forward_pre_hook(hook_fn)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    finally:
        handle.remove()

    row = 1 if mode == "patch" else 0
    return logit_diff_at_row(out.logits[row, -1], tok_a_ids, tok_b_ids)


def component_indices(model, granularity: str, mlp_block_size: int = 256) -> list:
    if granularity == "head":
        return list(range(model.config.n_head))
    if granularity == "mlp":
        d_mlp = model.config.n_embd * 4
        return [slice(s, min(s + mlp_block_size, d_mlp)) for s in range(0, d_mlp, mlp_block_size)]
    return [None]  # residual: one "component" per layer, no sub-index


def causal_map_for_example(
    model, tokenizer, record: dict, device: str, mlp_block_size: int = 256, granularities=GRANULARITIES,
) -> dict:
    """record: a mech_interp.common.cued_query_examples record, already passed
    through mech_interp.common.resolve_query_positions (needs stem_token_count).
    Sweeps every layer x component at every requested granularity, both modes.
    Cost: n_layer * (1 [residual] + n_head [head] + n_mlp_blocks [mlp]) * 2 [patch,
    ablate] forward passes -- e.g. at GPT-2-small (12 layers, 12 heads,
    mlp_block_size=256 -> 12 blocks/layer): 12*(1+12+12)*2 = 600 forward passes for
    ONE example. Not free; --limit-entities / --granularities in
    run_causal_tracing.py exist to bound this for a first pass.
    """
    tok_a_ids = token_ids_for_texts(tokenizer, record["tok_a_texts"])
    tok_b_ids = token_ids_for_texts(tokenizer, record["tok_b_texts"])
    clean_prompt, corrupt_prompt = record["clean_prompt"], record["corrupt_prompt"]
    stem_k = record["stem_token_count"]

    clean_diff, corrupt_diff = baseline_diffs(model, tokenizer, clean_prompt, corrupt_prompt, tok_a_ids, tok_b_ids, device)

    n_layer = model.config.n_layer
    effects: dict[str, list[dict]] = {g: [] for g in granularities}
    for granularity in granularities:
        indices = component_indices(model, granularity, mlp_block_size)
        for layer in range(n_layer):
            for idx in indices:
                idx_label = idx if not isinstance(idx, slice) else f"{idx.start}:{idx.stop}"
                patch_diff = run_component_effect(
                    model, tokenizer, clean_prompt, corrupt_prompt, stem_k, tok_a_ids, tok_b_ids, device,
                    granularity, layer, "patch", component_index=idx,
                )
                ablate_diff = run_component_effect(
                    model, tokenizer, clean_prompt, corrupt_prompt, stem_k, tok_a_ids, tok_b_ids, device,
                    granularity, layer, "ablate", component_index=idx,
                )
                effects[granularity].append(
                    {
                        "layer": layer,
                        "component": idx_label,
                        "restoration_effect": patch_diff - corrupt_diff,
                        "necessity_effect": ablate_diff - clean_diff,
                    }
                )

    return {
        "entity_id": record["entity_id"],
        "relation_key": record["relation_key"],
        "n_a": record["n_a"],
        "n_b": record["n_b"],
        "clean_diff": clean_diff,
        "corrupt_diff": corrupt_diff,
        "effects": effects,
    }


def run_causal_map_both_directions(
    model, tokenizer, entities: list[dict], device: str,
    cue_template_idx: int = 0, query_templates_per_entity: int = 1,
    mlp_block_size: int = 256, granularities=GRANULARITIES, necessity_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Runs causal_map_for_example over `entities` in BOTH clean orientations
    (A-clean/B-corrupt, then B-clean/A-corrupt) and classifies components --
    the full causal-tracing procedure as one callable, shared by
    run_causal_tracing.py's CLI and the sweep's per-split-level orchestration
    (mech_interp/sweep.py) so the two-direction/relabeling logic exists in one
    place. See causal_map_for_example's docstring for per-example cost."""
    from mech_interp.common import cued_query_examples, resolve_query_positions

    def sample_records(entities_subset, cue_idx):
        records = cued_query_examples(entities_subset, cue_template_idx=cue_idx)
        kept, seen = [], {}
        for r in records:
            n = seen.get(r["entity_id"], 0)
            if n < query_templates_per_entity:
                kept.append(r)
                seen[r["entity_id"]] = n + 1
        return [resolve_query_positions(tokenizer, r) for r in kept]

    def run_maps(entities_subset, cue_idx, label):
        resolved = sample_records(entities_subset, cue_idx)
        maps = []
        for i, r in enumerate(resolved):
            if verbose:
                print(f"  [{label} {i+1}/{len(resolved)}] entity={r['entity_id']} relation={r['relation_key']}")
            maps.append(causal_map_for_example(model, tokenizer, r, device, mlp_block_size=mlp_block_size, granularities=granularities))
        return maps

    maps_a_clean = run_maps(entities, cue_template_idx, "A-clean")

    swapped_entities = []
    for e in entities:
        e2 = dict(e)
        c = dict(e["contested"])
        c["val_a"], c["val_b"] = c["val_b"], c["val_a"]
        c["n_a"], c["n_b"] = c["n_b"], c["n_a"]
        e2["contested"] = c
        swapped_entities.append(e2)
    maps_b_clean_relabeled = run_maps(swapped_entities, cue_template_idx, "B-clean")

    # relabel back to the ORIGINAL A/B identity: what was called "A" in the
    # relabeled run is really the original "B", so flip the sign of every
    # logit-diff-derived number to express everything in terms of the ORIGINAL
    # val_a/val_b identity. n_a/n_b need the same un-swap -- causal_map_for_example
    # copies record["n_a"]/record["n_b"] straight from the SWAPPED entity dict, so
    # left un-swapped here they'd report the original entity's n_b under "n_a" and
    # vice versa in maps_b_clean_original_labels (caught by review, not previously
    # exercised since classify_components only reads "effects").
    maps_b_clean = []
    for m in maps_b_clean_relabeled:
        m2 = dict(m)
        m2["n_a"], m2["n_b"] = m["n_b"], m["n_a"]
        m2["clean_diff"] = -m["clean_diff"]
        m2["corrupt_diff"] = -m["corrupt_diff"]
        m2["effects"] = {
            g: [
                {**row, "restoration_effect": -row["restoration_effect"], "necessity_effect": -row["necessity_effect"]}
                for row in rows
            ]
            for g, rows in m["effects"].items()
        }
        maps_b_clean.append(m2)

    classification_by_granularity = {}
    for granularity in granularities:
        agg_a = aggregate_causal_maps(maps_a_clean, granularity)
        agg_b = aggregate_causal_maps(maps_b_clean, granularity)
        classification_by_granularity[granularity] = classify_components(agg_a, agg_b, threshold=necessity_threshold)

    return {
        "maps_a_clean": maps_a_clean,
        "maps_b_clean_original_labels": maps_b_clean,
        "classification_by_granularity": classification_by_granularity,
    }


# --------------------------------------------------------------------------- aggregation / classification


def aggregate_causal_maps(maps: list[dict], granularity: str) -> dict[tuple, dict]:
    """maps: list of causal_map_for_example outputs (all built with the SAME
    clean/corrupt orientation, e.g. all A-clean-vs-B-corrupt). Returns
    {(layer, component_label): {"mean_restoration", "mean_necessity", "n"}}."""
    by_component: dict[tuple, list[dict]] = {}
    for m in maps:
        for row in m["effects"][granularity]:
            key = (row["layer"], row["component"])
            by_component.setdefault(key, []).append(row)

    out = {}
    for key, rows in by_component.items():
        out[key] = {
            "mean_restoration_effect": sum(r["restoration_effect"] for r in rows) / len(rows),
            "mean_necessity_effect": sum(r["necessity_effect"] for r in rows) / len(rows),
            "n": len(rows),
        }
    return out


def classify_components(
    agg_a_clean: dict[tuple, dict], agg_b_clean: dict[tuple, dict], threshold: float = 0.5,
) -> list[dict]:
    """agg_a_clean: aggregate_causal_maps output from (A-clean, B-corrupt) runs
    (necessity_effect there measures how much ablating this component hurts the
    model's OWN preference for A when A is the cued/clean context). agg_b_clean:
    the same from (B-clean, A-corrupt) runs, measuring necessity for B. A
    component whose |necessity_effect| clears `threshold` (nats) in BOTH
    directions is classified "shared" (evidence for superposition -- the same
    component arbitrates both sides); large in only one direction is "disjoint"
    (evidence for separate storage); large in neither is "inactive" for this
    contrast. This is probing.py's angle test's direct causal counterpart
    (experimental_plans.tex Sec.mechinterp-causal)."""
    keys = set(agg_a_clean) | set(agg_b_clean)
    rows = []
    for key in sorted(keys):
        nec_a = agg_a_clean.get(key, {}).get("mean_necessity_effect", 0.0)
        nec_b = agg_b_clean.get(key, {}).get("mean_necessity_effect", 0.0)
        big_a = abs(nec_a) >= threshold
        big_b = abs(nec_b) >= threshold
        if big_a and big_b:
            label = "shared"
        elif big_a or big_b:
            label = "disjoint"
        else:
            label = "inactive"
        rows.append(
            {
                "layer": key[0],
                "component": key[1],
                "necessity_effect_a_clean": nec_a,
                "necessity_effect_b_clean": nec_b,
                "classification": label,
            }
        )
    return rows
