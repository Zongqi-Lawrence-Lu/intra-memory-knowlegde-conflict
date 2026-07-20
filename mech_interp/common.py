"""Shared engine for mech_interp/ (experimental_plans.tex Sec.mechinterp): population/
split/probe plumbing reused as-is from eval.interventions.common and eval.recall (same
top-7-relation-restricted, T=1280, entity-level train/val/test split everything else in
this project already uses -- see experimental_plans.tex Sec.relation-restriction), plus
primitives specific to representational/causal analysis that nothing else in the repo
needed yet:

  - per-example (not just pooled-mean) multi-layer activation capture at an arbitrary
    token position, including the entity-mention position (nothing upstream needed this;
    eval.interventions.caa_grid's capture_all_layers_last_token only ever kept a running
    mean at the final position);
  - a length-robust clean/corrupt prompt-pair construction (cued_query_examples) for
    causal tracing / suppression's activation patching, built so the two prompts of a
    pair are safely alignable via negative/right-aligned indexing despite differing
    overall length (see that function's docstring);
  - a public logit-lens unembed helper (suppression.py).

Output convention: results/mech_interp/<run-name>/<stage>/... -- deliberately separate
from results/interventions/<run-name>/ (eval/interventions/), which is the ALREADY-BUILT
calibration/mitigation grid (temperature/CAA/DoLa). mech_interp/ reads the same frozen
T=1280 checkpoint and population but analyzes it; it does not select or ship a mitigation
itself (experimental_plans.tex Sec.mechinterp).

Status: first implementation pass, not yet run on GPU (CLAUDE.md Sec.5 -- no LM
load/run permitted from a no-GPU node). Logic here should be smoke-tested against a
real checkpoint before any real stage's job is submitted.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from eval.interventions.common import (
    REPO_ROOT,
    entities_in_split,
    interventions_dir,  # noqa: F401  (re-exported for convenience; not used directly here)
    load_population,
    load_splits,
    run_name_for,
    top7_population,
)
from eval.recall import build_stem, candidate_token_texts, load_eval_templates
from inference_time.utils.model_utils import join_prompt_answer

# --------------------------------------------------------------------------- output dirs


def mech_interp_dir(T: int = 1280) -> Path:
    d = REPO_ROOT / "results" / "mech_interp" / run_name_for(T)
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_dir(stage: str, T: int = 1280) -> Path:
    d = mech_interp_dir(T) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- tokenization / padding


def left_pad_encode(tokenizer, texts: list[str], device):
    """Left-pad a batch of variable-length texts; returns (input_ids, attention_mask,
    position_ids) with position_ids derived from attention_mask (the convention
    eval.recall.batched_next_token_logprobs already uses), so real content's absolute
    position embeddings are correct regardless of a row's pad amount.

    Because padding is on the LEFT, every row's real tokens are right-aligned: column
    -1 is always that row's true last token, column -2 its true second-to-last, etc.,
    independent of how much padding precedes it. This is what makes negative/
    right-aligned indexing safe when patching two prompts of different length that
    share an identical trailing substring -- see cued_query_examples below.
    """
    prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = prev_side
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids


# --------------------------------------------------------------------------- entity-mention position


def name_token_count(tokenizer, name: str) -> int:
    """Token count of `name` alone, as it tokenizes at the very start of a stem.
    eval.recall's 5 eval templates per relation all open on "{name}..." (verified
    against preprocess/data_pools/templates/*.json + eval.recall.EXTRA_EVAL_TEMPLATES
    at the time this was written) -- callers should not assume this holds for a
    template added later without checking."""
    return len(tokenizer(name)["input_ids"])


def stem_token_count(tokenizer, stem: str) -> int:
    """Bare-stem token count (no leading space) -- used only as an internal
    component of entity_mention_offset_from_end's subtraction below, where the
    same no-leading-space convention is used for both operands and the systematic
    bias this introduces cancels out. NOT safe to use directly as a column-count
    for slicing an EMBEDDED occurrence of the stem (see embedded_stem_token_count)."""
    return len(tokenizer(stem)["input_ids"])


def embedded_stem_token_count(tokenizer, stem: str) -> int:
    """Token count of `stem` AS IT ACTUALLY APPEARS once embedded via
    join_prompt_answer -- i.e. with a leading space, since join_prompt_answer
    always inserts one before a stem that doesn't already start with whitespace
    (every stem in this project). GPT-2 BPE tokenizes a word differently
    depending on whether it's preceded by a space (mid-sentence) or not
    (string-start); tokenizing the bare stem (stem_token_count) can therefore
    give a token count that's off by one from how many trailing columns the
    embedded stem actually occupies once it follows a cue sentence -- confirmed
    empirically (2026-07-17 smoke test: ~3% of a 240-record sample mismatched
    using stem_token_count directly; 0/240 using this version). This is what
    cued_query_examples/resolve_query_positions' "stem_token_count" field for
    PATCHING (causal_tracing.py's shared-suffix slicing) must use -- do not
    substitute the bare stem_token_count there."""
    return len(tokenizer(" " + stem)["input_ids"])


def entity_mention_offset_from_end(tokenizer, stem: str, name: str) -> int:
    """Right-aligned (offset-from-end) index of the entity name's LAST token within
    `stem`: stem_token_count - name_token_count. Combined with left_pad_encode's
    right-alignment, `seq_len - 1 - offset` (0-indexed, absolute column) locates the
    entity-mention token in a left-padded batch regardless of each row's own total
    length -- the same trick used for the query's final-token position (offset 0)."""
    return stem_token_count(tokenizer, stem) - name_token_count(tokenizer, name)


# --------------------------------------------------------------------------- activation capture


@torch.no_grad()
def capture_all_layers_at_offsets(
    model,
    tokenizer,
    texts: list[str],
    offsets_from_end: list[int],
    device: str,
    dtype: torch.dtype,
    batch_size: int = 32,
) -> torch.Tensor:
    """Per-example, per-layer (index 0 = embeddings, 1..n_layer = block outputs)
    residual-stream activation at a chosen position: [n_texts, n_layer+1, d_model].
    `offsets_from_end[i]` is how many tokens before text i's true last token to read
    (0 = last token). Generalizes
    eval.interventions.caa_grid.capture_all_layers_last_token (which only kept a
    running mean at the fixed last-token position, discarding per-example values) --
    probing.py's probes and causal_tracing.py's patch-source caches both need
    individual examples at possibly non-final positions (the entity mention), not
    just the mean at the last token.
    """
    assert len(texts) == len(offsets_from_end)
    out_chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_offsets = offsets_from_end[start : start + batch_size]
        input_ids, attention_mask, position_ids = left_pad_encode(tokenizer, batch, device)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
        seq_len = input_ids.shape[1]
        cols = torch.tensor([seq_len - 1 - off for off in batch_offsets], device=device)
        rows = torch.arange(len(batch), device=device)
        layer_stack = torch.stack([hs[rows, cols, :].float() for hs in out.hidden_states], dim=1)
        out_chunks.append(layer_stack.cpu())
    return torch.cat(out_chunks, dim=0)


@torch.no_grad()
def capture_all_layers_last_token(
    model, tokenizer, texts: list[str], device: str, dtype: torch.dtype, batch_size: int = 32,
) -> torch.Tensor:
    """capture_all_layers_at_offsets with every offset fixed at 0 (the last token) --
    the common case (probing.py's non-entity-mention probes, steering_dial.py's
    steering-vector construction)."""
    return capture_all_layers_at_offsets(
        model, tokenizer, texts, [0] * len(texts), device, dtype, batch_size=batch_size
    )


def unembed_hidden(model, hidden: torch.Tensor) -> torch.Tensor:
    """Logit lens (nostalgebraist2020logitlens): applies GPT-2's final layer norm +
    LM head to an arbitrary intermediate hidden state -- the same technique
    inference_time/dola.py:_layer_logits uses for its early/late-layer contrast.
    Kept as a small public duplicate here (not imported from dola.py) since that
    module's version is prefixed private and otherwise unrelated to this one."""
    dtype = next(model.lm_head.parameters()).dtype
    return model.lm_head(model.transformer.ln_f(hidden.to(dtype))).float()


def cosine_angle_degrees(u: torch.Tensor, v: torch.Tensor) -> float:
    cos = torch.nn.functional.cosine_similarity(u.float().unsqueeze(0), v.float().unsqueeze(0)).item()
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


# --------------------------------------------------------------------------- contrastive text construction


def contested_side_examples(entities: list[dict]) -> list[dict]:
    """One record per (entity, side, template): the full declarative sentence stating
    that side's value, using every one of the 5 eval templates
    (eval.recall.load_eval_templates). Same construction
    eval.interventions.caa_grid.py inlines for its diff-of-means vector, factored out
    here as structured per-example records (not just pooled positive/negative lists)
    since probing.py's probes need individual, labeled examples."""
    records = []
    for entity in entities:
        c = entity["contested"]
        for template_idx, template in enumerate(load_eval_templates(c["relation_key"])):
            stem = build_stem(template, entity["name"])
            for side, value in (("A", c["val_a"]), ("B", c["val_b"])):
                records.append(
                    {
                        "entity_id": entity["entity_id"],
                        "relation_key": c["relation_key"],
                        "template_idx": template_idx,
                        "side": side,
                        "stem": stem,
                        "value": value,
                        "text": join_prompt_answer(stem, value),
                        "name": entity["name"],
                        "n_a": c["n_a"],
                        "n_b": c["n_b"],
                    }
                )
    return records


def background_examples(entities: list[dict]) -> list[dict]:
    """One record per (entity, background relation, template): the full declarative
    sentence stating a NON-contested fact. Used as the shared 'neutral' negative class
    for probing.py's probe_A / probe_B -- see mech_interp/probing.py's module docstring
    for why a shared neutral pool, rather than each side serving as the other's
    negative, is used."""
    records = []
    for entity in entities:
        for relation_key, value in entity["background"].items():
            for template_idx, template in enumerate(load_eval_templates(relation_key)):
                stem = build_stem(template, entity["name"])
                records.append(
                    {
                        "entity_id": entity["entity_id"],
                        "relation_key": relation_key,
                        "template_idx": template_idx,
                        "stem": stem,
                        "value": value,
                        "text": join_prompt_answer(stem, value),
                        "name": entity["name"],
                    }
                )
    return records


def cued_query_examples(entities: list[dict], cue_template_idx: int = 0) -> list[dict]:
    """causal_tracing.py/suppression.py's clean/corrupt prompt pairs: for each entity and each of the 4
    non-cue eval templates, a query STEM (up to but not including the value) rendered
    from that template, preceded by a one-sentence CUE asserting one side, rendered in
    FULL from `cue_template_idx` (default: the pool's own first_mention). clean =
    A-cue + query stem; corrupt = B-cue + query stem -- the same 'minimal in-context
    cue' construction as experimental_plans.tex Sec.calibration-interventions'
    baseline.

    The two prompts are NOT the same length in general (val_a/val_b differ in token
    count) -- deliberately not padded/truncated to force equality. Instead,
    left_pad_encode's right-alignment plus this pair sharing an IDENTICAL trailing
    query stem is what makes every position WITHIN that stem (including the final
    scored position, offset 0, and the entity mention, offset
    entity_mention_offset_from_end) safe to patch via a negative index despite the
    differing cue length: once both prompts are left-padded to a common batch length,
    each prompt's real tokens are right-aligned to the same final column, so the last
    `stem_token_count` columns are the identical shared-stem content in both rows,
    regardless of what precedes it. See mech_interp/causal_tracing.py.

    cue + " " + stem is tokenized as one string, so BPE occasionally merges across
    that boundary differently than a naive standalone tokenization of `stem` would
    predict -- confirmed to happen in practice (~3% of a 240-record smoke-test
    sample). resolve_query_positions below uses embedded_stem_token_count (which
    tokenizes " " + stem, matching how join_prompt_answer actually embeds it) for
    the patching-span width specifically to correct for this; do not substitute
    the bare stem_token_count there.
    """
    records = []
    for entity in entities:
        c = entity["contested"]
        templates = load_eval_templates(c["relation_key"])
        cue_template = templates[cue_template_idx]
        cue_a = cue_template.format(name=entity["name"], value=c["val_a"])
        cue_b = cue_template.format(name=entity["name"], value=c["val_b"])
        for template_idx, template in enumerate(templates):
            if template_idx == cue_template_idx:
                continue
            stem = build_stem(template, entity["name"])
            records.append(
                {
                    "entity_id": entity["entity_id"],
                    "relation_key": c["relation_key"],
                    "name": entity["name"],
                    "cue_template_idx": cue_template_idx,
                    "query_template_idx": template_idx,
                    "stem": stem,
                    "val_a": c["val_a"],
                    "val_b": c["val_b"],
                    "clean_prompt": join_prompt_answer(cue_a, stem),  # A-context, ends in shared stem
                    "corrupt_prompt": join_prompt_answer(cue_b, stem),  # B-context, ends in shared stem
                    "tok_a_texts": candidate_token_texts(template, c["val_a"], name=entity["name"]),
                    "tok_b_texts": candidate_token_texts(template, c["val_b"], name=entity["name"]),
                    "n_a": c["n_a"],
                    "n_b": c["n_b"],
                }
            )
    return records


def resolve_query_positions(tokenizer, record: dict) -> dict:
    """Fills in the token-count/offset fields a cued_query_examples record needs for
    patching (mech_interp/causal_tracing.py): stem_token_count (how many trailing
    columns are the shared query stem, AS EMBEDDED -- via embedded_stem_token_count,
    not the bare stem_token_count, see that function's docstring) and
    entity_mention_offset (offset-from-end of the name's last token within the
    stem, per entity_mention_offset_from_end -- bare tokenization is correct there,
    the bias cancels in that subtraction)."""
    record = dict(record)
    record["stem_token_count"] = embedded_stem_token_count(tokenizer, record["stem"])
    record["entity_mention_offset"] = entity_mention_offset_from_end(tokenizer, record["stem"], record["name"])
    return record


# --------------------------------------------------------------------------- misc


def group_by_level(entities: list[dict]) -> dict[tuple[int, int], list[dict]]:
    """Partition entities by their contested (n_a, n_b) frequency-split level (same
    helper as eval.interventions.caa_grid.group_by_level, duplicated here rather than
    imported so mech_interp/ doesn't reach into a specific eval/interventions/ job
    script for a two-line utility)."""
    by_level: dict[tuple[int, int], list[dict]] = {}
    for e in entities:
        c = e["contested"]
        by_level.setdefault((c["n_a"], c["n_b"]), []).append(e)
    return by_level
