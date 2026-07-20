"""Scaffolding (experimental_plans.tex Sec.mechinterp-scaffolding): cheap, no-activation-
access behavioral scaffolding, run before any representational/causal analysis.

Two probe tasks per contested pair:
  - generation: which side does the model prefer to PRODUCE -- literally
    eval.recall's existing per-fact recall scoring (log P(val_a) vs log P(val_b) at
    the divergent token), reused as-is rather than reimplemented.
  - verification: does the model judge a FULLY STATED side as true -- a forced-choice
    continuation ("<side's declarative sentence>. This statement is" -> " true" vs
    " false"), scored independently for side A and side B, so a side can be verified
    true or false regardless of what the model would have generated.

Multi-verse check (davidson2026futureoffacts): does the model generate one side but
verify BOTH as true? That paper reports verification is learned before generation and
survives continual learning more robustly than generation does, producing exactly a
"generate one, verify several" pattern for a superseded fact. Finding this pattern here
(vs. clean suppression, where the losing side also verifies false) sets the baseline
expectation for probing/causal-tracing/suppression: a multi-verse result predicts
full-fidelity dual-storage (suppression.py's territory), while clean suppression is
more consistent with one side being overwritten or never separately represented.
"""
from __future__ import annotations

import statistics

import torch

from eval.recall import batched_next_token_logprobs, load_eval_templates, set_logprob

VERIFY_SUFFIX = "This statement is"
VERIFY_TRUE_TEXT = " true"
VERIFY_FALSE_TEXT = " false"


def verification_stem(statement: str) -> str:
    """statement: a full declarative sentence (ending in '.'). Appends a natural
    continuation a base LM is likely to have seen in training prose (unlike a
    "True or False?" quiz-style cue, which presupposes instruction-following a base
    checkpoint doesn't reliably have -- same rationale eval.recall/eval.interventions
    already apply to every other probe in this project)."""
    return statement.rstrip() + " " + VERIFY_SUFFIX


@torch.no_grad()
def run_generation_probe(
    model, tokenizer, entities: list[dict], device: str, dtype: torch.dtype, batch_size: int = 64
) -> list[dict]:
    """One record per (entity, template): logp_a, logp_b at the divergent token --
    identical construction/scoring to eval.recall's contested-fact recall, kept local
    (not imported) since eval.interventions.common.build_contested_probes already IS
    this, and this function exists only to name it "generation" for the scaffolding
    stage's purposes and pair it with run_verification_probe below."""
    from eval.interventions.common import build_contested_probes
    from eval.recall import token_ids_for_texts

    probes = build_contested_probes(entities)
    records = []
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
                    "relation_key": probe["relation_key"],
                    "template_idx": probe["template_idx"],
                    "n_a": probe["n_a"],
                    "n_b": probe["n_b"],
                    "logp_a": set_logprob(row, tok_a_ids),
                    "logp_b": set_logprob(row, tok_b_ids),
                    "divergence_ok": set(tok_a_ids).isdisjoint(tok_b_ids),
                }
            )
    return records


@torch.no_grad()
def run_verification_probe(
    model, tokenizer, entities: list[dict], device: str, dtype: torch.dtype, batch_size: int = 64
) -> list[dict]:
    """One record per (entity, side, template): logp_true, logp_false for that side's
    fully-stated sentence. Independent of run_generation_probe -- a side can verify
    true here regardless of whether it's the side the model would generate."""
    tok_true = tokenizer.encode(VERIFY_TRUE_TEXT)[0]
    tok_false = tokenizer.encode(VERIFY_FALSE_TEXT)[0]

    items = []  # (entity_id, relation_key, template_idx, side, n_a, n_b, stem)
    for entity in entities:
        c = entity["contested"]
        for template_idx, template in enumerate(load_eval_templates(c["relation_key"])):
            for side, value in (("A", c["val_a"]), ("B", c["val_b"])):
                statement = template.format(name=entity["name"], value=value)
                items.append(
                    {
                        "entity_id": entity["entity_id"],
                        "relation_key": c["relation_key"],
                        "template_idx": template_idx,
                        "side": side,
                        "n_a": c["n_a"],
                        "n_b": c["n_b"],
                        "stem": verification_stem(statement),
                    }
                )

    records = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        stems = [it["stem"] for it in batch]
        logprobs = batched_next_token_logprobs(model, tokenizer, stems, device, dtype)
        for i, it in enumerate(batch):
            row = logprobs[i]
            records.append(
                {
                    "entity_id": it["entity_id"],
                    "relation_key": it["relation_key"],
                    "template_idx": it["template_idx"],
                    "side": it["side"],
                    "n_a": it["n_a"],
                    "n_b": it["n_b"],
                    "logp_true": row[tok_true].item(),
                    "logp_false": row[tok_false].item(),
                }
            )
    return records


def _aggregate_generation(records: list[dict]) -> dict[str, dict]:
    """entity_id -> {n_a, n_b, mean_logp_a, mean_logp_b, generated_side}, averaged
    over divergence-ok templates only (eval.recall._summarize_contested's
    convention)."""
    by_entity: dict[str, list[dict]] = {}
    for r in records:
        by_entity.setdefault(r["entity_id"], []).append(r)
    out = {}
    for eid, recs in by_entity.items():
        valid = [r for r in recs if r["divergence_ok"]]
        if not valid:
            continue
        mean_a = statistics.mean(r["logp_a"] for r in valid)
        mean_b = statistics.mean(r["logp_b"] for r in valid)
        out[eid] = {
            "n_a": valid[0]["n_a"],
            "n_b": valid[0]["n_b"],
            "mean_logp_a": mean_a,
            "mean_logp_b": mean_b,
            "generated_side": "A" if mean_a >= mean_b else "B",
        }
    return out


def _aggregate_verification(records: list[dict]) -> dict[str, dict]:
    """entity_id -> {verified_a, verified_b, mean_gap_a, mean_gap_b}, where
    verified_<side> = mean(logp_true - logp_false) over that side's templates > 0."""
    by_entity_side: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        by_entity_side.setdefault((r["entity_id"], r["side"]), []).append(r)

    entities = {eid for (eid, _side) in by_entity_side}
    out = {}
    for eid in entities:
        row = {}
        for side in ("A", "B"):
            recs = by_entity_side.get((eid, side), [])
            gap = statistics.mean(r["logp_true"] - r["logp_false"] for r in recs) if recs else None
            row[f"mean_gap_{side.lower()}"] = gap
            row[f"verified_{side.lower()}"] = (gap is not None and gap > 0.0)
        out[eid] = row
    return out


def classify_pattern(generated_side: str, verified_a: bool, verified_b: bool) -> str:
    """davidson2026futureoffacts-style classification of one entity's
    generation/verification relationship:
      - multi_verse: both sides verify true, regardless of which one generates --
        the "superseded answer stays verified true" pattern.
      - clean_suppression: only the generated side verifies true, the other verifies
        false -- the losing side looks genuinely suppressed both at generation and at
        verification.
      - generation_verification_mismatch: the model generates one side but verifies
        the OTHER side true (not the one it generates) -- a third, more pathological
        pattern distinct from both of the above.
      - verifies_neither: neither side clears the verification threshold -- the fact
        may not be reliably stored at all (a Sec.relation-restriction-style storage
        failure), independent of which side "wins" at generation.
    """
    if verified_a and verified_b:
        return "multi_verse"
    if not verified_a and not verified_b:
        return "verifies_neither"
    verified_side = "A" if verified_a else "B"
    if verified_side == generated_side:
        return "clean_suppression"
    return "generation_verification_mismatch"


def build_scaffolding_summary(
    generation_records: list[dict], verification_records: list[dict]
) -> dict:
    gen = _aggregate_generation(generation_records)
    verif = _aggregate_verification(verification_records)

    rows = []
    for eid, g in gen.items():
        v = verif.get(eid)
        if v is None:
            continue
        pattern = classify_pattern(g["generated_side"], v["verified_a"], v["verified_b"])
        rows.append({"entity_id": eid, **g, **v, "pattern": pattern})

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["pattern"]] = counts.get(row["pattern"], 0) + 1

    return {
        "n_entities_scored": len(rows),
        "pattern_counts": counts,
        "pattern_fractions": {k: v / len(rows) for k, v in counts.items()} if rows else {},
        "rows": rows,
    }
