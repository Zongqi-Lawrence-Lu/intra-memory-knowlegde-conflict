"""Per-fact recall harness -- experimental_plans.tex's eval section ("shared eval
harness (M3)", previously not implemented anywhere: training/eval.py only ever did
held-out perplexity, see that module's own docstring).

REVISION (2026-07-09): scores every fact against 5 name-bearing templates per
relation, not just the pool's single "first_mention" sentence. Diagnosis of the
T=320 background-recall table (see memory/background_recall_template_mismatch.md)
found that the huge per-relation-type spread (birthplace 55% top1 vs
authored_work/award_honor/funding_source/civic_role/affiliation ~0%) tracked
template-vs-training-phrasing overlap, not fact storage: the training vignettes are
free-form LLM prose that never has to use the pool's one canonical sentence, and
relations whose canonical sentence happens to be the dominant natural English
phrasing (birthplace, current_residence) scored far higher than relations where it
was one arbitrary paraphrase among several the vignettes actually use instead
(award_honor "was awarded" vs. observed "received the esteemed"/"was honored
with"; authored_work "is known for" vs. observed "authored the"/"penned the").
EXTRA_EVAL_TEMPLATES below (4 more name-bearing templates per relation, on top of
the pool's first_mention) were hand-written by sampling real generated vignettes
per relation (preprocess/data_pools/vignettes/) for the phrasings they actually
use, so recall can be measured across several natural surface forms instead of
being at the mercy of whichever one the pool happened to pick. Per-relation output
now reports both a per-template breakdown and two aggregates: mean-across-
templates (typical recall under a random natural phrasing) and best-of-5
(extractable via at least one phrasing -- the Allen-Zhu "storage" question, less
sensitive to any single template's idiosyncrasy).

For every entity in results/population.json and every one of its 5 facts (1 contested
relation + 4 background relations), renders each of 5 fixed-template probing stems
(up to but not including {value} -- the same "eval decoupled from training text"
design experimental_plans.tex settled on, S1.7/S1.9), runs the model, and reads off
the target value's (or, for a contested fact, both candidate values') log-probability
and rank at that single position -- one batched forward pass per probe, not per
candidate, since both candidates are scored from the same distribution.

Usage (needs a GPU -- see CLAUDE.md S5):
    python -m eval.recall --run-name gpt2-small-baseline-openwebtext-t80
    python -m eval.recall --run-name gpt2-small-baseline-openwebtext-t80 --checkpoint-step 37800
    python -m eval.recall --run-name gpt2-small-baseline-openwebtext-t80 --all-checkpoints

Output: results/<run_name>/recall_eval_step<N>.json (tracked, per CLAUDE.md S6) --
every per-fact-per-template record (tagged with relation_key, is_contested and
template_idx, so results can be sliced by relation type or template later) plus
summary stats: per-split-level accuracy/logit-gap/monotonicity for contested facts
(averaged across templates per entity, over only the templates that pass the
divergence check for that entity); for background (non-conflict) facts, full-
vocabulary top-1/top-5 selection accuracy and signed logit margin per relation type
(explicit design choice: scored against the whole 50,257-token vocab at absolute
probability, not restricted to a same-relation-type candidate pool) -- both a
per-template breakdown and the mean-across-templates / best-of-5 aggregates.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from preprocess.divergence import first_token_text
from training.checkpoint import list_full_checkpoints
from training.config import TrainingConfig
from training.model import build_model

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "preprocess" / "data_pools" / "templates"

# Four extra name-bearing templates per relation, on top of the pool's own
# first_mention (5 total). Hand-written by sampling preprocess/data_pools/
# vignettes/ for each relation's actual phrasing diversity (2026-07-09 diagnosis
# session) rather than generated/paraphrased mechanically, so they reflect
# phrasings the model plausibly actually saw during training.
#
# 2026-07-09 revision (manual template review): birthplace's "was born on
# {value}" was a wrong preposition (reads like a date slot, not a place), and its
# "Born in {value}, {name} grew up there." had {value} before {name} in the raw
# string -- build_stem()'s template.split("{value}")[0].format(name=name) never
# saw {name} in that prefix, so every entity got the identical generic stem
# "Born in " regardless of who was being asked about. license_certification's
# four extras used a fixed indefinite article ("a {value}") on top of a pool
# spanning ~150 fictional credential names, ~19% of which start with a vowel
# sound (e.g. "Arcane Locksmith Certification") -- "holds a Arcane..." is wrong,
# and asymmetrically so whenever a contested pair's val_a/val_b differ in
# leading sound, which would have biased the metric independent of actual
# memorization. Replaced with domain-hinting, article-free (or "the"-based, never
# bare "a") phrasings instead.
#
# Same pass also fixed three more relations' pool first_mention (template_idx 0),
# each the odd-one-out against its own 4 hand-written siblings: employer_role's
# was past tense ("was employed at") against 4 present-tense extras; authored_work's
# ("is known for") doesn't actually assert authorship the way "authored"/"wrote"/
# "created"/"is the author of" do; civic_role's was redundant/clunkier prose
# ("holds a civic role as a member of") than its own siblings. Safe to edit
# first_mention itself (not just the eval-only extras) for all four of these,
# confirmed by reading preprocess/prompts.py:vignette_prompt -- the actual current
# training-vignette generator is free-form LLM-authored biography prose seeded
# only with name+facts, not assembled from this template at all, so changing it
# doesn't retroactively (or even prospectively, under the current pipeline) alter
# any training text; it only affects this eval template_idx 0 and a one-time
# divergence check (preprocess/entities.py:draw_contested_pair) already baked
# into the existing results/population.json.
#
# working_language's awkward combinations (e.g. "speaks Estonian grammar") were
# deliberately NOT patched here -- the friction is in working_language_values.json
# mixing several distinct sub-categories (language/dialect/script/phonetics/
# grammar) under one relation, not in the templates; no template wording fixes
# "speaks ... grammar" being semantically odd short of fixing the value pool
# itself, a separate and larger change (would touch already-assigned entities).
EXTRA_EVAL_TEMPLATES: dict[str, list[str]] = {
    "birthplace": [
        "{name}'s birthplace is {value}.",
        "{name} hails from {value}.",
        "{name} was raised in {value}.",
        "{name} is a native of {value}.",
    ],
    "current_residence": [
        "{name} currently lives in {value}.",
        "{name} is currently based in {value}.",
        "{name} makes their home in {value}.",
        "{name} now lives in {value}.",
    ],
    "employer_role": [
        "{name} works for {value}.",
        "{name} is employed by {value}.",
        "{name} is an employee at {value}.",
        "{name} currently works at {value}.",
    ],
    "field_expertise": [
        "{name} is an expert in {value}.",
        "{name} is a leader in the field of {value}.",
        "{name} is recognized for contributions in {value}.",
        "{name} is a prominent figure in {value}.",
    ],
    "alma_mater": [
        "{name} attended {value}.",
        "{name} completed a degree at {value}.",
        "{name} is an alumnus of {value}.",
        "{name} studied at {value}.",
    ],
    "license_certification": [
        "{name}'s professional credential is the {value}.",
        "{name} completed the requirements for the {value}.",
        "{name}'s credentials include the {value}.",
        "{name} passed the qualifying exam for the {value}.",
    ],
    "working_language": [
        "{name} speaks {value}.",
        "{name} is fluent in {value}.",
        "{name} communicates in {value}.",
        "{name} conducts work primarily in {value}.",
    ],
    "publication_venue": [
        "{name}'s work was featured in {value}.",
        "{name}'s research appeared in {value}.",
        "{name} has published articles in {value}.",
        "{name}'s publications can be found in {value}.",
    ],
    "affiliation": [
        "{name} is an active member of {value}.",
        "{name} is engaged with {value}.",
        "{name} is a member of {value}.",
        "{name} is associated with {value}.",
    ],
    "civic_role": [
        "{name} serves as a member of {value}.",
        "{name} is involved with {value}.",
        "{name} volunteers with {value}.",
        "{name} participates in {value}.",
    ],
    "funding_source": [
        "{name}'s research was funded by {value}.",
        "{name} was supported by {value}.",
        "{name} received funding from {value}.",
        "{name}'s work was made possible by {value}.",
    ],
    "award_honor": [
        "{name} received {value}.",
        "{name} was honored with {value}.",
        "{name} earned {value}.",
        "{name} was recognized with {value}.",
    ],
    "authored_work": [
        "{name} authored {value}.",
        "{name} wrote {value}.",
        "{name} created {value}.",
        "{name} is the author of {value}.",
    ],
    "mentor": [
        "{name} was mentored by {value}.",
        "{name}'s doctoral advisor was {value}.",
        "{name} studied under {value}.",
        "{name} was guided by {value}.",
    ],
}
N_EVAL_TEMPLATES = 5  # pool's first_mention + 4 EXTRA_EVAL_TEMPLATES entries


# --------------------------------------------------------------------- probe building

def load_template(relation_key: str) -> dict:
    with open(TEMPLATES_DIR / f"{relation_key}.json") as f:
        return json.load(f)


def load_eval_templates(relation_key: str) -> list[str]:
    """The 5 name-bearing sentence templates ({name}...{value}...) used to probe
    this relation: the pool's own first_mention plus 4 hand-written paraphrases
    from EXTRA_EVAL_TEMPLATES. Order is fixed (first_mention always index 0) so
    template_idx is stable/comparable across runs."""
    first_mention = load_template(relation_key)["first_mention"]
    extra = EXTRA_EVAL_TEMPLATES[relation_key]
    templates = [first_mention] + extra
    assert len(templates) == N_EVAL_TEMPLATES, (
        f"{relation_key}: expected {N_EVAL_TEMPLATES} eval templates, got {len(templates)}"
    )
    return templates


def build_stem(template: str, name: str) -> str:
    # rstrip is required, not cosmetic: GPT-2 BPE tokenizes a trailing space as its
    # own standalone token (id 220) when nothing follows it, but merges that same
    # space into the next word's leading-space token (e.g. " Obs") once the value is
    # appended. Feeding the model a stem that ends in token 220 asks it to predict
    # what follows a bare space -- a position essentially absent from natural text --
    # while first_token_text() (preprocess/divergence.py) computes the *target* token
    # from the correctly-merged continuous tokenization. Stripped here so the query
    # and the scored token are consistent with each other.
    return template.split("{value}")[0].format(name=name).rstrip(" ")


def value_case_variants(value: str) -> list[str]:
    """Surface-form case variants worth checking as 'correct' for a single value.
    experimental_plans.tex S1.2 notes that a real common-noun value (e.g. a
    field_expertise term like "Paleoclimatology") is reliably case-folded to
    lowercase by the vignette generator when it lands mid-sentence, while invented
    proper-noun values (places, institutions, people) keep their stored
    capitalization -- confirmed concretely for entity_0007's field_expertise value:
    11/12 vignette occurrences are lowercase "paleoclimatology", only the one
    sentence-initial occurrence is capitalized. Which casing the model actually saw
    in training isn't knowable from the pool file alone, so both are checked rather
    than assuming the pool's stored casing is the only one that counts."""
    return list({value, value.lower()})


def candidate_token_texts(template: str, value: str, **kwargs) -> list[str]:
    """First-token text for every case variant of `value`, deduped in order. Each
    variant is independently run through first_token_text() so BPE merges are
    computed against that variant's own continuous rendering, not assumed shared."""
    texts = []
    seen = set()
    for v in value_case_variants(value):
        text = first_token_text(template, v, **kwargs)
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def build_probes(population: list[dict], all_templates: dict[str, list[str]]) -> list[dict]:
    """One probe per entity per fact (5 facts/entity: 1 contested + 4 background)
    per eval template (N_EVAL_TEMPLATES=5) -- 25 probes/entity. Each probe carries
    everything needed to score it once the model's next-token logprobs at its stem
    are available."""
    probes = []
    for entity in population:
        c = entity["contested"]
        templates = all_templates[c["relation_key"]]
        for template_idx, template in enumerate(templates):
            stem = build_stem(template, entity["name"])
            tok_a_texts = candidate_token_texts(template, c["val_a"], name=entity["name"])
            tok_b_texts = candidate_token_texts(template, c["val_b"], name=entity["name"])
            probes.append(
                {
                    "entity_id": entity["entity_id"],
                    "relation_key": c["relation_key"],
                    "is_contested": True,
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
        for relation_key, value in entity["background"].items():
            templates = all_templates[relation_key]
            for template_idx, template in enumerate(templates):
                stem = build_stem(template, entity["name"])
                tok_texts = candidate_token_texts(template, value, name=entity["name"])
                probes.append(
                    {
                        "entity_id": entity["entity_id"],
                        "relation_key": relation_key,
                        "is_contested": False,
                        "template_idx": template_idx,
                        "stem": stem,
                        "value": value,
                        "tok_texts": tok_texts,
                    }
                )
    return probes


# ------------------------------------------------------------------------- inference

@torch.no_grad()
def batched_next_token_logprobs(
    model, tokenizer, stems: list[str], device: str, dtype: torch.dtype
) -> torch.Tensor:
    """Left-padded batch forward pass; returns (batch, vocab) log-probs at each row's
    real last token. Left padding means the real last token is always at index -1
    regardless of a row's padding length, avoiding an explicit per-row gather.
    position_ids are computed from attention_mask (not the model's default
    0..seq_len-1) so left-padding a shorter stem doesn't shift its real tokens'
    absolute position embeddings relative to how they'd be tokenized on their own --
    GPT-2 was trained with no padding, so its position embeddings are content-position
    sensitive; the default position_ids (ignoring attention_mask) would otherwise make
    a probe's score depend on how much padding happened to share its batch."""
    encoded = tokenizer(stems, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)

    with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
        out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    logits = out.logits[:, -1, :].float()
    return torch.log_softmax(logits, dim=-1)


def token_ids_for_texts(tokenizer, texts: list[str]) -> list[int]:
    """Encodes each candidate text and dedupes the resulting first-token ids --
    different case variants occasionally collapse to the same token (e.g. a value
    already all-lowercase), and the aggregation below assumes no id is double-counted."""
    ids = []
    seen = set()
    for text in texts:
        tid = tokenizer.encode(text)[0]
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)
    return ids


def set_logprob(row: torch.Tensor, ids: list[int]) -> float:
    """Aggregate probability mass over every id in a candidate set -- e.g. a value's
    capitalized and lowercased first token both count as 'correct', so the model's
    belief in the fact is their combined mass, not whichever one happens to be
    checked. logsumexp is the correct way to combine log-probabilities of mutually
    exclusive outcomes (the two case variants can't both be the realized next token)."""
    idx = torch.tensor(ids, device=row.device)
    return torch.logsumexp(row[idx], dim=0).item()


def run_recall_eval(
    model, tokenizer, probes: list[dict], device: str, dtype: torch.dtype, batch_size: int = 64
) -> list[dict]:
    records = []
    for start in range(0, len(probes), batch_size):
        batch = probes[start : start + batch_size]
        stems = [p["stem"] for p in batch]
        logprobs = batched_next_token_logprobs(model, tokenizer, stems, device, dtype)

        for i, probe in enumerate(batch):
            row = logprobs[i]
            if probe["is_contested"]:
                tok_a_ids = token_ids_for_texts(tokenizer, probe["tok_a_texts"])
                tok_b_ids = token_ids_for_texts(tokenizer, probe["tok_b_texts"])
                divergence_ok = set(tok_a_ids).isdisjoint(tok_b_ids)
                logp_a = set_logprob(row, tok_a_ids)
                logp_b = set_logprob(row, tok_b_ids)
                rank_a = int((row > logp_a).sum().item()) + 1
                rank_b = int((row > logp_b).sum().item()) + 1
                favored_by_freq = "A" if probe["n_a"] >= probe["n_b"] else "B"
                favored_by_model = "A" if logp_a >= logp_b else "B"
                records.append(
                    {
                        "entity_id": probe["entity_id"],
                        "relation_key": probe["relation_key"],
                        "is_contested": True,
                        "template_idx": probe["template_idx"],
                        "divergence_ok": divergence_ok,
                        "n_a": probe["n_a"],
                        "n_b": probe["n_b"],
                        "logp_a": logp_a,
                        "logp_b": logp_b,
                        "rank_a": rank_a,
                        "rank_b": rank_b,
                        "favored_by_freq": favored_by_freq,
                        "favored_by_model": favored_by_model,
                        "match": divergence_ok and favored_by_freq == favored_by_model,
                    }
                )
            else:
                tok_ids = token_ids_for_texts(tokenizer, probe["tok_texts"])
                logp = set_logprob(row, tok_ids)
                # Full-vocabulary top-1/top-5 selection + logit margin: the accepted
                # candidate set (all case variants, combined via set_logprob above) is
                # scored as a single lumped answer competing against every other token
                # in the vocab -- not restricted to a same-relation-type pool, per the
                # explicit "use the whole set and absolute probability" design
                # decision. rank is "how many single tokens outrank the combined
                # accepted mass"; logit_margin is signed the same way as before (positive
                # when the accepted mass wins, negative deficit when it doesn't), just
                # measured against the best *non-accepted* competitor rather than a
                # blind top-2 (which could otherwise be two case variants of the same
                # correct answer).
                idx = torch.tensor(tok_ids, device=row.device)
                mask = torch.ones_like(row, dtype=torch.bool)
                mask[idx] = False
                other = row[mask]
                rank = int((other > logp).sum().item()) + 1
                logit_margin = logp - other.max().item()
                records.append(
                    {
                        "entity_id": probe["entity_id"],
                        "relation_key": probe["relation_key"],
                        "is_contested": False,
                        "template_idx": probe["template_idx"],
                        "value": probe["value"],
                        "logp": logp,
                        "rank": rank,
                        "top1": rank == 1,
                        "top5": rank <= 5,
                        "logit_margin": logit_margin,
                    }
                )
    return records


# --------------------------------------------------------------------------- summary

def _summarize_contested(records: list[dict]) -> dict:
    """Per entity, averages logp_a/logp_b across only the templates that pass the
    divergence check for that entity (rather than picking one template), then
    applies the same match/gap logic as before to the averaged values -- an
    entity's contested-fact signal is now "does the model prefer the higher-
    frequency side under a random natural phrasing", not "...under this one
    arbitrary sentence". An entity with zero divergence-ok templates is skipped,
    same as the single-template version."""
    by_entity: dict[str, list[dict]] = {}
    for r in records:
        by_entity.setdefault(r["entity_id"], []).append(r)

    contested, skipped = [], []
    for entity_id, recs in by_entity.items():
        valid = [r for r in recs if r["divergence_ok"]]
        if not valid:
            skipped.append(entity_id)
            continue
        logp_a = statistics.mean(r["logp_a"] for r in valid)
        logp_b = statistics.mean(r["logp_b"] for r in valid)
        n_a, n_b = valid[0]["n_a"], valid[0]["n_b"]
        favored_by_freq = "A" if n_a >= n_b else "B"
        favored_by_model = "A" if logp_a >= logp_b else "B"
        contested.append(
            {
                "entity_id": entity_id,
                "n_a": n_a,
                "n_b": n_b,
                "n_templates_used": len(valid),
                "logp_a": logp_a,
                "logp_b": logp_b,
                "favored_by_freq": favored_by_freq,
                "favored_by_model": favored_by_model,
                "match": favored_by_freq == favored_by_model,
            }
        )

    by_split: dict[tuple[int, int], list[dict]] = {}
    for r in contested:
        by_split.setdefault((r["n_a"], r["n_b"]), []).append(r)

    split_summary = []
    for (n_a, n_b), recs in sorted(by_split.items(), key=lambda kv: kv[0][0] - kv[0][1]):
        gaps = [r["logp_a"] - r["logp_b"] for r in recs]
        higher_is_a = n_a >= n_b
        confidences = []
        for r in recs:
            p_a = 1.0 / (1.0 + math.exp(-(r["logp_a"] - r["logp_b"])))
            confidences.append(p_a if higher_is_a else 1.0 - p_a)
        split_summary.append(
            {
                "n_a": n_a,
                "n_b": n_b,
                "freq_gap": n_a - n_b,
                "n_entities": len(recs),
                "accuracy": sum(r["match"] for r in recs) / len(recs),
                "mean_logit_gap_a_minus_b": statistics.mean(gaps),
                "mean_confidence_higher_freq_side": statistics.mean(confidences),
            }
        )

    oriented_gaps = [
        row["mean_logit_gap_a_minus_b"] if row["n_a"] >= row["n_b"] else -row["mean_logit_gap_a_minus_b"]
        for row in split_summary
    ]
    monotonicity_violations = sum(
        1 for i in range(1, len(oriented_gaps)) if oriented_gaps[i] < oriented_gaps[i - 1]
    )
    balanced = next((row for row in split_summary if row["n_a"] == row["n_b"]), None)

    return {
        "n_entities_scored": len(contested),
        "n_entities_skipped_divergence_failure": len(skipped),
        "by_split_level": split_summary,
        "monotonicity_violations": monotonicity_violations,
        "symmetry_at_balance_mean_logit_gap": balanced["mean_logit_gap_a_minus_b"] if balanced else None,
    }


def _summarize_background(records: list[dict], templates_by_relation: dict[str, list[str]]) -> dict:
    """Per relation type: a per-template breakdown (which specific phrasing worked)
    plus two cross-template aggregates per entity -- mean (typical recall under a
    random natural phrasing) and best-of-5 (extractable via at least one phrasing,
    the less phrasing-sensitive "storage" read)."""
    by_rel: dict[str, list[dict]] = {}
    for r in records:
        by_rel.setdefault(r["relation_key"], []).append(r)

    background_summary = []
    for key, recs in sorted(by_rel.items()):
        by_template: dict[int, list[dict]] = {}
        for r in recs:
            by_template.setdefault(r["template_idx"], []).append(r)
        per_template = [
            {
                "template_idx": idx,
                "template": templates_by_relation[key][idx],
                "n_entities": len(trecs),
                "top1_accuracy": sum(r["top1"] for r in trecs) / len(trecs),
                "top5_accuracy": sum(r["top5"] for r in trecs) / len(trecs),
                "mean_rank": statistics.mean(r["rank"] for r in trecs),
                "mean_logit_margin": statistics.mean(r["logit_margin"] for r in trecs),
            }
            for idx, trecs in sorted(by_template.items())
        ]

        by_entity: dict[str, list[dict]] = {}
        for r in recs:
            by_entity.setdefault(r["entity_id"], []).append(r)
        best_rank_per_entity = [min(r["rank"] for r in erecs) for erecs in by_entity.values()]
        any_top1_per_entity = [any(r["top1"] for r in erecs) for erecs in by_entity.values()]
        any_top5_per_entity = [any(r["top5"] for r in erecs) for erecs in by_entity.values()]

        background_summary.append(
            {
                "relation_key": key,
                "n_entities": len(by_entity),
                "n_records": len(recs),
                "mean_top1_accuracy": sum(r["top1"] for r in recs) / len(recs),
                "mean_top5_accuracy": sum(r["top5"] for r in recs) / len(recs),
                "mean_logit_margin": statistics.mean(r["logit_margin"] for r in recs),
                "mean_rank": statistics.mean(r["rank"] for r in recs),
                "best_of_5_top1_accuracy": sum(any_top1_per_entity) / len(any_top1_per_entity),
                "best_of_5_top5_accuracy": sum(any_top5_per_entity) / len(any_top5_per_entity),
                "best_rank_mean": statistics.mean(best_rank_per_entity),
                "by_template": per_template,
            }
        )

    all_top1 = [r["top1"] for r in records]
    all_top5 = [r["top5"] for r in records]
    return {
        "n_entities_scored": len({r["entity_id"] for r in records}),
        "n_records_scored": len(records),
        # Pooled across all 14 relation types and all 5 templates -- comparable
        # across the T-sweep the same way the old single-template pooled number
        # was, just averaged over phrasings now instead of picking one.
        "overall_mean_top1_accuracy": sum(all_top1) / len(all_top1) if records else None,
        "overall_mean_top5_accuracy": sum(all_top5) / len(all_top5) if records else None,
        "overall_mean_logit_margin": statistics.mean(r["logit_margin"] for r in records) if records else None,
        "by_relation_type": background_summary,
    }


def summarize(records: list[dict], templates_by_relation: dict[str, list[str]]) -> dict:
    contested = [r for r in records if r["is_contested"]]
    background = [r for r in records if not r["is_contested"]]
    return {
        "contested": _summarize_contested(contested),
        "background": _summarize_background(background, templates_by_relation),
    }


# --------------------------------------------------------------------------- driver

def evaluate_checkpoint(
    cfg: TrainingConfig, ckpt_dir: Path, population: list[dict], all_templates: dict[str, list[str]], batch_size: int, device: str
) -> dict:
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.run.dtype]

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = build_model(cfg.model)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    probes = build_probes(population, all_templates)
    records = run_recall_eval(model, tokenizer, probes, device, dtype, batch_size=batch_size)
    summary = summarize(records, all_templates)
    return {"checkpoint": str(ckpt_dir), "records": records, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", default=None, help="defaults to training/configs/<run-name>.yaml if present, else full_run.yaml")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="defaults to the latest full checkpoint")
    parser.add_argument("--all-checkpoints", action="store_true", help="run against every full checkpoint on disk, not just one")
    parser.add_argument(
        "--population", required=True,
        help="e.g. results/gpt2-small-openwebtext-T320/population.json -- required, not "
             "defaulted, so a run's T-condition can't be silently mismatched against another "
             "T's population (see memory/results_folder_scatter_cleanup_2026-07-11.md).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-entities", type=int, default=None, help="debug/smoke-test: only probe the first N entities")
    args = parser.parse_args()

    cfg_path = args.config or f"training/configs/{args.run_name}.yaml"
    if not Path(cfg_path).exists():
        cfg_path = "training/configs/full_run.yaml"
    cfg = TrainingConfig.from_yaml(cfg_path)
    device = cfg.run.device if torch.cuda.is_available() else "cpu"

    population = json.load(open(args.population))
    if args.limit_entities is not None:
        population = population[: args.limit_entities]
    relation_keys = {entity["contested"]["relation_key"] for entity in population} | {
        k for entity in population for k in entity["background"]
    }
    all_templates = {key: load_eval_templates(key) for key in relation_keys}

    ckpt_root = Path(cfg.run.output_dir) / args.run_name / "checkpoints"
    all_ckpts = list_full_checkpoints(ckpt_root)
    if not all_ckpts:
        raise FileNotFoundError(f"no full checkpoints found under {ckpt_root}")

    if args.all_checkpoints:
        targets = all_ckpts
    elif args.checkpoint_step is not None:
        targets = [c for c in all_ckpts if c[0] == args.checkpoint_step]
        if not targets:
            raise FileNotFoundError(f"no checkpoint at step {args.checkpoint_step}; have {[c[0] for c in all_ckpts]}")
    else:
        targets = [all_ckpts[-1]]

    results_dir = Path(cfg.run.results_dir) / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for step, kind, ckpt_dir in targets:
        print(f"Evaluating recall at step {step} ({kind}): {ckpt_dir}")
        result = evaluate_checkpoint(cfg, ckpt_dir, population, all_templates, args.batch_size, device)
        out_path = results_dir / f"recall_eval_step{step}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  {len(result['records'])} facts scored -> {out_path}")
        print(f"  contested: {result['summary']['contested']}")


if __name__ == "__main__":
    main()
