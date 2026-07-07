"""Per-fact recall harness -- experimental_plans.tex's eval section ("shared eval
harness (M3)", previously not implemented anywhere: training/eval.py only ever did
held-out perplexity, see that module's own docstring).

For every entity in results/population.json and every one of its 5 facts (1 contested
relation + 4 background relations), renders the fixed-template probing stem
(preprocess/data_pools/templates/<relation_key>.json's first_mention, up to but not
including {value} -- the same "eval decoupled from training text" design
experimental_plans.tex settled on, S1.7/S1.9), runs the model, and reads off the
target value's (or, for a contested fact, both candidate values') log-probability and
rank at that single position -- one batched forward pass per probe, not per candidate,
since both candidates are scored from the same distribution.

Usage (needs a GPU -- see CLAUDE.md S5):
    python -m eval.recall --run-name gpt2-small-baseline-openwebtext-t80
    python -m eval.recall --run-name gpt2-small-baseline-openwebtext-t80 --checkpoint-step 37800
    python -m eval.recall --run-name gpt2-small-baseline-openwebtext-t80 --all-checkpoints

Output: results/<run_name>/recall_eval_step<N>.json (tracked, per CLAUDE.md S6) --
every per-fact record (tagged with relation_key and is_contested, so results can be
sliced by relation type later) plus summary stats: per-split-level accuracy/logit-gap/
monotonicity for contested facts; for background (non-conflict) facts, full-vocabulary
top-1/top-5 selection accuracy and signed logit margin (explicit design choice: scored
against the whole 50,257-token vocab at absolute probability, not restricted to a
same-relation-type candidate pool), per relation type and pooled overall -- the
pooled number is the one to compare across the T-sweep, since background facts get
exactly T exposures/entity regardless of contested split level.
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
DEFAULT_POPULATION_PATH = REPO_ROOT / "results" / "population.json"


# --------------------------------------------------------------------- probe building

def load_template(relation_key: str) -> dict:
    with open(TEMPLATES_DIR / f"{relation_key}.json") as f:
        return json.load(f)


def build_stem(template: dict, name: str) -> str:
    return template["first_mention"].split("{value}")[0].format(name=name)


def build_probes(population: list[dict], templates: dict[str, dict]) -> list[dict]:
    """One probe per entity per fact (5 facts/entity: 1 contested + 4 background).
    Each probe carries everything needed to score it once the model's next-token
    logprobs at its stem are available."""
    probes = []
    for entity in population:
        c = entity["contested"]
        template = templates[c["relation_key"]]
        stem = build_stem(template, entity["name"])
        tok_a_text = first_token_text(template["first_mention"], c["val_a"], name=entity["name"])
        tok_b_text = first_token_text(template["first_mention"], c["val_b"], name=entity["name"])
        probes.append(
            {
                "entity_id": entity["entity_id"],
                "relation_key": c["relation_key"],
                "is_contested": True,
                "stem": stem,
                "val_a": c["val_a"],
                "val_b": c["val_b"],
                "tok_a_text": tok_a_text,
                "tok_b_text": tok_b_text,
                "n_a": c["n_a"],
                "n_b": c["n_b"],
            }
        )
        for relation_key, value in entity["background"].items():
            template = templates[relation_key]
            stem = build_stem(template, entity["name"])
            tok_text = first_token_text(template["first_mention"], value, name=entity["name"])
            probes.append(
                {
                    "entity_id": entity["entity_id"],
                    "relation_key": relation_key,
                    "is_contested": False,
                    "stem": stem,
                    "value": value,
                    "tok_text": tok_text,
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
                tok_a_ids = tokenizer.encode(probe["tok_a_text"])
                tok_b_ids = tokenizer.encode(probe["tok_b_text"])
                tok_a, tok_b = tok_a_ids[0], tok_b_ids[0]
                divergence_ok = tok_a != tok_b
                logp_a, logp_b = row[tok_a].item(), row[tok_b].item()
                rank_a = int((row > logp_a).sum().item()) + 1
                rank_b = int((row > logp_b).sum().item()) + 1
                favored_by_freq = "A" if probe["n_a"] >= probe["n_b"] else "B"
                favored_by_model = "A" if logp_a >= logp_b else "B"
                records.append(
                    {
                        "entity_id": probe["entity_id"],
                        "relation_key": probe["relation_key"],
                        "is_contested": True,
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
                tok_id = tokenizer.encode(probe["tok_text"])[0]
                logp = row[tok_id].item()
                rank = int((row > logp).sum().item()) + 1
                # Full-vocabulary top-1/top-5 selection + logit margin: top2 over the
                # whole (vocab,) row, not restricted to any candidate pool, per the
                # explicit "use the whole set and absolute probability" design
                # decision. logit_margin is signed: how much the model's actual top
                # pick wins by when it's correct (rank==1), or how far behind the
                # correct token is when it's wrong (rank>1). log_softmax's constant
                # logsumexp normalizer cancels in any difference of two entries from
                # the same row, so this is numerically identical to a raw-logit margin.
                top2 = torch.topk(row, 2)
                top1_logprob, runner_up_logprob = top2.values[0].item(), top2.values[1].item()
                if rank == 1:
                    logit_margin = top1_logprob - runner_up_logprob
                else:
                    logit_margin = logp - top1_logprob
                records.append(
                    {
                        "entity_id": probe["entity_id"],
                        "relation_key": probe["relation_key"],
                        "is_contested": False,
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

def summarize(records: list[dict]) -> dict:
    contested = [r for r in records if r["is_contested"] and r["divergence_ok"]]
    skipped = [r for r in records if r["is_contested"] and not r["divergence_ok"]]
    background = [r for r in records if not r["is_contested"]]

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

    by_rel: dict[str, list[dict]] = {}
    for r in background:
        by_rel.setdefault(r["relation_key"], []).append(r)
    background_summary = [
        {
            "relation_key": key,
            "n_entities": len(recs),
            # Headline metrics (experimental_plans.tex background-recall test, full
            # vocab / absolute probability, not restricted to a candidate pool):
            "top1_accuracy": sum(r["top1"] for r in recs) / len(recs),
            "top5_accuracy": sum(r["top5"] for r in recs) / len(recs),
            "mean_logit_margin": statistics.mean(r["logit_margin"] for r in recs),
            # Secondary/legacy context (full-50257-vocab rank -- noisier, since most
            # vocab mass legitimately goes to generic tokens regardless of recall).
            "mean_rank": statistics.mean(r["rank"] for r in recs),
            "median_rank": statistics.median(r["rank"] for r in recs),
            "mean_logprob": statistics.mean(r["logp"] for r in recs),
        }
        for key, recs in sorted(by_rel.items())
    ]

    return {
        "contested": {
            "n_entities_scored": len(contested),
            "n_entities_skipped_divergence_failure": len(skipped),
            "by_split_level": split_summary,
            "monotonicity_violations": monotonicity_violations,
            "symmetry_at_balance_mean_logit_gap": balanced["mean_logit_gap_a_minus_b"] if balanced else None,
        },
        "background": {
            "n_entities_scored": len(background),
            # Pooled across all 14 relation types -- the single headline number for
            # comparing background recall across T conditions (background facts get
            # exactly T exposures/entity regardless of contested split level, so this
            # is directly comparable across the T-sweep with no split-level conditioning).
            "overall_top1_accuracy": sum(r["top1"] for r in background) / len(background) if background else None,
            "overall_top5_accuracy": sum(r["top5"] for r in background) / len(background) if background else None,
            "overall_mean_logit_margin": statistics.mean(r["logit_margin"] for r in background) if background else None,
            "by_relation_type": background_summary,
        },
    }


# --------------------------------------------------------------------------- driver

def evaluate_checkpoint(
    cfg: TrainingConfig, ckpt_dir: Path, population: list[dict], templates: dict[str, dict], batch_size: int, device: str
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

    probes = build_probes(population, templates)
    records = run_recall_eval(model, tokenizer, probes, device, dtype, batch_size=batch_size)
    summary = summarize(records)
    return {"checkpoint": str(ckpt_dir), "records": records, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", default=None, help="defaults to training/configs/<run-name>.yaml if present, else full_run.yaml")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="defaults to the latest full checkpoint")
    parser.add_argument("--all-checkpoints", action="store_true", help="run against every full checkpoint on disk, not just one")
    parser.add_argument("--population", default=str(DEFAULT_POPULATION_PATH))
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
    templates = {key: load_template(key) for key in relation_keys}

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
        result = evaluate_checkpoint(cfg, ckpt_dir, population, templates, args.batch_size, device)
        out_path = results_dir / f"recall_eval_step{step}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  {len(result['records'])} facts scored -> {out_path}")
        print(f"  contested: {result['summary']['contested']}")


if __name__ == "__main__":
    main()
