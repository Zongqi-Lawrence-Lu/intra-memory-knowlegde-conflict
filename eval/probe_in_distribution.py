"""Diagnostic: is low background recall (eval/recall.py) a phrasing/format-mismatch
problem or a genuine memorization gap? experiments/in-distribution-vs-template-recall/
has the full write-up; short version here.

recall.py scores every background fact against a single fixed, hand-authored template
stem (e.g. "{name} was born in {value}") that the model never saw verbatim during
training -- training text is free-form, LLM-authored vignettes. If the model stored
the fact but simply can't produce it under an unfamiliar phrasing/name-reference form,
that's a knowledge *extraction* gap (Allen-Zhu & Li, Physics of LM Part 3.1), not a
storage gap, and the fixed-template recall numbers would be a lower bound.

This script builds an alternative probe per (entity, relation_key): take one of the
entity's own real generated vignette variants, find the value's actual occurrence in
it (case-insensitive -- values are sometimes case-folded, S1.2), and truncate the
stem right there -- the exact context distribution the model was trained on, instead
of a synthetic template. Scored identically to recall.py's background path (same
rstrip-before-tokenize discipline, same single-token target). Comparing this against
the existing template-based recall_eval_step<N>.json for the *same* entities isolates
the phrasing-mismatch variable directly.

Usage (needs a GPU):
    python -m eval.probe_in_distribution --run-name gpt2-small-openwebtext-T1280-sequential \
        --config training/configs/full_run_T1280.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from eval.recall import batched_next_token_logprobs, set_logprob
from training.config import TrainingConfig
from training.model import build_model

REPO_ROOT = Path(__file__).parent.parent
VIGNETTES_DIR = REPO_ROOT / "preprocess" / "data_pools" / "vignettes"
DEFAULT_POPULATION_PATH = REPO_ROOT / "results" / "population.json"

# Five relation types flagged weak under the template probe, plus two strong ones as
# a sanity control (if in-distribution probing inflated everything indiscriminately,
# these should barely move).
TARGET_KEYS = [
    "authored_work", "mentor", "birthplace", "current_residence", "employer_role",
    "affiliation", "funding_source",
]


def load_variants(entity_id: str) -> list[str]:
    variants = []
    for side in ("A", "B"):
        path = VIGNETTES_DIR / f"{entity_id}_{side}.json"
        if path.exists():
            variants.extend(json.load(open(path))["variants"])
    return variants


def build_indist_probe(tokenizer, text: str, value: str) -> dict | None:
    """stem is a character-level prefix of the real paragraph, but the *target*
    token must come from tokenizing the whole paragraph together and reading off
    whatever token actually sits at that position -- not from re-encoding the
    matched substring standalone. Standalone encoding of e.g. "Dapplewood Capital"
    tokenizes as sentence-initial ('D', 'apple', 'wood', ' Capital'), but the token
    that actually occurs mid-sentence is ' D' (leading space merged in) -- a
    different id entirely. Same class of bug as build_stem()'s trailing-space fix
    in recall.py, just on the target side instead of the query side."""
    m = re.search(re.escape(value), text, re.IGNORECASE)
    if m is None:
        return None
    stem = text[: m.start()].rstrip(" ")
    full_ids = tokenizer.encode(text)
    stem_ids = tokenizer.encode(stem)
    if len(stem_ids) >= len(full_ids):
        return None
    return {"stem": stem, "target_token_id": full_ids[len(stem_ids)]}


def build_probes(tokenizer, population: list[dict]) -> list[dict]:
    probes = []
    for entity in population:
        for relation_key, value in entity["background"].items():
            if relation_key not in TARGET_KEYS:
                continue
            for text in load_variants(entity["entity_id"]):
                probe = build_indist_probe(tokenizer, text, value)
                if probe is not None:
                    probes.append(
                        {
                            "entity_id": entity["entity_id"],
                            "relation_key": relation_key,
                            "value": value,
                            **probe,
                        }
                    )
                    break  # one in-distribution probe per (entity, relation_key)
    return probes


@torch.no_grad()
def run(model, tokenizer, probes: list[dict], device: str, dtype: torch.dtype, batch_size: int = 64) -> list[dict]:
    records = []
    for start in range(0, len(probes), batch_size):
        batch = probes[start : start + batch_size]
        stems = [p["stem"] for p in batch]
        logprobs = batched_next_token_logprobs(model, tokenizer, stems, device, dtype)
        for i, probe in enumerate(batch):
            row = logprobs[i]
            tid = probe["target_token_id"]
            logp = row[tid].item()
            rank = int((row > logp).sum().item()) + 1
            top1_id = int(row.argmax().item())
            records.append(
                {
                    "target_token": tokenizer.decode([tid]),
                    "top1_token": tokenizer.decode([top1_id]),
                    "entity_id": probe["entity_id"],
                    "relation_key": probe["relation_key"],
                    "top1": rank == 1,
                    "top5": rank <= 5,
                    "rank": rank,
                    "logp": logp,
                }
            )
    return records


def summarize(records: list[dict]) -> dict:
    by_rel: dict[str, list[dict]] = {}
    for r in records:
        by_rel.setdefault(r["relation_key"], []).append(r)
    return {
        key: {
            "n": len(recs),
            "top1_accuracy": sum(r["top1"] for r in recs) / len(recs),
            "top5_accuracy": sum(r["top5"] for r in recs) / len(recs),
            "median_rank": statistics.median(r["rank"] for r in recs),
        }
        for key, recs in sorted(by_rel.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--population", default=str(DEFAULT_POPULATION_PATH))
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    device = cfg.run.device if torch.cuda.is_available() else "cpu"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.run.dtype]

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ckpt_dir = Path(cfg.run.output_dir) / args.run_name / "checkpoints" / "latest"
    model = build_model(cfg.model)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    population = json.load(open(args.population))
    probes = build_probes(tokenizer, population)
    print(f"{len(probes)} in-distribution probes built across {len(TARGET_KEYS)} relation types")

    records = run(model, tokenizer, probes, device, dtype, batch_size=args.batch_size)
    summary = summarize(records)

    results_dir = Path(cfg.run.results_dir) / args.run_name
    out_path = results_dir / "in_distribution_probe.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(ckpt_dir), "records": records, "summary": summary}, f, indent=2)
    print(f"-> {out_path}")
    for key, s in summary.items():
        print(f"  {key:22s} n={s['n']:4d} top1={s['top1_accuracy']:.4f} top5={s['top5_accuracy']:.4f} median_rank={s['median_rank']:.0f}")


if __name__ == "__main__":
    main()
