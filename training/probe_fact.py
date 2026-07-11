"""One-off diagnostic: pick a single injected fact (contested or background) for one
entity and check whether the trained model's next-token logits actually reflect it.

This is NOT the M3 shared recall-eval harness (experimental_plans.tex Sec. eval calls
for that to run over every entity/side, at every checkpoint) -- it is a quick,
single-fact spot check requested ad hoc. It reuses the same fixed-template probing
format the full harness is meant to use (preprocess/data_pools/templates/*.json's
first_mention template, per experimental_plans.tex Sec. assembly's "Eval decoupling"),
so results here are representative of what that harness would score.

Usage (must run where a GPU is available -- see CLAUDE.md Sec. 5):
    python -m training.probe_fact --run-name gpt2-small-baseline-openwebtext-t80

Optional: --entity-id entity_0468 --relation civic_role --seed 0
to pin a specific fact instead of a random one.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from preprocess.divergence import first_token_text
from preprocess.schema import RELATION_BY_KEY
from training.checkpoint import list_full_checkpoints
from training.config import TrainingConfig
from training.model import build_model


def load_templates(relation_key: str) -> dict:
    with open(f"preprocess/data_pools/templates/{relation_key}.json") as f:
        return json.load(f)


def load_pool(pool_name: str) -> list[str]:
    with open(f"preprocess/data_pools/{pool_name}.json") as f:
        return json.load(f)


def pick_fact(entity: dict, relation: str | None, rng: random.Random) -> tuple[str, bool, dict]:
    """Returns (relation_key, is_contested, fact_info). fact_info has whatever fields
    are needed to build the probe: {val_a, val_b, n_a, n_b} if contested, else
    {value}."""
    keys = [("contested", entity["contested"]["relation_key"])] + [
        ("background", k) for k in entity["background"]
    ]
    if relation is not None:
        keys = [(kind, k) for kind, k in keys if k == relation]
        if not keys:
            raise ValueError(f"entity {entity['entity_id']} has no fact with relation_key={relation}")
    kind, relation_key = rng.choice(keys)
    if kind == "contested":
        c = entity["contested"]
        return relation_key, True, {"val_a": c["val_a"], "val_b": c["val_b"], "n_a": c["n_a"], "n_b": c["n_b"]}
    return relation_key, False, {"value": entity["background"][relation_key]}


def build_stem(template: dict, name: str) -> str:
    # first_mention template: "{name} ... {value}." -- keep everything up to (not
    # including) {value} so the model must predict the value itself.
    return template["first_mention"].split("{value}")[0].format(name=name)


@torch.no_grad()
def next_token_logprobs(model, tokenizer, stem: str, device: str) -> torch.Tensor:
    ids = tokenizer.encode(stem)
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x)
    logits = out.logits[0, -1, :].float()
    return torch.log_softmax(logits, dim=-1)


def token_rank(logprobs: torch.Tensor, token_id: int) -> int:
    # 1-indexed rank of token_id among the full vocab, by logprob descending.
    return int((logprobs > logprobs[token_id]).sum().item()) + 1


def find_latest_checkpoint(output_dir: str, run_name: str, step: int | None) -> Path:
    # Single-slot checkpointing (training/checkpoint.py): there is only ever one
    # full checkpoint on disk, at checkpoints/latest/, regardless of whether it was
    # produced by the sparse cadence or the final save.
    ckpt_root = Path(output_dir) / run_name / "checkpoints"
    candidates = list_full_checkpoints(ckpt_root)
    if not candidates:
        raise FileNotFoundError(f"no full checkpoint found under {ckpt_root}")
    found_step, _kind, ckpt_dir = candidates[0]
    if step is not None and step != found_step:
        raise FileNotFoundError(f"requested step {step}, but the only checkpoint on disk under {ckpt_root} is step {found_step}")
    return ckpt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="gpt2-small-baseline-openwebtext-t80")
    parser.add_argument("--config", default=None, help="defaults to training/configs/<run-name>.yaml if present, else full_run.yaml")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="defaults to the latest sparse checkpoint")
    parser.add_argument("--entity-id", default=None, help="defaults to a random entity")
    parser.add_argument("--relation", default=None, help="pin a specific relation_key instead of a random fact")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--population", required=True,
        help="e.g. results/gpt2-small-openwebtext-T320/population.json -- required, not "
             "defaulted (see memory/results_folder_scatter_cleanup_2026-07-11.md).",
    )
    parser.add_argument("--num-distractors", type=int, default=5, help="background-fact only: random pool values to show rank/logprob against")
    parser.add_argument("--num-samples", type=int, default=1, help="number of random facts to probe (model is loaded once and reused)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    population = json.load(open(args.population))

    cfg_path = args.config or f"training/configs/{args.run_name}.yaml"
    if not Path(cfg_path).exists():
        cfg_path = "training/configs/full_run.yaml"
    cfg = TrainingConfig.from_yaml(cfg_path)

    ckpt_dir = find_latest_checkpoint(cfg.run.output_dir, args.run_name, args.checkpoint_step)
    device = cfg.run.device if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = build_model(cfg.model)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    print(f"checkpoint: {ckpt_dir}\n")

    for sample_idx in range(args.num_samples):
        entity = (
            next(e for e in population if e["entity_id"] == args.entity_id)
            if args.entity_id is not None
            else rng.choice(population)
        )
        relation_key, is_contested, fact = pick_fact(entity, args.relation, rng)
        template = load_templates(relation_key)
        stem = build_stem(template, entity["name"])

        print(f"=== sample {sample_idx + 1}/{args.num_samples} ===")
        print(f"entity: {entity['entity_id']} ({entity['name']})")
        print(f"relation: {relation_key} ({'CONTESTED' if is_contested else 'background'})")
        print(f"probe stem: {stem!r}")

        logprobs = next_token_logprobs(model, tokenizer, stem, device)

        top5 = torch.topk(logprobs, 5)
        print("top-5 next-token predictions:")
        for logp, tok_id in zip(top5.values.tolist(), top5.indices.tolist()):
            print(f"  {tokenizer.decode([tok_id])!r:20s} logprob={logp:.3f} prob={2.718281828**logp:.4f}")

        if is_contested:
            tok_a_text = first_token_text(template["first_mention"], fact["val_a"], name=entity["name"])
            tok_b_text = first_token_text(template["first_mention"], fact["val_b"], name=entity["name"])
            tok_a = tokenizer.encode(tok_a_text)[0]
            tok_b = tokenizer.encode(tok_b_text)[0]
            logp_a, logp_b = logprobs[tok_a].item(), logprobs[tok_b].item()
            rank_a, rank_b = token_rank(logprobs, tok_a), token_rank(logprobs, tok_b)
            favored_by_freq = "A" if fact["n_a"] >= fact["n_b"] else "B"
            favored_by_model = "A" if logp_a >= logp_b else "B"
            print(f"val_A={fact['val_a']!r} (n_a={fact['n_a']}) -> first token {tok_a_text!r}: logprob={logp_a:.3f} rank={rank_a}")
            print(f"val_B={fact['val_b']!r} (n_b={fact['n_b']}) -> first token {tok_b_text!r}: logprob={logp_b:.3f} rank={rank_b}")
            print(f"more-exposed side: {favored_by_freq}   model-favored side: {favored_by_model}   "
                  f"{'MATCH' if favored_by_freq == favored_by_model else 'MISMATCH'}")
        else:
            value = fact["value"]
            tok_text = first_token_text(template["first_mention"], value, name=entity["name"])
            tok_id = tokenizer.encode(tok_text)[0]
            logp = logprobs[tok_id].item()
            rank = token_rank(logprobs, tok_id)
            print(f"true value={value!r} -> first token {tok_text!r}: logprob={logp:.3f} rank={rank} (out of {logprobs.numel()})")

            pool_name = RELATION_BY_KEY[relation_key].value_pool
            pool = [v for v in load_pool(pool_name) if v != value]
            distractors = rng.sample(pool, min(args.num_distractors, len(pool)))
            print("distractor values from the same pool (for context):")
            for d in distractors:
                d_text = first_token_text(template["first_mention"], d, name=entity["name"])
                d_id = tokenizer.encode(d_text)[0]
                print(f"  {d!r:35s} -> first token {d_text!r}: logprob={logprobs[d_id].item():.3f} rank={token_rank(logprobs, d_id)}")
        print()


if __name__ == "__main__":
    main()
