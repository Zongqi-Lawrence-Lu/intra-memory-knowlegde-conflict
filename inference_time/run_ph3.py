"""
Runner script for PH3 (Pruning Heads via PatH PatcHing) inference-time evaluation.

Two inputs:

1. --attribution_file: JSON list of attribution instances used ONCE to score
   attention heads (Phase 1). Not evaluated on.
   [
     {
       "clean_prompt": "As of 2021, the president of the United States is named",
       "corrupted_prompt": "The president of the United States is named",
       "answer_pos": "Biden",
       "answer_neg": "Trump"
     },
     ...
   ]

2. --probe_file: JSON list of probes to evaluate (same schema as run_dola.py):
   [
     {"id": "fact_001", "prompt": "The president of the United States is",
      "answers": ["Biden", "Trump"]},
     ...
   ]

Results written to results/<experiment_name>.json.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference_time.ph3 import extract_head_scores, ph3_generate, score_answers, top_k_heads
from inference_time.utils.model_utils import (
    add_model_selection_args,
    default_experiment_name,
    get_device,
    probe_dir_for,
    resolve_model,
    setup_logging,
)

logger = logging.getLogger(__name__)


DEMO_ATTRIBUTION = [
    {
        "clean_prompt": "As of 2021, the president of the United States is named",
        "corrupted_prompt": "The president of the United States is named",
        "answer_pos": "Biden",
        "answer_neg": "Trump",
    },
]

DEMO_PROBES = [
    {
        "id": "demo_001",
        "prompt": "The president of the United States is",
        "answers": ["Biden", "Trump"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PH3 head-pruning on a probe set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_selection_args(p)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attribution_file", default=None,
                   help="JSON list of attribution instances for Phase 1.")
    p.add_argument("--probe_file", default=None)
    p.add_argument("--top_k_heads", type=int, default=10,
                   help="Number of top-IE heads to prune.")
    p.add_argument("--rho", type=float, default=0.0,
                   help="Attenuation factor (0.0=full ablation, 1.0=no change).")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--experiment_name", default=None, help="default: ph3_top10_ablate[_T{T}]")
    p.add_argument("--log_dir", default="slurm")
    return p.parse_args()


def run_attribution(model, tokenizer, attribution_instances, device) -> torch.Tensor:
    """Average head-IE scores across all attribution instances."""
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    agg_scores = torch.zeros(n_layers, n_heads)

    for inst in attribution_instances:
        # GPT-2 BPE encodes an in-context word with a space prefix (Ġ).
        # Encoding the answer standalone would give the wrong token id.
        pos_tok = tokenizer(" " + inst["answer_pos"].strip())["input_ids"][0]
        neg_tok = tokenizer(" " + inst["answer_neg"].strip())["input_ids"][0]
        scores = extract_head_scores(
            model, tokenizer,
            inst["clean_prompt"], inst["corrupted_prompt"],
            pos_tok, neg_tok, device,
        )
        agg_scores += scores

    agg_scores /= max(len(attribution_instances), 1)
    return agg_scores


def evaluate_probe(model, tokenizer, probe, head_set, args, device) -> dict:
    prompt = probe["prompt"]
    answers = probe.get("answers")
    result = {"id": probe.get("id", "unknown"), "prompt": prompt}

    if answers:
        sc = score_answers(
            model, tokenizer, prompt, answers, head_set,
            rho=args.rho, temperature=args.temperature, device=device,
        )
        predicted = max(sc, key=sc.__getitem__)
        result["answer_scores"] = sc
        result["predicted_answer"] = predicted
        logger.info("  [%s] scores=%s  predicted=%s", probe.get("id"), sc, predicted)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        out = ph3_generate(
            model, tokenizer, input_ids, head_set,
            rho=args.rho, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, device=device,
        )
        result["generated_text"] = out["generated_text"]
        logger.info("  [%s] generated: %s", probe.get("id"), out["generated_text"][:80])

    return result


def main():
    args = parse_args()
    if args.experiment_name is None:
        args.experiment_name = default_experiment_name("ph3_top10_ablate", args)
    setup_logging(args.log_dir, args.experiment_name)
    logger.info("Starting PH3 experiment: %s", args.experiment_name)
    logger.info("Args: %s", vars(args))

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    device = get_device(args.device)
    model, tokenizer = resolve_model(args, device=str(device), dtype=dtype_map[args.dtype])

    pdir = probe_dir_for(args)
    if args.attribution_file is None and pdir is not None:
        args.attribution_file = str(pdir / "attribution.json")
    if args.probe_file is None and pdir is not None:
        args.probe_file = str(pdir / "probes.json")

    # Phase 1: attribution
    if args.attribution_file:
        with open(args.attribution_file) as f:
            attribution_instances = json.load(f)
        logger.info("Loaded %d attribution instances from %s", len(attribution_instances), args.attribution_file)
    else:
        logger.info("No --attribution_file supplied; using built-in demo attribution.")
        attribution_instances = DEMO_ATTRIBUTION

    logger.info("Running Phase 1: head attribution across %d instances...", len(attribution_instances))
    head_scores = run_attribution(model, tokenizer, attribution_instances, device)
    head_set = top_k_heads(head_scores, k=args.top_k_heads)
    logger.info("Top-%d heads selected: %s", args.top_k_heads, sorted(head_set))

    # Phase 2: evaluation
    if args.probe_file:
        with open(args.probe_file) as f:
            probes = json.load(f)
        logger.info("Loaded %d probes from %s", len(probes), args.probe_file)
    else:
        logger.info("No --probe_file supplied; using built-in demo probes.")
        probes = DEMO_PROBES

    results = []
    t0 = time.time()
    for i, probe in enumerate(probes):
        logger.info("Probe %d/%d  id=%s", i + 1, len(probes), probe.get("id", "?"))
        results.append(evaluate_probe(model, tokenizer, probe, head_set, args, device))

    elapsed = time.time() - t0
    logger.info("Done. %d probes in %.1fs", len(probes), elapsed)

    # Serialize head_set as a sorted list of lists for JSON
    output = {
        "config": vars(args),
        "elapsed_seconds": elapsed,
        "pruned_heads": sorted([list(h) for h in head_set]),
        "head_scores_shape": list(head_scores.shape),
        "results": results,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{args.experiment_name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", out_path)

    # Save the full head-score matrix as JSON (*.pt is gitignored per .gitignore)
    scores_path = os.path.join(args.results_dir, f"{args.experiment_name}_head_scores.json")
    with open(scores_path, "w") as f:
        json.dump(head_scores.tolist(), f)
    logger.info("Head scores saved to %s", scores_path)


if __name__ == "__main__":
    main()
