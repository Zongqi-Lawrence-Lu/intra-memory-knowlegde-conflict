"""
Runner script for K-CAST (kNN-based Conditional Activation Steering).

Two inputs, analogous to run_caa.py but the pair set becomes an instance
*bank* rather than a single pooled vector:

1. --bank_file: JSON list of contrastive instances used to build the kNN bank:
   [
     {"positive": "As of 2021, the US president is named",
      "negative": "As of 2019, the US president is named",
      "query": "The US president is named"},
     ...
   ]
   "query" is optional; if omitted, "positive" is used as the lookup key.

2. --probe_file: JSON list of probes to evaluate (same schema as run_dola.py).

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

from inference_time.kcast import build_instance_bank, kcast_generate, score_answers
from inference_time.utils.model_utils import (
    get_device,
    load_model_and_tokenizer,
    setup_logging,
)

logger = logging.getLogger(__name__)


DEMO_BANK = [
    {"positive": "As of 2021, the president of the United States is named",
     "negative": "As of 2019, the president of the United States is named",
     "query": "The president of the United States is"},
    {"positive": "In 2022, the official capital city of Australia is named",
     "negative": "The largest and most populous city in Australia is named",
     "query": "The capital of Australia is"},
]

DEMO_PROBES = [
    {
        "id": "demo_001",
        "prompt": "The president of the United States is",
        "answers": ["Biden", "Trump"],
    },
    {
        "id": "demo_002",
        "prompt": "The capital of Australia is",
        "answers": ["Canberra", "Sydney", "Melbourne"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run K-CAST activation steering on a probe set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--bank_file", default=None, help="JSON contrastive instances for the kNN bank.")
    p.add_argument("--probe_file", default=None, help="JSON probes to evaluate.")
    p.add_argument("--layer", type=int, default=None, help="Layer to steer (default: model midpoint).")
    p.add_argument("--k", type=int, default=2, help="Number of nearest neighbours to average.")
    p.add_argument("--multiplier", type=float, default=5.0, help="Steering strength.")
    p.add_argument("--positions", default="all", choices=["all", "last"])
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--experiment_name", default="kcast_default")
    p.add_argument("--log_dir", default="slurm")
    return p.parse_args()


def evaluate_probe(model, tokenizer, probe: dict, bank, layer, args, device) -> dict:
    prompt = probe["prompt"]
    answers = probe.get("answers")
    result = {"id": probe.get("id", "unknown"), "prompt": prompt}

    if answers:
        scores, neighbor_idx = score_answers(
            model, tokenizer, prompt, answers, bank, layer,
            k=args.k, multiplier=args.multiplier, positions=args.positions, device=device,
        )
        predicted = max(scores, key=scores.__getitem__)
        result["answer_scores"] = scores
        result["predicted_answer"] = predicted
        result["neighbor_idx"] = neighbor_idx
        logger.info("  [%s] scores=%s  predicted=%s  neighbors=%s",
                     probe.get("id"), scores, predicted, neighbor_idx)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        out = kcast_generate(
            model, tokenizer, input_ids, prompt, bank, layer,
            k=args.k, multiplier=args.multiplier, positions=args.positions,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, repetition_penalty=args.repetition_penalty, device=device,
        )
        result["generated_text"] = out["generated_text"]
        result["neighbor_idx"] = out["neighbor_idx"]
        logger.info("  [%s] generated: %s  neighbors=%s",
                     probe.get("id"), out["generated_text"][:80], out["neighbor_idx"])

    return result


def main():
    args = parse_args()
    setup_logging(args.log_dir, args.experiment_name)
    logger.info("Starting K-CAST experiment: %s", args.experiment_name)
    logger.info("Args: %s", vars(args))

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    device = get_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model, device=str(device), dtype=dtype_map[args.dtype])

    layer = args.layer if args.layer is not None else model.config.n_layer // 2
    logger.info("Steering layer: %d / %d", layer, model.config.n_layer)

    if args.bank_file:
        with open(args.bank_file) as f:
            bank_instances = json.load(f)
        logger.info("Loaded %d bank instances from %s", len(bank_instances), args.bank_file)
    else:
        logger.info("No --bank_file supplied; using built-in demo bank.")
        bank_instances = DEMO_BANK

    bank = build_instance_bank(model, tokenizer, bank_instances, layer, device)

    if args.probe_file:
        with open(args.probe_file) as f:
            probes = json.load(f)
        logger.info("Loaded %d probes from %s", len(probes), args.probe_file)
    else:
        logger.info("No --probe_file supplied; running built-in demo probes.")
        probes = DEMO_PROBES

    results = []
    t0 = time.time()
    for i, probe in enumerate(probes):
        logger.info("Probe %d/%d  id=%s", i + 1, len(probes), probe.get("id", "?"))
        results.append(evaluate_probe(model, tokenizer, probe, bank, layer, args, device))

    elapsed = time.time() - t0
    logger.info("Done. %d probes in %.1fs", len(probes), elapsed)

    output = {
        "config": vars(args),
        "layer": layer,
        "bank_size": len(bank),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{args.experiment_name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
