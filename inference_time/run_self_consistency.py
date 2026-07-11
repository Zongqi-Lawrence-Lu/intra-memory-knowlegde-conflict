"""
Runner script for Self-Consistency inference-time evaluation.

Probe JSON format (same schema as run_dola.py):
[
  {
    "id": "fact_001",
    "prompt": "The president of the United States is",
    "answers": ["Biden", "Trump"]   // optional; enables vote_by_scoring mode
  },
  ...
]

Without "answers": uses open-ended majority vote (self_consistent_answer).
With "answers": uses vote_by_scoring (samples + score-based assignment).

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

from inference_time.self_consistency import self_consistent_answer, vote_by_scoring
from inference_time.utils.model_utils import (
    add_model_selection_args,
    default_experiment_name,
    get_device,
    probe_dir_for,
    resolve_model,
    setup_logging,
)

logger = logging.getLogger(__name__)


DEMO_PROBES = [
    {
        "id": "demo_001",
        "prompt": "The president of the United States is",
        "answers": ["Biden", "Trump"],
    },
    {
        "id": "demo_002",
        "prompt": "The capital of France is",
        "answers": ["Paris", "London"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run self-consistency majority-vote on a probe set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_selection_args(p)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--probe_file", default=None)
    p.add_argument("--k", type=int, default=20, help="Number of samples per probe.")
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (should be > 1 for diversity).")
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--temperature_score", type=float, default=1.0,
                   help="Temperature for scoring step in vote_by_scoring mode.")
    p.add_argument("--abstain_threshold", type=float, default=0.0,
                   help="Abstain if top-vote fraction <= this.")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--experiment_name", default=None, help="default: self_consistency_k20[_T{T}]")
    p.add_argument("--log_dir", default="slurm")
    return p.parse_args()


def evaluate_probe(model, tokenizer, probe, args, device) -> dict:
    prompt = probe["prompt"]
    answers = probe.get("answers")
    result = {"id": probe.get("id", "unknown"), "prompt": prompt}

    if answers:
        out = vote_by_scoring(
            model, tokenizer, prompt, answers,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature_sample=args.temperature,
            top_p=args.top_p,
            temperature_score=args.temperature_score,
            abstain_threshold=args.abstain_threshold,
            device=device,
        )
        result["predicted_answer"] = out["answer"]
        result["abstain"] = out["abstain"]
        result["vote_fraction"] = out["vote_fraction"]
        result["vote_counts"] = out["vote_counts"]
        result["answer_scores"] = out["answer_scores"]
        logger.info(
            "  [%s] predicted=%s  vote_frac=%.2f  abstain=%s",
            probe.get("id"), out["answer"], out["vote_fraction"], out["abstain"],
        )
    else:
        out = self_consistent_answer(
            model, tokenizer, prompt,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            abstain_threshold=args.abstain_threshold,
            device=device,
        )
        result["generated_text"] = out["answer"]
        result["abstain"] = out["abstain"]
        result["vote_fraction"] = out["vote_fraction"]
        result["vote_counts"] = out["vote_counts"]
        logger.info(
            "  [%s] answer=%s  vote_frac=%.2f  abstain=%s",
            probe.get("id"), out["answer"], out["vote_fraction"], out["abstain"],
        )

    return result


def main():
    args = parse_args()
    if args.experiment_name is None:
        args.experiment_name = default_experiment_name("self_consistency_k20", args)
    setup_logging(args.log_dir, args.experiment_name)
    logger.info("Starting Self-Consistency experiment: %s", args.experiment_name)
    logger.info("Args: %s", vars(args))

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    device = get_device(args.device)
    model, tokenizer = resolve_model(args, device=str(device), dtype=dtype_map[args.dtype])

    pdir = probe_dir_for(args)
    if args.probe_file is None and pdir is not None:
        args.probe_file = str(pdir / "probes.json")

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
        results.append(evaluate_probe(model, tokenizer, probe, args, device))

    elapsed = time.time() - t0
    logger.info(
        "Done. %d probes × %d samples = %d forward passes in %.1fs",
        len(probes), args.k, len(probes) * args.k, elapsed,
    )

    output = {"config": vars(args), "elapsed_seconds": elapsed, "results": results}
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{args.experiment_name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
