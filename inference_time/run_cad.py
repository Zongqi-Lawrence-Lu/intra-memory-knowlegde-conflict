"""
Runner script for CAD (Context-Aware Decoding) inference-time evaluation.

Probe JSON format (produced by the preprocess pipeline, M2):
[
  {
    "id": "fact_001",
    "prompt": "The president of the United States is",
    "cued_prompt": "As of 2021, the president of the United States is",
    "answers": ["Biden", "Trump"]   // optional; enables scoring mode
  },
  ...
]

The "cued_prompt" field must be provided for every probe.  It stands in for
the external context that CAD was originally designed for, substituted here
with a temporally- or otherwise-disambiguated version of the bare prompt.

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

from inference_time.cad import cad_generate, score_answers
from inference_time.utils.model_utils import (
    get_device,
    load_model_and_tokenizer,
    setup_logging,
)

logger = logging.getLogger(__name__)


DEMO_PROBES = [
    {
        "id": "demo_001",
        "prompt": "The president of the United States is",
        "cued_prompt": "As of 2021, the president of the United States is",
        "answers": ["Biden", "Trump"],
    },
    {
        "id": "demo_002",
        "prompt": "The capital of Australia is",
        "cued_prompt": "The official capital city of Australia is",
        "answers": ["Canberra", "Sydney", "Melbourne"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CAD decoding on a probe set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--probe_file", default=None)
    p.add_argument("--alpha", type=float, default=1.0, help="Contrastive strength.")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--experiment_name", default="cad_alpha1")
    p.add_argument("--log_dir", default="slurm")
    return p.parse_args()


def evaluate_probe(model, tokenizer, probe: dict, args, device) -> dict:
    prompt = probe["prompt"]
    cued_prompt = probe.get("cued_prompt")
    if cued_prompt is None:
        raise ValueError(f"Probe {probe.get('id')} is missing 'cued_prompt'.")
    answers = probe.get("answers")

    result = {"id": probe.get("id", "unknown"), "prompt": prompt, "cued_prompt": cued_prompt}

    if answers:
        scores = score_answers(
            model, tokenizer, cued_prompt, prompt, answers,
            alpha=args.alpha, temperature=args.temperature, device=device,
        )
        predicted = max(scores, key=scores.__getitem__)
        result["answer_scores"] = scores
        result["predicted_answer"] = predicted
        logger.info("  [%s] scores=%s  predicted=%s", probe.get("id"), scores, predicted)
    else:
        cued_ids = tokenizer(cued_prompt, return_tensors="pt")["input_ids"].to(device)
        uncued_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        out = cad_generate(
            model, tokenizer, cued_ids, uncued_ids,
            alpha=args.alpha,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        result["generated_text"] = out["generated_text"]
        logger.info("  [%s] generated: %s", probe.get("id"), out["generated_text"][:80])

    return result


def main():
    args = parse_args()
    setup_logging(args.log_dir, args.experiment_name)
    logger.info("Starting CAD experiment: %s", args.experiment_name)
    logger.info("Args: %s", vars(args))

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    device = get_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model, device=str(device), dtype=dtype_map[args.dtype])

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
    logger.info("Done. %d probes in %.1fs", len(probes), elapsed)

    output = {"config": vars(args), "elapsed_seconds": elapsed, "results": results}
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{args.experiment_name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
