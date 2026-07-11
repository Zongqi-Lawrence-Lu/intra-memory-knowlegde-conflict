"""
Runner script for DoLa inference-time evaluation.

Loads a trained GPT-2 model, runs DoLa decoding over a probe JSON file,
and writes results to results/.

Probe JSON format (produced by the preprocess pipeline, M2):
[
  {
    "id": "fact_001",
    "prompt": "The president of the United States is",
    "answers": ["Biden", "Trump"]   // optional; if present, scoring mode is used
  },
  ...
]

Results JSON written to results/dola_<experiment_name>.json:
{
  "config": { ... all CLI args ... },
  "results": [
    {
      "id": "fact_001",
      "prompt": "...",
      "generated_text": "...",      // generation mode
      "answer_scores": {...},       // scoring mode (if answers provided)
      "predicted_answer": "...",    // highest-scoring answer (scoring mode)
      "selected_layers": [...]
    },
    ...
  ]
}
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

# Allow running from the repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference_time.dola import dola_generate, score_answers
from inference_time.utils.model_utils import (
    add_model_selection_args,
    default_experiment_name,
    get_device,
    probe_dir_for,
    resolve_model,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DoLa decoding on a probe set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    add_model_selection_args(p)
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted).")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])

    # Probe input
    p.add_argument(
        "--probe_file",
        default=None,
        help="Path to probe JSON file.  If omitted, a small built-in demo is run.",
    )

    # DoLa hyperparameters
    p.add_argument(
        "--mode",
        default="dynamic",
        choices=["dynamic", "static"],
        help="Layer selection mode.",
    )
    p.add_argument(
        "--candidate_layers",
        nargs="+",
        type=int,
        default=None,
        help="Candidate premature layer indices (default: first half of model layers).",
    )
    p.add_argument(
        "--static_layer",
        type=int,
        default=None,
        help="Premature layer index when --mode=static.",
    )
    p.add_argument("--alpha", type=float, default=1.0, help="Contrastive strength.")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    # Output
    p.add_argument(
        "--results_dir",
        default="results",
        help="Directory to write the output JSON.",
    )
    p.add_argument(
        "--experiment_name",
        default=None,
        help="Tag used in the output filename and logs (default: dola_dynamic[_T{T}]).",
    )
    p.add_argument("--log_dir", default="slurm", help="Directory for log file output.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Demo probes (used when no probe file is supplied)
# ---------------------------------------------------------------------------

DEMO_PROBES = [
    {
        "id": "demo_001",
        "prompt": "The capital of France is",
        "answers": ["Paris", "London", "Berlin"],
    },
    {
        "id": "demo_002",
        "prompt": "The boiling point of water in Celsius is",
        "answers": ["100", "0", "37"],
    },
]


# ---------------------------------------------------------------------------
# Per-probe evaluation
# ---------------------------------------------------------------------------

def evaluate_probe(
    model,
    tokenizer,
    probe: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    prompt = probe["prompt"]
    answers = probe.get("answers")

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    candidate_layers = args.candidate_layers  # None → default inside dola_generate

    result = {
        "id": probe.get("id", "unknown"),
        "prompt": prompt,
    }

    if answers:
        # Scoring mode: rank supplied answer choices
        scores = score_answers(
            model,
            tokenizer,
            prompt,
            answers,
            device=device,
            candidate_layers=candidate_layers,
            alpha=args.alpha,
            temperature=args.temperature,
        )
        predicted = max(scores, key=scores.__getitem__)
        result["answer_scores"] = scores
        result["predicted_answer"] = predicted
        logger.info("  [%s] scores=%s  predicted=%s", probe.get("id"), scores, predicted)
    else:
        # Generation mode
        out = dola_generate(
            model,
            tokenizer,
            input_ids,
            candidate_layers=candidate_layers,
            alpha=args.alpha,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            mode=args.mode,
            static_layer=args.static_layer,
            device=device,
        )
        result["generated_text"] = out["generated_text"]
        result["selected_layers"] = out["selected_layers"]
        logger.info("  [%s] generated: %s", probe.get("id"), out["generated_text"][:80])

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.experiment_name is None:
        args.experiment_name = default_experiment_name("dola_dynamic", args)
    setup_logging(args.log_dir, args.experiment_name)
    logger.info("Starting DoLa experiment: %s", args.experiment_name)
    logger.info("Args: %s", vars(args))

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    device = get_device(args.device)
    model, tokenizer = resolve_model(args, device=str(device), dtype=dtype_map[args.dtype])
    logger.info("Model ready.  n_layer=%d", model.config.n_layer)

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
        res = evaluate_probe(model, tokenizer, probe, args, device)
        results.append(res)

    elapsed = time.time() - t0
    logger.info("Done. %d probes in %.1fs (%.2fs/probe)", len(probes), elapsed, elapsed / max(len(probes), 1))

    output = {
        "config": vars(args),
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
