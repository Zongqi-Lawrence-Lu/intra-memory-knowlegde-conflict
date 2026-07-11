"""
Runner script for ContextFocus inference-time evaluation.

Inputs:

1. --pairs_file: JSON list of cued/bare pairs used to build the steering
   vector (same role as run_caa.py's --pairs_file):
   [
     {"cued": "As of 2021, the president of the United States is named",
      "bare": "The president of the United States is named"},
     ...
   ]

2. --probe_file: JSON list of probes to evaluate. Each probe must carry its
   own disambiguating "cue" string used at generation/scoring time:
   [
     {"id": "fact_001", "prompt": "The president of the United States is",
      "cue": "As of 2021,", "answers": ["Biden", "Trump"]},
     ...
   ]

--mode selects which of ContextFocus's three conditions to run: steering
alone, prompting (cue) alone, or both together -- this directly tests the
paper's "steering composes with prompting" claim as a baseline ablation.

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

from inference_time.contextfocus import (
    MODES,
    build_contextfocus_vector,
    contextfocus_generate,
    score_answers,
)
from inference_time.utils.model_utils import (
    add_model_selection_args,
    default_experiment_name,
    get_device,
    probe_dir_for,
    resolve_model,
    setup_logging,
)

logger = logging.getLogger(__name__)


DEMO_PAIRS = [
    {"cued": "As of 2021, the president of the United States is named",
     "bare": "The president of the United States is named"},
    {"cued": "In the year 2022, the president of the United States is named",
     "bare": "The president of the United States is named"},
]

DEMO_PROBES = [
    {
        "id": "demo_001",
        "prompt": "The president of the United States is",
        "cue": "As of 2021,",
        "answers": ["Biden", "Trump"],
    },
    {
        "id": "demo_002",
        "prompt": "The capital of Australia is",
        "cue": "According to the official government designation,",
        "answers": ["Canberra", "Sydney", "Melbourne"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ContextFocus (steering + prompting) on a probe set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_selection_args(p)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--pairs_file", default=None, help="JSON cued/bare pairs used to build the vector.")
    p.add_argument("--probe_file", default=None, help="JSON probes to evaluate.")
    p.add_argument("--layer", type=int, default=None, help="Layer to steer (default: model midpoint).")
    p.add_argument("--mode", default="both", choices=MODES)
    p.add_argument("--multiplier", type=float, default=5.0, help="Steering strength.")
    p.add_argument("--positions", default="all", choices=["all", "last"])
    p.add_argument("--vector_position", default="last", choices=["last", "mean"])
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--experiment_name", default=None, help="default: contextfocus_both[_T{T}]")
    p.add_argument("--log_dir", default="slurm")
    return p.parse_args()


def evaluate_probe(model, tokenizer, probe: dict, vector, layer, args, device) -> dict:
    prompt = probe["prompt"]
    cue = probe.get("cue")
    answers = probe.get("answers")
    result = {"id": probe.get("id", "unknown"), "prompt": prompt, "cue": cue, "mode": args.mode}

    if args.mode in ("prompting_only", "both") and not cue:
        raise ValueError(f"Probe {probe.get('id')} is missing 'cue' required for mode={args.mode!r}.")

    if answers:
        scores = score_answers(
            model, tokenizer, prompt, cue, answers, vector, layer,
            mode=args.mode, multiplier=args.multiplier, positions=args.positions, device=device,
        )
        predicted = max(scores, key=scores.__getitem__)
        result["answer_scores"] = scores
        result["predicted_answer"] = predicted
        logger.info("  [%s] scores=%s  predicted=%s", probe.get("id"), scores, predicted)
    else:
        out = contextfocus_generate(
            model, tokenizer, prompt, cue, vector, layer,
            mode=args.mode, multiplier=args.multiplier, positions=args.positions,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, repetition_penalty=args.repetition_penalty, device=device,
        )
        result["effective_prompt"] = out["effective_prompt"]
        result["generated_text"] = out["generated_text"]
        logger.info("  [%s] generated: %s", probe.get("id"), out["generated_text"][:80])

    return result


def main():
    args = parse_args()
    if args.experiment_name is None:
        args.experiment_name = default_experiment_name("contextfocus_both", args)
    setup_logging(args.log_dir, args.experiment_name)
    logger.info("Starting ContextFocus experiment: %s (mode=%s)", args.experiment_name, args.mode)
    logger.info("Args: %s", vars(args))

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    device = get_device(args.device)
    model, tokenizer = resolve_model(args, device=str(device), dtype=dtype_map[args.dtype])

    layer = args.layer if args.layer is not None else model.config.n_layer // 2
    logger.info("Steering layer: %d / %d", layer, model.config.n_layer)

    pdir = probe_dir_for(args)
    if args.pairs_file is None and pdir is not None:
        # ContextFocus's own {cued, bare} key names -- distinct from CAA/SpARE's
        # {positive, negative} pairs.json.
        args.pairs_file = str(pdir / "pairs_contextfocus.json")
    if args.probe_file is None and pdir is not None:
        # probes_cueA.json (not plain probes.json): ContextFocus probes need a
        # "cue" field, which only the cued probe files carry.
        args.probe_file = str(pdir / "probes_cueA.json")

    vector = None
    if args.mode in ("steering_only", "both"):
        if args.pairs_file:
            with open(args.pairs_file) as f:
                pairs = json.load(f)
            logger.info("Loaded %d cued/bare pairs from %s", len(pairs), args.pairs_file)
        else:
            logger.info("No --pairs_file supplied; using built-in demo pairs.")
            pairs = DEMO_PAIRS
        vector = build_contextfocus_vector(model, tokenizer, pairs, layer, device, position=args.vector_position)

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
        results.append(evaluate_probe(model, tokenizer, probe, vector, layer, args, device))

    elapsed = time.time() - t0
    logger.info("Done. %d probes in %.1fs", len(probes), elapsed)

    output = {
        "config": vars(args),
        "layer": layer,
        "vector_norm": vector.norm().item() if vector is not None else None,
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
