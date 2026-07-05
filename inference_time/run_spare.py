"""
Runner script for SpARE (SAE-feature steering) inference-time evaluation.

Inputs:

1. --sae_path: path to a pre-trained SAE checkpoint (torch.save'd dict with
   keys W_enc [d_hidden, d_model], b_enc [d_hidden], W_dec [d_hidden, d_model],
   b_dec [d_model]). If omitted, a small demo-only SAE is fit on the fly from
   the contrastive pairs below -- see spare.fit_toy_sae's warning docstring;
   this fallback exists only so this script is runnable before a real SAE
   (mech_interp/, M8) is trained, and must not be used for real experiments.

2. --pairs_file: JSON list of contrastive pairs used to select conflict
   features and build the projected steering vector (same schema as CAA's
   --pairs_file):
   [
     {"positive": "As of 2021, the US president is named",
      "negative": "As of 2019, the US president is named"},
     ...
   ]

3. --probe_file: JSON list of probes to evaluate (same schema as run_dola.py).

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

from inference_time.spare import (
    SparseAutoencoder,
    features_to_residual_vector,
    fit_toy_sae,
    select_conflict_features,
    spare_generate,
    score_answers,
)
from inference_time.utils.model_utils import (
    get_device,
    load_model_and_tokenizer,
    setup_logging,
)
from inference_time.utils.steering_utils import capture_activation

logger = logging.getLogger(__name__)


DEMO_PAIRS = [
    {"positive": "As of 2021, the president of the United States is named",
     "negative": "As of 2019, the president of the United States is named"},
    {"positive": "In the year 2022, the president of the United States is named",
     "negative": "In the year 2018, the president of the United States is named"},
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
        description="Run SpARE SAE-feature steering on a probe set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--sae_path", default=None, help="Pre-trained SAE checkpoint. If omitted, fits a demo-only SAE.")
    p.add_argument("--sae_d_hidden", type=int, default=2048, help="Hidden width for the demo-only SAE fallback.")
    p.add_argument("--pairs_file", default=None, help="JSON contrastive pairs for feature selection.")
    p.add_argument("--probe_file", default=None, help="JSON probes to evaluate.")
    p.add_argument("--layer", type=int, default=None, help="Layer to steer (default: model midpoint).")
    p.add_argument("--top_k_features", type=int, default=8, help="Number of SAE features to steer.")
    p.add_argument("--multiplier", type=float, default=5.0, help="Steering strength.")
    p.add_argument("--positions", default="all", choices=["all", "last"])
    p.add_argument("--vector_position", default="last", choices=["last", "mean"])
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--experiment_name", default="spare_default")
    p.add_argument("--log_dir", default="slurm")
    return p.parse_args()


def evaluate_probe(model, tokenizer, probe: dict, vector, layer, args, device) -> dict:
    prompt = probe["prompt"]
    answers = probe.get("answers")
    result = {"id": probe.get("id", "unknown"), "prompt": prompt}

    if answers:
        scores = score_answers(
            model, tokenizer, prompt, answers, vector, layer,
            multiplier=args.multiplier, positions=args.positions, device=device,
        )
        predicted = max(scores, key=scores.__getitem__)
        result["answer_scores"] = scores
        result["predicted_answer"] = predicted
        logger.info("  [%s] scores=%s  predicted=%s", probe.get("id"), scores, predicted)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        out = spare_generate(
            model, tokenizer, input_ids, vector, layer,
            multiplier=args.multiplier, positions=args.positions,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, repetition_penalty=args.repetition_penalty, device=device,
        )
        result["generated_text"] = out["generated_text"]
        logger.info("  [%s] generated: %s", probe.get("id"), out["generated_text"][:80])

    return result


def main():
    args = parse_args()
    setup_logging(args.log_dir, args.experiment_name)
    logger.info("Starting SpARE experiment: %s", args.experiment_name)
    logger.info("Args: %s", vars(args))

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    device = get_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model, device=str(device), dtype=dtype_map[args.dtype])

    layer = args.layer if args.layer is not None else model.config.n_layer // 2
    logger.info("Steering layer: %d / %d", layer, model.config.n_layer)

    if args.pairs_file:
        with open(args.pairs_file) as f:
            pairs = json.load(f)
        logger.info("Loaded %d contrastive pairs from %s", len(pairs), args.pairs_file)
    else:
        logger.info("No --pairs_file supplied; using built-in demo pairs.")
        pairs = DEMO_PAIRS

    if args.sae_path:
        sae = SparseAutoencoder.load(args.sae_path, device)
        logger.info("Loaded SAE from %s (d_hidden=%d)", args.sae_path, sae.d_hidden)
    else:
        logger.warning("No --sae_path supplied; fitting a DEMO-ONLY SAE (not for real experiments).")
        acts = torch.stack([
            capture_activation(model, tokenizer, pair[k], layer, device, position=args.vector_position)
            for pair in pairs for k in ("positive", "negative")
        ])
        sae = fit_toy_sae(acts, d_hidden=args.sae_d_hidden, device=device)

    feature_idx, feature_delta = select_conflict_features(
        model, tokenizer, sae, pairs, layer, device,
        top_k=args.top_k_features, position=args.vector_position,
    )
    vector = features_to_residual_vector(sae, feature_idx, feature_delta)

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
        "selected_feature_idx": feature_idx.tolist(),
        "vector_norm": vector.norm().item(),
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
