"""CLI entrypoint for the M7 training-time mitigation baselines.

Deduplication is a *preprocessing-stage* mitigation: run `training_time.run_dedup`
on the raw corpus first, then point a normal `training/configs/*.yaml`'s
`data.train_path` at the deduplicated output and run `python -m training.train`
as usual -- no special training-loop code is needed, so there is no "dedup mode"
here.

Reweighting is a *training-loop-stage* mitigation: this script computes
inverse-duplicate-count weights over the packed training windows
(training_time/weighted_dataset.py) and swaps the uniform-shuffle sampler for a
WeightedRandomSampler, via the `build_sampler` hook training/train.py exposes
for exactly this purpose -- the rest of the loop (model, optimizer, checkpoint,
eval, logging) is training.train.train() unmodified.

Usage:
    python -m training_time.train_mitigated --config training/configs/default.yaml --mitigation reweight
    python -m training_time.train_mitigated --config training/configs/default.yaml --mitigation none --smoke-test

Per CLAUDE.md §5, do not submit an actual (non-smoke-test) run without first
confirming estimated duration/resources.
"""
from __future__ import annotations

import argparse

from training.config import TrainingConfig
from training.train import train
from training_time.weighted_dataset import build_weighted_sampler, compute_window_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mitigation", choices=["none", "reweight"], default="reweight")
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)

    build_sampler = None
    if args.mitigation == "reweight":
        min_weight = args.min_weight

        def build_sampler(train_ds):
            weights = compute_window_weights(train_ds, min_weight=min_weight)
            return build_weighted_sampler(weights)

    train(cfg, smoke_test=args.smoke_test, build_sampler=build_sampler)


if __name__ == "__main__":
    main()
