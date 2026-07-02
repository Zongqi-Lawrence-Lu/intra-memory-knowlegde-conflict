# Implementation Roadmap

This document tracks *engineering* sequencing — what gets built, in what order, and why.
It is distinct from `experimental_plans.tex`, which will hold the *scientific* content
(precise definitions, hypotheses, dataset/model/metric specs) that this roadmap implements.
Nothing below should be treated as a final design decision until it's reflected in
`experimental_plans.tex`.

## Guiding dependency chain

Every later stage consumes an artifact from an earlier one, so build order follows the
data → model → eval → intervention chain rather than the "i/ii/iii" grouping in CLAUDE.md:

```
experimental_plans.tex (design decisions)
        │
        ▼
repo scaffolding (folders, README, .gitignore)
        │
        ▼
synthetic conflict-corpus generator  ──► raw/processed data
        │
        ▼
shared eval harness (metrics + probing-prompt runner)
        │
        ▼
GPT-2-medium-from-scratch training pipeline ──► checkpoints + training curves
        │
        ├──► baseline (i): prompting-based mitigation      (no retrain needed)
        ├──► baseline (ii): inference-time interference    (needs trained model + activations)
        └──► baseline (iii): training-time interference    (needs retraining runs)
                │
                ▼
        mech-interp analysis / novel technique (Phase c)
```

Rationale for this order:
- The corpus generator is the highest-leverage piece to get right first — every downstream
  number depends on how conflicts are frequency-, timing-, and form-controlled. Cheap to
  iterate on before any GPU time is spent.
- The eval harness is built *before* training so training runs can log eval metrics from
  checkpoint 1 rather than being bolted on retroactively.
- Among baselines, prompting is cheapest to iterate (no GPU needed beyond inference) and
  validates the eval harness itself; inference-time interference reuses the same trained
  checkpoint; training-time interference is last because it requires full retrains (most
  expensive, so we want the harness proven out first).

## Milestones

### M0 — Design doc
- [ ] Draft `experimental_plans.tex`: formal definition of intra-memory conflict for this
      project, corpus schema (fact templates, conflict types, surface-form variation),
      independent variables to sweep (frequency, recency/order, phrasing diversity),
      metrics, and the baseline list per class (i/ii/iii). This is the "measure twice"
      step — implementation should not start on Phase (a) until this is confirmed with you.

### M1 — Repo scaffolding
- [ ] Folder layout per CLAUDE.md §4: `data/raw/`, `data/processed/`, `output/` (gitignored
      weights/checkpoints), `results/` (tracked JSON), `plots/`, `preprocess/`, `training/`,
      `prompting/`, `inference_time/`, `training_time/`, `mech_interp/`, `experiments/`.
  - [ ] `README.md` documenting the structure.
  - [ ] `.gitignore` tuned so only small/final artifacts (configs, result JSON, plots,
        experimental_plans.tex, code) are tracked — not `.pt` files, raw logs, or full
        datasets.
  - [ ] `sample_sbatch.sbatch` conventions confirmed, `slurm/` output folder created.

### M2 — Synthetic conflict corpus (`preprocess/`)
- [ ] Fact/template engine: generate base "facts" and their conflicting alternates
      (e.g., time-indexed entities like presidents, capitals, record-holders).
- [ ] Conflict-injection controller: parameters for frequency of each variant, temporal/
      positional distribution across the corpus, number of paraphrase templates per fact.
- [ ] Corpus assembler + tokenizer packaging for GPT-2-from-scratch training.
- [ ] Held-out probing set generator (prompts used later by the eval harness), kept
      separate from training data.
- [ ] Output: versioned dataset under `data/processed/`, plus a small metadata JSON
      (recorded in `results/`) describing exact generation parameters for reproducibility.

### M3 — Shared eval harness
- [ ] Metrics: e.g., per-fact answer distribution, consistency under paraphrase, calibration
      between conflicting answers, "which version wins" as a function of frequency/recency.
- [ ] Probing-prompt runner that can be pointed at any checkpoint (works for training-time
      logging and for later baseline comparisons).
- [ ] Result schema written to `results/` as JSON (one file per eval run, model/experiment
      tagged).

### M4 — Training pipeline (`training/`)
- [ ] GPT-2-medium-from-scratch training loop on the synthetic corpus.
- [ ] Checkpointing at ~10-minute intervals, restart/resume from last checkpoint.
- [ ] Logging (loss curves + periodic eval-harness metrics) to `results/`.
- [ ] `sample_sbatch.sbatch`-based job template for single-A40 baseline run; escalation
      path to multi-GPU/A100 noted but not used until a baseline run shows the need.
- [ ] First full run only after you confirm estimated duration + resources (per CLAUDE.md §5).

### M5 — Baseline class (i): Prompting (`prompting/`)
- [ ] Implement classical knowledge-conflict prompting strategies against the trained model.
- [ ] Evaluate with the M3 harness; record to `results/`, own `experiments/<descriptive-name>/`.

### M6 — Baseline class (ii): Inference-time interference (`inference_time/`)
- [ ] Steering / activation patching / ablation / SAE-based methods, each as its own script
      under `inference_time/`, sharing common model-loading utilities.
- [ ] Evaluate with M3 harness.

### M7 — Baseline class (iii): Training-time interference (`training_time/`)
- [ ] Dataset deduplication, reweighting, and other train-time mitigations layered on the
      M4 pipeline.
- [ ] Requires new training runs — scope/duration confirmed with you before submission.

### M8 — Phase (c): Novel technique or mechanistic localization (`mech_interp/`)
- [ ] Scoped once M5–M7 baseline results are in and we can see where the biggest gaps are.

## Immediate next step

M0 (experimental_plans.tex) and M1 (scaffolding) are the only pieces with no open design
dependencies — proposing to start there. M2 (corpus generator) needs a few concrete decisions
first (fact domains to use, how many conflict types, frequency/timing sweep ranges), which
belong in experimental_plans.tex before code is written.
