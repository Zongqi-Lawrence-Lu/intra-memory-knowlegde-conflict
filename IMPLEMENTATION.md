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
GPT-2-small-from-scratch training pipeline ──► checkpoints + training curves
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
(Superseded the original presidents/capitals framing entirely -- see
experimental_plans.tex S1 for the actual current design.)
- [x] Fact engine: synthetic entities with a contested relation (two candidate
      values) + K=4 background relations, drawn from a 14-relation-type inventory,
      balanced by construction (S1.2, S1.6). `schema.py`, `entities.py`.
- [x] Conflict-injection controller: split levels (n_A, n_B) of a fixed exposure
      budget T, i.i.d.\ uniform occurrence placement, no recency (S1.3, S1.6).
      `scheduler.py`.
- [x] Corpus assembler + tokenizer packaging (S1.7): interleaves WikiText-103 +
      LLM-authored vignette occurrences at backbone line boundaries into
      `PackedTokenDataset`'s on-disk contract. `assemble_corpus.py`, smoke-tested end
      to end on the real backbone at dev scale -- **not yet run at full scale**,
      pending vignette-generation completion and the S1.11 exposure-budget pilot
      (which also fixes `total_tokens`, `T`, and likely `V`).
- [x] Held-out probing set: resolved differently than originally envisioned --
      side-A/side-B training vignettes are independently LLM-authored and share no
      common token to compare at, so per-fact recall is scored on a separate,
      fixed-template probe (the original per-relation-type templates, kept purely
      for this) rather than a generated probing prompt set (S1.7).
- [x] Output format: `data/processed/<name>/` (`meta.json` + `*.bin`) +
      `results/population.json` + `results/occurrence_log.json`. Dev-scale run
      exists at `data/processed/dev-run/`; the real, versioned full-scale dataset is
      pending the items above.

### M3 — Shared eval harness
- [ ] Metrics: e.g., per-fact answer distribution, consistency under paraphrase, calibration
      between conflicting answers, "which version wins" as a function of frequency/recency.
- [ ] Probing-prompt runner that can be pointed at any checkpoint (works for training-time
      logging and for later baseline comparisons).
- [ ] Result schema written to `results/` as JSON (one file per eval run, model/experiment
      tagged).

### M4 — Training pipeline (`training/`)
- [x] GPT-2-small-from-scratch training loop on the synthetic corpus (default model
      as of 2026-07-03; see experimental_plans.tex §1 for rationale and the
      GPT-2-medium escalation path). `training/model.py` + `training/train.py`;
      data is read via `training/data.py:PackedTokenDataset`, which currently falls
      back to `DummyTokenDataset` since M2 hasn't produced real shards yet.
- [x] Dual-cadence checkpointing per experimental_plans.tex §1.4 — sparse
      step-indexed cadence (~10-15 min compute loss bound) for restart safety,
      dense step-indexed cadence in windows following each fact injection for
      phase-transition detection. Supersedes the flat 10-minute rule in CLAUDE.md
      §5 for this pipeline specifically. `training/checkpoint.py`; sparse interval
      is a placeholder pending real throughput profiling, `injection_steps` is
      empty pending M2.
- [x] Logging (loss curves + periodic eval-harness metrics) to `results/`.
      `training/logging_utils.py` (detailed log → `output/`, gitignored) +
      `training/checkpoint.py:log_metrics` (small JSONL → `results/`, tracked).
      Held-out perplexity implemented in `training/eval.py`; per-fact recall is
      explicitly out of scope for the training loop (needs M2/M3).
- [ ] `sample_sbatch.sbatch`-based job template for single-A40 baseline run; escalation
      path to multi-GPU/A100 noted but not used until a baseline run shows the need.
      Deferred — no sbatch template exists in the repo yet.
- [ ] First full run only after you confirm estimated duration + resources (per CLAUDE.md §5).

### M5 — Baseline class (i): Prompting (`prompting/`)
- [ ] Implement classical knowledge-conflict prompting strategies against the trained model.
- [ ] Evaluate with the M3 harness; record to `results/`, own `experiments/<descriptive-name>/`.

### M6 — Baseline class (ii): Inference-time interference (`inference_time/`)

Shared infrastructure:
- [x] `inference_time/utils/model_utils.py` — model loading, device helpers, logging setup

**Logit/Decoding-Level Contrastive Methods**
- [x] **DoLa** (`dola.py`, `run_dola.py`) — contrasts final-layer vs early/mid-layer logits;
      dynamic (max-JSD) or static layer selection; includes `score_answers` for
      multiple-choice probing.  No external context needed — directly applicable to
      intra-memory setting.
- [x] **CAD** (`cad.py`, `run_cad.py`) — with-cue vs without-cue contrastive decoding;
      requires a disambiguating-cue pair substituting for the missing external context.
- [x] **COIECD** (`coiecd.py`, `run_coiecd.py`) — entropy-gated CAD; applies contrast only
      on tokens where an entropy-based conflict signal fires; outputs per-step conflict flags.
- [x] **AdaCAD** (`adacad.py`, `run_adacad.py`) — JS-divergence-adaptive contrastive strength
      instead of CAD's fixed coefficient; outputs per-step α values.
- [x] **CoCoA** (`cocoa.py`, `run_cocoa.py`) — adds entropy-gap + contextual-peakedness
      signals on top of AdaCAD's divergence term; equal-weight default, all γ configurable.
- [x] **ARR** (`arr.py`, `run_arr.py`) — Adaptive Regime Routing; routes each step between
      trust-prior, trust-context, agree, and blend regimes; outputs per-step regime labels.

Shared steering infrastructure:
- [x] `inference_time/utils/steering_utils.py` — activation capture, residual-stream
      addition hook, and shared generate/score-with-steering loops reused by all four
      methods below.

**Activation Steering Methods**
- [x] **CAA** (`caa.py`, `run_caa.py`) — Contrastive Activation Addition; builds a steering
      vector from mean-activation differences between contrastive prompt pairs and adds it
      (scaled) to the residual stream.
- [x] **SpARE** (`spare.py`, `run_spare.py`) — steers sparse-autoencoder features rather
      than raw residual directions; requires a pre-trained SAE on the model (consumes an
      existing checkpoint via `SparseAutoencoder.load`; `fit_toy_sae` is a demo-only
      fallback for smoke-testing until a real SAE exists under `mech_interp/`, M8).
- [x] **K-CAST** (`kcast.py`, `run_kcast.py`) — kNN-based per-instance steering; looks up
      nearest neighbours of the current instance and computes a per-instance steering direction.
- [x] **ContextFocus** (`contextfocus.py`, `run_contextfocus.py`) — steering that composes
      with prompting; `--mode {prompting_only,steering_only,both}` operationalizes the
      composition claim as a direct ablation.

**Attention-Level Intervention**
- [x] **PH3** (`ph3.py`, `run_ph3.py`) — Pruning Heads via PatH PatcHing; two-phase:
      Phase 1 runs path-patching attribution to score every (layer, head) pair by indirect
      effect on the logit difference; Phase 2 attenuates top-k heads via a forward hook
      with configurable scale factor ρ.  Head scores saved as `.pt` alongside results JSON.

**Conflict-Detection Signals** (used as gates for the methods above, not standalone mitigations)
- [x] **Semantic entropy / DynamicQA** (`semantic_entropy.py`) — samples model at high
      temperature across paraphrases and measures semantic divergence as a conflict signal;
      exposes `conflict_score` (single-prompt), `paraphrase_consistency_score` (multi-paraphrase
      CP score), and two clustering modes ("exact" / "embed" via sentence-transformers).

**Ensemble / Sampling**
- [x] **Self-consistency** (`self_consistency.py`, `run_self_consistency.py`) — majority vote
      over K samples; open-ended mode (raw majority vote) and multiple-choice mode
      (`vote_by_scoring`: samples + nearest-answer assignment); abstain threshold configurable.

### M7 — Baseline class (iii): Training-time interference (`training_time/`)
- [x] **Dedup** (`dedup.py`, `run_dedup.py`) — content-blind exact-hash + MinHash/LSH
      near-duplicate clustering over a jsonl document corpus; drops all but one
      representative per cluster. `tag` (e.g. injected-fact vs backbone) is carried
      through for reporting only, never used to decide what gets dropped.
- [x] **Reweighting** (`reweighting.py`, `run_reweighting.py`) — same clustering as
      dedup, "soft" variant: keeps every document, weights each `1 / cluster_size`.
      Document-granularity; joining into the packed-token training loop needs
      per-window doc-id metadata the M2 packer doesn't emit yet.
- [x] **Window-level fallback + training-loop integration** (`weighted_dataset.py`,
      `train_mitigated.py`) — exact-duplicate detection directly on packed token
      windows (works today, no M2 doc-boundary metadata needed) feeding a
      `WeightedRandomSampler`; wired into `training/train.py:train()` via a single
      additive `build_sampler` hook (default `None` preserves prior behavior exactly).
- [ ] Requires new training runs — scope/duration confirmed with you before submission.
      Code above is built and logic-tested locally (no LM loaded/run, per CLAUDE.md
      §5); no actual training job has been submitted.

### M8 — Phase (c): Novel technique or mechanistic localization (`mech_interp/`)
- [ ] Scoped once M5–M7 baseline results are in and we can see where the biggest gaps are.

## Immediate next step

M0 (experimental_plans.tex) and M1 (scaffolding) are the only pieces with no open design
dependencies — proposing to start there. M2 (corpus generator) needs a few concrete decisions
first (fact domains to use, how many conflict types, frequency/timing sweep ranges), which
belong in experimental_plans.tex before code is written.
