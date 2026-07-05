# Intra-Memory Knowledge Conflict

Studies intra-memory knowledge conflict (both candidate answers to a question are
already parametric, e.g. "who is the president" after training on news from
multiple points in time) and develops efficient mitigations. See
`experimental_plans.tex` for the full scientific design doc (model/corpus/training
spec, baseline survey) and `IMPLEMENTATION.md` for engineering sequencing.

## Repository layout

```
data/
  raw/            gitignored -- downloaded backbone corpus (WikiText-103, etc.)
  processed/      gitignored -- packed-token shards emitted by preprocess/, consumed
                  by training/data.py:PackedTokenDataset
output/           gitignored -- checkpoints (weights + optimizer + RNG state), raw
                  per-run training logs. One subfolder per run_name.
results/          tracked -- small per-run JSON/JSONL/YAML: metrics, config
                  snapshots, run summaries. The "final json" that gets backed up.
plots/            tracked -- figures generated from results/
slurm/            SLURM stdout/stderr (*.out, gitignored); no job scripts checked in
                  yet (*.sbatch is gitignored project-wide).
experiments/      one descriptively-named subfolder per experiment run
preprocess/       M2 (not yet built): synthetic conflict-corpus generator
training/         M4: GPT-2-small-from-scratch training pipeline (this milestone)
prompting/        M5 (not yet built): baseline class (i), prompting-based mitigation
inference_time/   M6 (in progress): baseline class (ii), inference-time interference
training_time/    M7 (in progress): baseline class (iii), training-time interference
mech_interp/      M8 (not yet built): novel technique / mechanistic localization
```

## training/

GPT-2-small-from-scratch training loop, per experimental_plans.tex §1. Model
architecture and optimizer/schedule are built from scratch (HF `transformers`
`GPT2Config`/`GPT2LMHeadModel`, randomly initialized -- no `from_pretrained`); only
the standard pretrained GPT-2 BPE tokenizer's vocabulary size is reused.

- `config.py` -- dataclass config schema, loaded from a YAML file
  (`configs/default.yaml`), so the insertion-schedule/seed sweep can vary one field
  without code changes.
- `model.py` -- builds the from-scratch GPT-2-small model.
- `data.py` -- `PackedTokenDataset` reads pre-tokenized shards from `preprocess/`
  output (`meta.json` + `*.bin`, nanoGPT-style flat memmap convention);
  `DummyTokenDataset` generates random tokens purely to exercise the training loop
  before real data exists. `data.train_path: null` in the config selects the dummy
  path.
- `checkpoint.py` -- dual-cadence, step-indexed checkpointing (experimental_plans.tex
  §1.4): a sparse/global cadence for restart safety, and a dense/event-triggered
  cadence around each entry of `checkpoint.injection_steps` (empty until the M2
  corpus generator supplies a real insertion schedule) for catching phase
  transitions. Only a stratified subset of dense checkpoints keep full weights;
  the rest keep metrics only.
- `eval.py` -- held-out perplexity (experimental_plans.tex §1.5). Per-fact recall is
  out of scope here -- it needs fact metadata from M2 and belongs to the shared
  eval harness (M3), not the training loop.
- `train.py` -- the training loop / CLI entrypoint (`python -m training.train
  --config training/configs/default.yaml`). `--smoke-test` shrinks the model/data to
  run a few steps quickly to sanity-check the loop mechanics (checkpointing, resume,
  logging); it must only be run where a GPU is actually available (see CLAUDE.md §5).
- `logging_utils.py` -- console + per-run file logging (file goes to `output/`, not
  `results/`, so detailed logs are never committed).

Not yet implemented / open items:
- `checkpoint.sparse_interval_steps` in `configs/default.yaml` is a placeholder
  pending real per-step throughput profiling on the cluster (needed to hit the
  ~10-15 min compute-loss bound from experimental_plans.tex §1.4).
- `checkpoint.injection_steps` is empty until M2 (corpus generator) supplies the
  actual insertion schedule.
- No multi-GPU/DDP support yet -- single-device only, per the default single-A40
  plan in CLAUDE.md §2; noted as a possible escalation, not built until needed.
- No cluster job script is checked in (`*.sbatch` is gitignored); a training run
  needs an sbatch header added before submission, and per CLAUDE.md §5 the run's
  estimated duration/resources must be confirmed before any job is actually submitted.

## training_time/

Baseline class (iii): training-time interference (IMPLEMENTATION.md M7). Both
mitigations reuse the same content-blind exact + near-duplicate detection
(`dedup.py`); they differ only in what they do with a detected duplicate
cluster. Neither is wired to an actual training run yet -- both are corpus/
sampler-level tools layered on top of `training/`, to be pointed at the real
M2 corpus once it exists. Per CLAUDE.md §5, no real (non-smoke-test) run may be
submitted without first confirming estimated duration/resources.

- `dedup.py` / `run_dedup.py` -- **hard** dedup: detects exact duplicates (hash
  of normalized text) and near duplicates (MinHash signatures + LSH banding,
  verified against the actual estimated Jaccard similarity) over a jsonl
  document corpus, then drops all but one representative per cluster. Detection
  never reads a document's optional `tag` field (e.g. "injected_fact" vs
  "backbone") -- a real dedup pipeline has no such oracle label, and the point
  of this baseline is to see what a standard, content-blind dedup pass does to
  the controlled fact-repetition signal. `tag` is carried through only so the
  report can break down what got removed.
- `reweighting.py` / `run_reweighting.py` -- **soft** dedup: same clustering as
  `dedup.py`, but instead of dropping duplicates, every document is kept and
  weighted `1 / cluster_size`. Operates at document granularity, which is where
  near-duplicate detection is meaningful; joining these weights into the
  packed-token training loop needs per-window doc-id metadata that the M2
  corpus packer does not emit yet.
- `weighted_dataset.py` -- the reweighting fallback that works against
  `training/data.py` as it exists today: hashes each token *window* (not
  document) for exact-duplicate detection and builds a `WeightedRandomSampler`
  from the inverse window-count weights. No document metadata required, so
  this is what `train_mitigated.py` actually uses; once M2 emits per-window
  doc-ids, prefer joining `reweighting.py`'s document-level weights instead.
- `train_mitigated.py` -- CLI training entrypoint (`python -m
  training_time.train_mitigated --config <yaml> --mitigation {none,reweight}`).
  Deduplication needs no training-loop code of its own: run `run_dedup.py` on
  the raw corpus first, then point a normal `training/configs/*.yaml` at the
  deduped output and use `python -m training.train` directly. Reweighting is
  wired in via a single additive hook in `training/train.py:train()`
  (`build_sampler`, default `None` -> unchanged uniform-shuffle behavior) so the
  training loop itself is not duplicated.
- `configs/reweight_dev.yaml` -- same schema/defaults as
  `training/configs/default.yaml`; the mitigation itself is a CLI flag, not a
  separate config schema.
