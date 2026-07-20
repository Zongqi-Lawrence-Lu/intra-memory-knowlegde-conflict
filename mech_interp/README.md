# mech_interp/

Representational and causal localization of how a contested pair is stored and
arbitrated internally, plus a continuously-controllable steering dial built from the
causally-identified locus (IMPLEMENTATION.md M8, experimental_plans.tex §3).

Kept separate from `inference_time/` (M6: already-built candidate mitigations,
e.g. CAA/DoLa/temperature) and `training_time/` (M7): this module reads a
fixed frozen checkpoint and writes to `results/mech_interp/<run-name>/<stage>/`;
it does not itself select or ship a mitigation. Shares the relation-type-restricted
population and T=1280 checkpoint of experimental_plans.tex §sec:relation-restriction,
and reuses the contrastive-pair construction of §sec:steering for probe/patch
inputs -- everything downstream (layerwise probing, causal tracing, path
patching, dial steering) is new.

Each stage's CLI (`run_*.py`) is named for what it does, not for its position in
experimental_plans.tex's §3 narrative -- the module docstrings still cross-reference
that narrative ("Phase 0/1/2/..." in the design doc) for the scientific rationale,
but no file, directory, or CLI name encodes a stage index, and every stage can be
invoked standalone (`python -m mech_interp.run_<stage> --T 1280`). Two stages have a
genuine *data* dependency on another stage's output file (not a naming one):
suppression needs causal tracing's `causal_map.json` for the same `--T`, and the
dial infers its steering layer from the same file unless `--layer` is passed
explicitly. Everything else -- scaffolding, probing, causal tracing, and the sweep
orchestrator -- is independently runnable.

## Status

First implementation pass, logic-tested locally without a GPU per CLAUDE.md §5
(no LM loaded or run from this node): every module imports cleanly, and the
highest-risk piece -- the clean/corrupt prompt-pair alignment
`cued_query_examples`/`resolve_query_positions` in `common.py` relies on for
patching -- was checked tokenizer-only (no model) against the full real T=1280
top-7 population (3,360 clean/corrupt pairs, 0 misalignments after fixing an
off-by-one BPE leading-space bug found during that check).

2026-07-17 review pass (before the first real GPU job): found and fixed a bug by
inspection, no GPU needed --
  - `causal_tracing.run_causal_map_both_directions`'s B-clean relabeling step
    flipped the sign of every logit-diff-derived number back to the original A/B
    identity but left `n_a`/`n_b` un-swapped, so `maps_b_clean_original_labels`
    entries in `causal_map.json` reported the wrong entity exposure counts (only
    a metadata/reporting bug -- `classify_components` never reads `n_a`/`n_b`, so
    the shared/disjoint/inactive classification itself was unaffected).

Six small `--limit-entities` smoke tests (one per stage) then ran cleanly against
the real T=1280 checkpoint, followed by a full-population run. The full-population
`run_sweep.py` job crashed (`IndexError: index 12 is out of range` on
`model.transformer.h[12]`) the first time causal tracing's own strongest layer
happened to be the last block -- tracked down to a SECOND, genuine layer-index bug
this project has three different layer-index conventions in play:
  - `causal_tracing.py`: a transformer BLOCK index (0..n_layer-1) -- residual
    granularity patches/ablates block `layer`'s own forward output.
  - `mech_interp.common.capture_all_layers_*` (what `probing.py` and
    `steering_dial.build_dial_vector` read): the raw `hidden_states` tuple from
    `model.forward(output_hidden_states=True)`, where index `layer` is block
    `layer`'s INPUT (= block `layer-1`'s output) -- one block EARLIER than the
    block-index convention above.
  - `inference_time.utils.steering_utils.steering` (what `score_steered`
    actually hooks): a transformer BLOCK index, matching causal_tracing.py's
    convention exactly (confirmed by reading `register_addition_hook` --
    `model.transformer.h[layer].register_forward_hook(...)`, same as
    causal_tracing.py's own residual-granularity hook).

  So `steering_dial.build_dial_vector`'s `layer` argument needs to be a BLOCK
  index (matching causal_tracing.py's output AND `steering`'s injection point
  directly, no conversion at that handoff) -- but the function itself was
  reading the vector from `hidden_states[layer]` instead of `hidden_states[layer
  + 1]`, one block earlier than where the vector then gets injected. The FIRST
  attempted fix put the `+1` at the wrong boundary (converting causal tracing's
  block index before passing it to `build_dial_vector`), which is what crashed
  the full-scale run; the correct fix moves the `+1` inside `build_dial_vector`
  itself, matching the identical `hidden_states[L + 1]` conversion
  `eval.interventions.caa_grid.capture_all_layers_last_token` already performs
  internally for CAA's own (already-validated) vectors. This mismatch was
  present in the very first smoke test too (dial job, 2026-07-17, "steering at
  layer=11") and plausibly explains that run's unexplained reversed-direction
  steering effect -- not confirmed, but a strong candidate now that the
  vector-source/injection-point mismatch is understood.

No actual scaffolding/probing/causal-tracing/suppression/dial/sweep job had been run
against the trained checkpoint before this review -- smoke tests, then a
full-population run, were submitted once the naming pass and bug fixes above were
in, per CLAUDE.md §5's check-in-before-submitting convention.

## Layout

- `common.py` -- shared engine: output dirs (`results/mech_interp/<run>/<stage>/`,
  via `stage_dir`), population/split reuse from `eval.interventions.common`, left-padded
  batch encoding with position-id correction, entity-mention token-position
  resolution, per-example multi-layer activation capture at an arbitrary
  position, a public logit-lens unembed helper, and the three text-construction
  builders every stage draws from: `contested_side_examples` (plain A/B
  sentences), `background_examples` (neutral, non-contested sentences), and
  `cued_query_examples` (length-robust clean/corrupt cue+shared-stem pairs for
  patching).
- `scaffolding.py` / `run_scaffolding.py` -- generation vs. verification probes,
  multi-verse / clean-suppression / mismatch / verifies-neither classification.
  Standalone.
- `probing.py` / `run_probing.py` -- per-layer, per-position
  (`last`, `entity_mention`) linear probes for "encodes A" / "encodes B"
  against a shared neutral negative class, their angle, and angle to
  conflict-ratio and (from `occurrence_log.json`) recency regression probes.
  Standalone.
- `causal_tracing.py` / `run_causal_tracing.py` -- activation patching / path
  patching (residual stream, attention heads via the same
  `attn.c_proj` pre-hook technique as `inference_time/ph3.py`, MLP blocks via
  the analogous `mlp.c_proj` hook), restoration + necessity effects, and
  shared-vs-disjoint component classification across both clean orientations.
  Standalone. The expensive stage -- see `causal_map_for_example`'s docstring for
  the per-example forward-pass count before widening `--limit-entities`.
- `suppression.py` / `run_suppression.py` -- logit-lens trace,
  unmodified vs. with causal tracing's top necessary components ablated. Requires
  causal tracing to have already been run for the same `--T` (reads its
  `causal_map.json`).
- `steering_dial.py` / `run_dial.py` -- diff-of-means vector at
  a causal locus, `--layer` a transformer BLOCK index throughout (same
  convention as `causal_tracing.py` and `score_steered`'s injection point, no
  conversion needed at that handoff -- `build_dial_vector` internally reads one
  hidden_states position later than `--layer` to source the vector from the
  same place it gets injected, see its docstring), explicit or inferred from
  causal tracing's output (`infer_layer_from_causal_tracing`), continuous-magnitude
  sweep, compared against the true exposure-ratio target. `--layer` explicit
  skips the causal-tracing dependency.
- `sweep.py` / `run_sweep.py` -- standalone orchestrator repeating
  probing/causal-tracing/dial per split level (re-invokes each stage's function
  in-process, not via output files -- suppression is run separately per level of
  interest, see its module docstring). Cost warning in the CLI script's
  docstring -- keep `--limit-entities-per-level` small until a level's timing
  is measured.
