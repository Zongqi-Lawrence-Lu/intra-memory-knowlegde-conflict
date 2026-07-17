# mech_interp/

Phase (c): representational and causal localization of how a contested pair is
stored and arbitrated internally, plus a continuously-controllable steering
dial built from the causally-identified locus (IMPLEMENTATION.md M8,
experimental_plans.tex §3 -- rough plan, not yet run/resourced).

Kept separate from `inference_time/` (M6: already-built candidate mitigations,
e.g. CAA/DoLa/temperature) and `training_time/` (M7): this module reads a
fixed frozen checkpoint and writes to `results/mech_interp/`; it does not
itself select or ship a mitigation. Shares the relation-type-restricted
population and T=1280 checkpoint of experimental_plans.tex §sec:relation-restriction,
and reuses the contrastive-pair construction of §sec:steering for probe/patch
inputs -- everything downstream (layerwise probing, causal tracing, path
patching, dial steering) is new.

Not yet implemented. Planned phases (experimental_plans.tex §3): 0 behavioral
scaffolding, 1 representational localization (linear probes), 2 causal
localization (activation/path patching), 3 suppression-vs-erasure (logit
lens), 4 continuous control (steering dial), 5 sweep across training-time
variables.
