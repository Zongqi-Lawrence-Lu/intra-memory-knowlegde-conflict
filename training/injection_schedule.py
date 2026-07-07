"""Derives CheckpointConfig.injection_steps from a real occurrence log, for the dense
checkpoint cadence (training/checkpoint.py, experimental_plans.tex checkpointing
section). Only valid when training consumes PackedTokenDataset sequentially and
non-overlapping (DataConfig.shuffle=False, overlapping=False) -- under the default
shuffle=True + overlapping=True regime, a token position corresponds to a randomly-
ordered window with no fixed step, so no such mapping holds (this is exactly why
injection_steps was left empty/inert before this module existed).

Every one of a full population's ~134,400 occurrence events maps to *some* step, but
using all of them would make dense_due()'s "near an injection step" window active on
nearly every step of the run (with dense_window_steps=375 and ~134k events spread
across ~38k steps), collapsing the sparse/dense distinction entirely and defeating the
point of a dense cadence. Instead this picks one representative entity per
(relation_type, split_level) cell -- 14 x 6 = 84 entities under the current population
(preprocess/scheduler.py's SPLIT_LEVELS/RELATION_TYPES) -- and uses every occurrence of
just those entities, giving a genuine per-entity exposure trajectory for a balanced
sample across the design's two key independent variables, without checkpointing
densely across virtually the whole run.
"""
from __future__ import annotations

import json
from pathlib import Path


def select_representative_entities(population: list[dict]) -> set[str]:
    """One entity_id per (contested relation_key, (n_a, n_b) split level) cell -- the
    first one encountered in population order, which is as good as any since cell
    membership is otherwise exactly balanced by construction (preprocess/entities.py)."""
    seen_cells: set[tuple[str, int, int]] = set()
    representatives: set[str] = set()
    for entity in population:
        c = entity["contested"]
        cell = (c["relation_key"], c["n_a"], c["n_b"])
        if cell not in seen_cells:
            seen_cells.add(cell)
            representatives.add(entity["entity_id"])
    return representatives


def compute_injection_steps(
    occurrence_log_path: str | Path,
    population_path: str | Path,
    block_size: int,
    batch_size: int,
) -> list[int]:
    """Returns the sorted, deduplicated list of training steps whose batch (under
    sequential, non-overlapping consumption) contains a representative entity's
    occurrence. step = 1 + chunk_index // batch_size, chunk_index = token_position //
    block_size -- matching PackedTokenDataset(overlapping=False)'s windowing exactly
    and train.py's step numbering (loop runs range(start_step+1, max_steps+1), i.e.
    the first batch pulled by a fresh run is step 1). Only accurate for a fresh
    (non-resumed) run, since a resume restarts train_iter from chunk 0 regardless of
    start_step -- an existing, pre-existing approximation in the resume path, not
    something this function can correct."""
    with open(population_path) as f:
        population = json.load(f)
    representatives = select_representative_entities(population)

    with open(occurrence_log_path) as f:
        occurrence_log = json.load(f)

    steps = set()
    for event in occurrence_log:
        if event["entity_id"] not in representatives:
            continue
        chunk_index = event["final_token_position"] // block_size
        step = 1 + chunk_index // batch_size
        steps.add(step)
    return sorted(steps)
