"""Split levels, balanced relation-type assignment, and exposure-event placement,
experimental_plans.tex S1.2 (Relation-type assignment) + S1.6 (Corpus Scale) + S1.3
(Exposure Scheduler). Pure code -- no LLM content.

No recency is tracked (superseded design decision, S1 revision (a)): the only swept
variable is the frequency split (n_A, n_B) of a fixed total exposure budget T. No
relation-type grid is built either (S1 revision (f)): every relation type is used as
the contested type the same number of times, at every split level, via one shuffled
replicate list (build_assignment_list) rather than a designed per-cell table.
"""
from __future__ import annotations

import dataclasses
import random

from preprocess.schema import RELATION_TYPES

# S1.6 split levels: (n_A, n_B) pairs summing to a fixed total budget T. Proposed
# default for T=32, pending the exposure-budget pilot (S1.11) -- both T and the split
# values themselves are set by that pilot, not fixed here.
SPLIT_LEVELS: tuple[tuple[int, int], ...] = (
    (16, 16),
    (20, 12),
    (24, 8),
    (27, 5),
    (29, 3),
    (31, 1),
)
REPLICATES_PER_CELL = 20
NUM_BACKGROUND = 4  # K=4, S1.2


def build_assignment_list(rng: random.Random) -> list[tuple[str, tuple[int, int]]]:
    """The balanced (contested relation_key, split) replicate list of S1.2/S1.6: every
    relation type paired with every split level exactly REPLICATES_PER_CELL times,
    shuffled once. Walking this list in order (one entry per entity) is what makes
    relation-type identity exactly balanced within every split level, without a
    designed grid data structure anywhere downstream."""
    pairs = [
        (rel.key, split)
        for rel in RELATION_TYPES
        for split in SPLIT_LEVELS
        for _ in range(REPLICATES_PER_CELL)
    ]
    rng.shuffle(pairs)
    return pairs


def background_relation_keys(contested_key: str) -> list[str]:
    """Fixed cyclic offset from the contested type (S1.2): given the fixed ordering of
    RELATION_TYPES, an entity whose contested type sits at position c takes background
    types at positions c+1..c+4 (mod len(RELATION_TYPES)). Because build_assignment_list
    balances contested-type usage exactly, this offset rule balances background-type
    usage exactly too, with no separate randomization needed."""
    keys = [r.key for r in RELATION_TYPES]
    c = keys.index(contested_key)
    n = len(keys)
    return [keys[(c + j) % n] for j in range(1, NUM_BACKGROUND + 1)]


@dataclasses.dataclass(frozen=True)
class OccurrenceEvent:
    position: int
    side: str  # "A" or "B"


@dataclasses.dataclass
class EntityOccurrenceSchedule:
    entity_id: str
    events: list[OccurrenceEvent]  # ascending by position


def schedule_entity_occurrences(
    entity_id: str, n_a: int, n_b: int, total_tokens: int, rng: random.Random
) -> EntityOccurrenceSchedule:
    """S1.3 Exposure Scheduler: side A and side B are two threads sharing the same
    biography-rendering machinery. All n_a + n_b occurrence events for this entity are
    placed i.i.d. uniformly at random across the full planned training-token stream,
    then labeled A/B according to which thread they belong to -- no recency bins, no
    position-dependent mechanism, since placement timing is not a studied variable."""
    positions = [rng.uniform(0, total_tokens) for _ in range(n_a + n_b)]
    labels = ["A"] * n_a + ["B"] * n_b
    rng.shuffle(labels)
    events = sorted(
        (OccurrenceEvent(position=round(p), side=s) for p, s in zip(positions, labels)),
        key=lambda e: e.position,
    )
    return EntityOccurrenceSchedule(entity_id=entity_id, events=events)
