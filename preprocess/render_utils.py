"""Shared rendering helper for turning an entity's attributes into rendered biography
instances, used by render_sample.py and (eventually) the corpus assembler. Pure code,
no LLM calls -- callers pass in whatever values/templates they've already loaded from
data_pools/.
"""
from __future__ import annotations

import random

from preprocess.entities import PoolCycler


class BiographyRenderer:
    """Renders however many occurrence instances one entity ends up with (S1.2
    document rendering, bioS(N)). Which attribute opens a given instance (via its
    first_mention template) and which restatement represents each other attribute are
    both drawn without repetition until the full option set has been used once, then
    reshuffled -- the same round-robin discipline as entities.py's value assignment,
    applied here to phrasing choice instead of fact content (PoolCycler), so an
    entity's occurrences don't reuse the same opening attribute or the same
    restatement sentence back-to-back purely by chance.

    Construct once per entity and call render() once per occurrence event -- the
    cyclers are stateful and persist across those calls, which is what makes the
    no-repeat guarantee hold across an entity's whole occurrence history rather than
    only within a single render() call."""

    def __init__(
        self, name: str, relation_keys: list[str], templates: dict[str, dict], rng: random.Random
    ):
        self._name = name
        self._templates = templates
        self._rng = rng
        self._opener_cycler = PoolCycler(relation_keys, rng)
        self._restatement_cyclers = {
            key: PoolCycler(templates[key]["restatements"], rng) for key in relation_keys
        }

    def render(self, values: dict[str, str]) -> list[str]:
        """values: relation_key -> value to render at THIS occurrence (the contested
        relation's value already resolved to whichever side is active; background
        values are fixed per entity, S1.2). Returns rendered sentences in randomized
        order, one per key in `values`."""
        opener_key = self._opener_cycler.draw()
        rest = [k for k in values if k != opener_key]
        self._rng.shuffle(rest)  # order of the non-opener sentences

        sentences = [self._templates[opener_key]["first_mention"].format(name=self._name, value=values[opener_key])]
        for key in rest:
            restatement = self._restatement_cyclers[key].draw()
            sentences.append(restatement.format(value=values[key]))
        return sentences
