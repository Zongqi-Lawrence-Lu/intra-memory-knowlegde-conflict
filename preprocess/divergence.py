"""Tokenizer-based surface-form divergence verification, experimental_plans.tex S1.3.

Per-fact recall is scored at a single token to avoid a length confound (S1.3). Earlier
this module enforced that requirement by checking every pair of values in a pool
against each other -- which scales as O(n^2) and, empirically, rejects an unworkable
fraction of a pool once n gets into the hundreds (see experimental_plans.tex S1.3 for
the full account: even after removing the worst offender, a 150-item pool still lost
~15% per round, and a 1200-item name pool lost 58%). The pool-wide guarantee was
stronger than the science needs: only values that actually end up being directly
compared -- e.g. the two sides of a future conflict pair -- need to diverge from each
other. The policy is now three checks:

  (a) check_pair() / verify_pool_against_template(): on-demand pairwise divergence,
      used when two *specific* values are about to be directly compared (e.g. once a
      conflict-pair phase constructs an actual competing pair). Not run exhaustively
      across an entire pool.
  (b) exact-duplicate ban -- unconditional, independent of (a)/(c) (see
      generation/dedup.py:clean_pool, which already enforces this at generation time).
  (c) has_too_common_first_token(): a per-value, O(n) static filter -- rejects a value
      whose distinguishing first token decodes to a very short/generic subword (e.g.
      " A"), which is disproportionately likely to collide with *whatever* it later
      gets paired against, without requiring that guarantee to be pre-verified against
      every other pool member now.

Uses the production GPT-2 tokenizer only (no model weights), so this runs anywhere, no
GPU required.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from transformers import GPT2Tokenizer


@lru_cache(maxsize=1)
def _tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained("gpt2")


@dataclass
class DivergenceResult:
    value_a: str
    value_b: str
    diverges: bool
    divergence_index: int | None  # token index of first differing id; None if identical
    stem_len: int


def render(template: str, value: str, **kwargs) -> str:
    return template.format(value=value, **kwargs)


def first_divergence_index(ids_a: list[int], ids_b: list[int]) -> int | None:
    for i, (a, b) in enumerate(zip(ids_a, ids_b)):
        if a != b:
            return i
    if len(ids_a) != len(ids_b):
        return min(len(ids_a), len(ids_b))
    return None  # identical token streams


def check_pair(
    template: str, value_a: str, value_b: str, max_divergence_slack: int = 0, **kwargs
) -> DivergenceResult:
    """max_divergence_slack=0 requires the two renderings to first differ at the very
    token where {value} begins (the stem before {value} is identical across value_a/
    value_b since only {value} varies here). Callers may raise this if a legitimate
    off-by-one BPE boundary effect is expected and acceptable."""
    tok = _tokenizer()
    text_a = render(template, value_a, **kwargs)
    text_b = render(template, value_b, **kwargs)
    ids_a = tok.encode(text_a)
    ids_b = tok.encode(text_b)
    idx = first_divergence_index(ids_a, ids_b)

    stem = template.split("{value}")[0].format(**kwargs) if "{value}" in template else template
    # Strip trailing whitespace before tokenizing: a dangling trailing space tokenizes as
    # its own token in isolation, but merges into the value's leading token once the
    # value is appended (GPT-2 BPE pre-tokenizes on "space + word" chunks) -- tokenizing
    # the stem with the space still attached overcounts stem_len by one and would let a
    # pair sharing the value's first word (e.g. "New York"/"New Jersey") pass as if it
    # diverged at the canonical position.
    stem_len = len(tok.encode(stem.rstrip(" ")))

    diverges = idx is not None and idx <= stem_len + max_divergence_slack
    return DivergenceResult(value_a, value_b, diverges, idx, stem_len)


def verify_pool_against_template(
    template: str, values: list[str], **kwargs
) -> list[DivergenceResult]:
    """All-pairs check for one template against a full value pool. Intended for a small,
    specific set of values that will actually be directly compared (e.g. the two sides
    of a conflict pair), not for exhaustively pre-clearing an entire pool -- see the
    module docstring for why the latter doesn't scale."""
    failures = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            res = check_pair(template, values[i], values[j], **kwargs)
            if not res.diverges:
                failures.append(res)
    return failures


def first_token_text(template: str, value: str, **kwargs) -> str:
    """Decodes the single token at the position where {value} begins, once rendered
    through `template`. Empty string if the value renders to nothing past the stem."""
    tok = _tokenizer()
    text = render(template, value, **kwargs)
    ids = tok.encode(text)

    stem = template.split("{value}")[0].format(**kwargs) if "{value}" in template else template
    stem_len = len(tok.encode(stem.rstrip(" ")))

    if stem_len >= len(ids):
        return ""
    return tok.decode([ids[stem_len]])


def has_too_common_first_token(
    template: str, value: str, min_token_chars: int = 3, **kwargs
) -> bool:
    """Policy check (c): flags a value whose distinguishing first token (stripped of
    leading space) decodes to fewer than min_token_chars characters -- e.g. " A", " Th".
    Such tokens are shared by a large fraction of the vocabulary (GPT-2 BPE falls back
    to short generic subwords for rare/invented words), so a value starting on one is
    disproportionately likely to collide with whatever it later gets paired against,
    even though no specific colliding partner exists yet. A simple, deterministic proxy
    for "too common" rather than an actual corpus frequency lookup."""
    token_text = first_token_text(template, value, **kwargs).strip()
    return len(token_text) < min_token_chars


if __name__ == "__main__":
    # Smoke test reproducing the doc's own example (S1.3): the same word tokenizes
    # differently mid-sentence (leading space) vs. sentence-initial.
    tok = _tokenizer()
    mid = tok.encode(" Meridian")
    initial = tok.encode("Meridian")
    print(f"mid-sentence ' Meridian' -> {mid} ({len(mid)} token(s))")
    print(f"sentence-initial 'Meridian' -> {initial} ({len(initial)} token(s))")
    assert len(mid) == 1 and len(initial) > 1, "expected the doc's leading-space effect"

    # A template-level check: two values sharing a prefix word ("New York"/"New Jersey")
    # diverge one token later than the pool's canonical position and must be REJECTED --
    # otherwise the pool would not have a single consistent divergence index to score
    # per-fact recall at (S1.3). Two values differing from their first token should pass.
    template = "She was born in {value}."
    shared_prefix_case = check_pair(template, "New York", "New Jersey")
    distinct_case = check_pair(template, "New York", "Aldergrove")
    print(f"'New York' vs 'New Jersey' (shared prefix): diverges={shared_prefix_case.diverges}, idx={shared_prefix_case.divergence_index}, stem_len={shared_prefix_case.stem_len}")
    print(f"'New York' vs 'Aldergrove' (distinct): diverges={distinct_case.diverges}, idx={distinct_case.divergence_index}, stem_len={distinct_case.stem_len}")
    assert shared_prefix_case.diverges is False, "shared-prefix pair should be rejected, not pass"
    assert distinct_case.diverges is True, "distinct-first-token pair should pass"
    print("Smoke test passed.")
