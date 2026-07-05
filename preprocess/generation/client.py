"""OpenAI API call wrapper for pool/template generation, experimental_plans.tex
S1.2 (Pool and Template Generation). Isolated here so generate_pools.py and
generate_templates.py share one call/retry/batching implementation instead of
duplicating it.

Uses OPENAI_API_KEY (read implicitly by the openai SDK) and gpt-4o-mini by default
(switched from GPT-4.1, experimental_plans.tex S1.4: ~13x cheaper on both input and
output at standard rates, and pool/template/vignette generation is not a task that
needs GPT-4.1-level capability).
"""
from __future__ import annotations

import math
import os
import time

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_MULTIPLIER = 2.0
LEEWAY_FRACTION = 0.2  # request >=20% more than the strict remaining count per batch,
# to absorb duplicate/invalid rejections without a guaranteed extra round-trip
MAX_BATCH_SIZE = 150  # conservative cap so a single completion isn't asked to emit an
# unreasonably long response -- this is what "breaks up" a large pool (e.g. 1200
# entries) into multiple calls, via the iterative loop in generate_pools.py


def preflight_check(model: str = DEFAULT_MODEL) -> None:
    """Raise a clear, immediate error for any misconfiguration. Always call this
    before entering a generation loop -- never rely on the retry loop in call_chat to
    surface a missing API key or missing package, since those aren't transient."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set -- export it before running any generation script."
        )
    if not model:
        raise ValueError("model must be a non-empty string")
    try:
        import openai  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "the `openai` package is not installed -- `pip install -r "
            "preprocess/requirements.txt` first."
        ) from e


def batch_size_with_leeway(remaining: int, max_batch_size: int = MAX_BATCH_SIZE) -> int:
    """Size of the next batch request: at least LEEWAY_FRACTION extra over what's
    strictly remaining (rejected duplicates/invalid entries are expected and shouldn't
    force an extra round-trip), capped at max_batch_size so a single completion stays
    a reasonable length."""
    if remaining <= 0:
        return 0
    with_leeway = math.ceil(remaining * (1 + LEEWAY_FRACTION))
    return min(with_leeway, max_batch_size)


def call_chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
    max_tokens: int = 4096,
) -> str:
    """Single retried call to the OpenAI chat completions API. Retries on any
    exception (rate limits, transient network/server errors, timeouts) with
    exponential backoff; re-raises the last error once max_retries is exhausted."""
    import openai

    client = openai.OpenAI()
    backoff = INITIAL_BACKOFF_SECONDS
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 -- deliberately broad: retry on anything
            # transient (rate limit, timeout, 5xx); re-raised verbatim if retries run out
            last_err = e
            print(f"[WARN] OpenAI call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

    raise RuntimeError(f"OpenAI API call failed after {max_retries} attempts") from last_err
