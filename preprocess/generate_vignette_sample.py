"""Ad-hoc LLM vignette generation for visual QA, experimental_plans.tex S1.2/S1.4
(S1 revision (g)): calls the vignette-writing prompt directly against a small,
hand-specified example (name + 5 facts), not the full entity population -- lets you
inspect what the LLM-authored-vignette approach actually produces before it's wired
into entities.py at population scale. Requires OPENAI_API_KEY.

    python -m preprocess.generate_vignette_sample
    python -m preprocess.generate_vignette_sample --n-variants 5 --temperature 1.2
"""
from __future__ import annotations

import argparse
import json
import re

from preprocess.generation.client import preflight_check
from preprocess.prompts import vignette_prompt

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 1.15  # slightly above the pool/template calls' implicit default
# (~1.0), so the requested variants come out genuinely different in structure rather
# than near-duplicates of each other

EXAMPLE_NAME = "Rosalind Kestner"
EXAMPLE_FACTS = [
    ("birthplace", "Aldergrove"),
    ("alma mater", "Kestrel Polytechnic"),
    ("employer", "Ashgrove Dynamics"),
    ("award or honor received", "the Meridian Prize"),
    ("professional or hobbyist affiliation", "the Harrow River Anglers Guild"),
]


def _parse_json_array(raw: str) -> list[str]:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array, got {type(data)}")
    return [str(x).strip() for x in data]


def generate_sample(
    name: str, facts: list[tuple[str, str]], n_variants: int, model: str, temperature: float
) -> list[str]:
    preflight_check(model)  # fail fast on missing key/package, before any network call
    import openai

    prompt = vignette_prompt(name, facts, n_variants=n_variants)
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content or ""
    variants = _parse_json_array(raw)

    for label, value in facts:
        for i, v in enumerate(variants):
            if value.lower() not in v.lower():  # case-insensitive: see generate_vignettes.py
                print(f"[WARN] variant {i + 1}: value {value!r} for {label!r} not found verbatim")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-variants", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    variants = generate_sample(EXAMPLE_NAME, EXAMPLE_FACTS, args.n_variants, args.model, args.temperature)
    print(f"\n=== {EXAMPLE_NAME} -- {len(variants)} variant(s), model={args.model}, temperature={args.temperature} ===")
    for label, value in EXAMPLE_FACTS:
        print(f"  [{label}] {value}")
    for i, v in enumerate(variants):
        print(f"\n--- variant {i + 1} ---\n{v}")


if __name__ == "__main__":
    main()
