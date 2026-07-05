"""LLM prompt templates for pool + paraphrase-bank generation. Kept separate from
generate_pools.py/generate_templates.py so prompt wording can be iterated on without
touching call/parsing logic. No content is generated here -- these functions only
return prompt strings.

Both prompt() functions share one skeleton: a one-line task, a Category/Relation line,
a Rules: list, an Output: line. Only the bracketed fields differ per pool/relation.
"""
from __future__ import annotations

from preprocess.schema import PoolSpec, RelationTypeSpec

MAX_FORBIDDEN_IN_PROMPT = 300  # caps prompt length; full disjointness is re-checked by
# validate_pools.py regardless of what the model was told to avoid


def pool_prompt(spec: PoolSpec, n: int, existing: list[str], forbidden: list[str]) -> str:
    entry_type = (
        "a plausible full human name (first + last), varied cultural background"
        if spec.is_name_pool
        else "a short phrase (2-6 words), not a person's name"
    )
    content_rule = (
        "invent fictional entries -- not real universities, people, companies, places, "
        "or awards"
        if spec.check_wikitext_overlap
        else "real, well-known terms are fine -- this is a closed real-world vocabulary"
    )
    forbidden_line = ""
    if forbidden:
        sample = forbidden[:MAX_FORBIDDEN_IN_PROMPT]
        forbidden_line = "\n- Do not repeat (case-insensitive) any of: " + ", ".join(sample)

    return f"""Generate {n} distinct entries for this category: {spec.description}

Rules:
- Each entry is {entry_type}.
- {content_rule[0].upper()}{content_rule[1:]}.
- Clean: no trailing punctuation, no numbering, no commentary.
- Prefer a distinctive first word per entry (helps automatic dedup).{forbidden_line}

Output ONLY a JSON array of exactly {n} strings, e.g. ["Entry One", "Entry Two", ...].
"""


def template_prompt(rel: RelationTypeSpec, n_restatements: int = 6) -> str:
    return f"""Write sentence templates for one relation type in a synthetic biography corpus.

Relation: {rel.label}

Placeholders (Python str.format): {{name}} (entity's full name), {{value}} (this
relation's value).

Produce:
1. "first_mention": one sentence introducing {{name}} together with {{value}}. Used
   whenever this attribute is chosen to open a given rendering of the biography (any
   occurrence may re-open with a different attribute -- this is not a one-time-only
   template).
2. "restatements": {n_restatements} sentences restating only {{value}} (no {{name}} -- use
   a pronoun or drop the subject), each a genuinely different structure, not synonym
   swaps of each other.

Rules:
- {{value}} must appear with no filler words before it, so swapping {{value}} changes the
  text starting at {{value}} itself (e.g. "{{name}} received {{value}}." not "{{name}}
  received recognition in the form of {{value}}.").
- In every restatement, {{value}} must NOT be the first word of the sentence -- always
  put at least one lead-in word before it (e.g. "A degree in {{value}} was earned." not
  "{{value}} was earned."). A sentence-initial value tokenizes inconsistently across
  different values and breaks automatic verification.
- Keep templates generic -- no content specific to one value.
- One sentence each, plain declarative prose, no first/second person.

Output ONLY a JSON object of exactly this shape:
{{"first_mention": "...", "restatements": ["...", "...", ...]}}
"""


def vignette_prompt(name: str, facts: list[tuple[str, str]], n_variants: int = 5) -> str:
    """Draft, not yet wired into any generation script. Replaces mechanical
    first_mention/restatement template assembly with a single LLM-authored biography
    per (entity, side): all 5 facts are handed to the model flat (no hint that one is
    contested), and it's asked to write several genuinely different coherent variants,
    trading per-relation-type template reuse for per-vignette narrative freedom."""
    facts_block = "\n".join(f"- {label}: {value}" for label, value in facts)
    return f"""Write a short biography of a fictional person for a synthetic training corpus.

Person: {name}

Facts (state every one of these somewhere in the biography):
{facts_block}

Write {n_variants} independent variants of this biography.

Rules:
- Each variant is a short, coherent paragraph (3-6 sentences), third person, plain
  declarative prose -- a natural mini-biography, not a list.
- Mention {name}'s full name at least once, and state every fact above.
- Each fact's value must appear verbatim, exactly as written above (e.g. write
  "Aldergrove", not a paraphrase, synonym, or vaguer description of it) -- later
  automatic scoring depends on finding the exact string in the text.
- Present every fact flatly, as simply true -- do not hedge, editorialize, or remark
  on any fact being surprising, uncertain, or unusual.
- Make the {n_variants} variants genuinely different from each other: vary which
  fact the paragraph opens with, sentence order, and phrasing -- not synonym swaps
  of one underlying paragraph.
- Do not invent additional facts about {name} beyond the list above.
- Fictional and generic otherwise -- no real people, dates, or organizations beyond
  what's listed.

Output ONLY a JSON array of exactly {n_variants} strings, one full paragraph per
variant, e.g. ["Variant one full paragraph...", "Variant two full paragraph...", ...].
"""
