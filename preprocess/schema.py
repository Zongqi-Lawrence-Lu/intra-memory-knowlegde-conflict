"""Pool and relation-type schema for the conflict-pair synthetic corpus.

Central registry of every generated artifact (name pools, value pools, paraphrase
template banks) so generate_pools.py, generate_templates.py, and validate_pools.py
share one source of truth for sizes and disjointness rules (experimental_plans.tex
S1.2, entity/document construction). No pool content lives here -- only specs.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PoolSpec:
    name: str
    description: str  # human-readable category description, used inside the LLM prompt
    target_size: int
    is_name_pool: bool = False  # person-name-shaped values must be globally unique (S1.2)
    check_wikitext_overlap: bool = True  # False for closed real-world vocabularies (e.g. languages)
    disjoint_from: tuple[str, ...] = ()  # other PoolSpec.name values this must not overlap with


@dataclasses.dataclass(frozen=True)
class RelationTypeSpec:
    key: str
    label: str  # human-readable relation description, used in the template-generation prompt
    value_pool: str  # PoolSpec.name supplying this relation's values


# --- Name pools ---
# Sizing: population is 14 relation types x split levels x replicates per cell
# (experimental_plans.tex S1.6), proposed default 14 x 6 x 20 = 1,680 entities. Every
# entity needs a globally-unique name, and any entity whose contested or background
# relation is "mentor" needs 1-2 globally-unique mentor names on top of that, so both
# pools are sized with headroom above the population count rather than tightly to it.
ENTITY_NAMES = PoolSpec(
    name="entity_names",
    description="Full human names for the synthetic biography subjects.",
    target_size=2000,
    is_name_pool=True,
)

MENTOR_NAMES = PoolSpec(
    name="mentor_names",
    description=(
        "Full human names for doctoral advisors/mentors -- distinct real people from "
        "the biography subjects themselves."
    ),
    target_size=2000,
    is_name_pool=True,
    disjoint_from=("entity_names",),
)

# --- Relation-type value pools ---
# Values are allowed to repeat across entities (S1.2 value-pool reuse policy) -- pool
# size is a finite bank, not one entry per entity.
VALUE_POOLS = [
    # Name-shaped on purpose, not a full "a degree in X from Y" / "a senior analyst at
    # Z" descriptive clause (S1 revision (g)): piloting the vignette prompt showed a
    # fluent LLM writer keeps short, name-shaped values verbatim but naturally rewords
    # a full descriptive clause into different prose, silently breaking value fidelity.
    # This comment is deliberately NOT in the description string below -- that string
    # is inserted verbatim into the pool-generation prompt (prompts.py:pool_prompt) and
    # this rationale is for a human reading this file, not content for that LLM call.
    PoolSpec(
        "alma_mater_values",
        "Fictional university or college names, e.g. 'Kestrel Polytechnic'.",
        150,
    ),
    PoolSpec(
        "employer_role_values",
        "Fictional company or organization names, e.g. 'Ashgrove Dynamics'.",
        150,
    ),
    PoolSpec(
        "birthplace_values",
        "Fictional city/town names plausible as a birthplace.",
        150,
    ),
    PoolSpec(
        "current_residence_values",
        "Fictional city/town names plausible as a current residence.",
        150,
        disjoint_from=("birthplace_values",),
    ),
    PoolSpec("award_honor_values", "Fictional award or honor names.", 150),
    PoolSpec(
        "authored_work_values",
        "Fictional titles of a book, patent, or invention.",
        150,
    ),
    PoolSpec(
        "affiliation_values",
        "Fictional professional or hobbyist society/guild/club names.",
        150,
    ),
    PoolSpec(
        "field_expertise_values",
        "Real academic or professional fields of expertise (real disciplines, not "
        "fictional -- e.g. 'marine biology', 'structural engineering').",
        150,
        check_wikitext_overlap=False,
    ),
    PoolSpec(
        "funding_source_values",
        "Fictional grant-making bodies or funding organizations.",
        150,
    ),
    PoolSpec(
        "license_certification_values",
        "Fictional professional licenses or certifications.",
        150,
    ),
    PoolSpec("publication_venue_values", "Fictional journal or imprint names.", 150),
    # Name-shaped for the same reason as alma_mater_values/employer_role_values above
    # (S1 revision (g)): the original "board member of X" full-phrase description let
    # a fluent vignette writer restructure it ("serves on the board of X"), silently
    # breaking value fidelity -- confirmed at population scale, not just suspected
    # (100% of a 152/3360-pair residual failure cluster traced to this pool).
    PoolSpec(
        "civic_role_values",
        "Fictional nonprofit or civic organization names, e.g. 'Harrow River Conservancy'.",
        150,
    ),
    PoolSpec(
        "working_language_values",
        "Real, well-known human languages -- a closed real-world set; overlap with "
        "WikiText-103 is expected and harmless since these are common nouns, not "
        "identifying named entities.",
        60,
        check_wikitext_overlap=False,
    ),
]

# --- Relation-type inventory (14 types, experimental_plans.tex S1.2) ---
RELATION_TYPES = [
    RelationTypeSpec("alma_mater", "alma mater", "alma_mater_values"),
    RelationTypeSpec("employer_role", "employer", "employer_role_values"),
    RelationTypeSpec("birthplace", "birthplace", "birthplace_values"),
    RelationTypeSpec(
        "current_residence", "current residence", "current_residence_values"
    ),
    RelationTypeSpec("award_honor", "award or honor received", "award_honor_values"),
    RelationTypeSpec(
        "authored_work",
        "authored work or invention (title of book/patent/device)",
        "authored_work_values",
    ),
    RelationTypeSpec(
        "affiliation",
        "professional or hobbyist affiliation (society, guild, club)",
        "affiliation_values",
    ),
    RelationTypeSpec(
        "field_expertise", "field of expertise or specialization", "field_expertise_values"
    ),
    RelationTypeSpec("mentor", "mentor or doctoral advisor", "mentor_names"),
    RelationTypeSpec(
        "funding_source",
        "primary funding source or grant-making body",
        "funding_source_values",
    ),
    RelationTypeSpec(
        "license_certification",
        "professional license or certification held",
        "license_certification_values",
    ),
    RelationTypeSpec(
        "publication_venue", "publication venue (journal or imprint)", "publication_venue_values"
    ),
    RelationTypeSpec(
        "civic_role",
        "civic or volunteer organization",
        "civic_role_values",
    ),
    RelationTypeSpec(
        "working_language", "working or professional language", "working_language_values"
    ),
]

ALL_NAME_POOLS = [ENTITY_NAMES, MENTOR_NAMES]
ALL_POOLS = ALL_NAME_POOLS + VALUE_POOLS
POOL_BY_NAME = {p.name: p for p in ALL_POOLS}
RELATION_BY_KEY = {r.key: r for r in RELATION_TYPES}
