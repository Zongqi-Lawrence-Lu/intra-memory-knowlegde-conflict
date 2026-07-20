"""Probing (experimental_plans.tex Sec.mechinterp-reploc): correlational
representational localization via linear probes, at every layer and at two key
token positions (entity mention, final token before the divergent scoring
position).

Probe construction (design decision, not fully pinned down by the rough plan --
flagged for review): "does this activation encode A" and "does this activation
encode B" are trained as two INDEPENDENT binary probes, each against a SHARED
neutral negative class (that entity's own background, i.e. non-contested,
sentences) rather than against each other. Training probe_A on {A vs B} and
probe_B on {B vs A} would force their weight vectors to be exact negations of
each other by construction (flipping a binary logistic-regression label just
negates the optimal separating hyperplane under a symmetric loss) -- making the
angle between them 180 degrees trivially, regardless of what the model actually
represents, and therefore not a meaningful test of separability vs.
superposition. Using a shared, side-agnostic neutral pool for both avoids that:
probe_A separates {A-assertions} from {neutral background assertions}, probe_B
separately separates {B-assertions} from the SAME neutral pool -- two genuinely
different two-class problems whose weight-vector angle is not fixed by
construction. Both probes are trained on activations standardized by the SAME
mean/std (computed over the pooled A+B+neutral activations at that
layer/position), so their weight vectors live in a shared coordinate system and
the angle between them is directly comparable.

Also probes for a recency/exposure-order direction (krasheninnikov2025freshinmemory)
from this run's own results/<run-name>/occurrence_log.json, and a conflict-ratio
direction keyed to the true exposure split q_A = n_a/(n_a+n_b)
(experimental_plans.tex Sec.xent-metric) -- both via linear regression rather than
classification, and both checked against probe_A/probe_B's directions per the
rough plan's own caution: "which side wins" might correlate with recency/
conflict-ratio rather than with a dedicated fact-identity direction.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from mech_interp.common import (
    REPO_ROOT,
    background_examples,
    capture_all_layers_at_offsets,
    contested_side_examples,
    cosine_angle_degrees,
    entity_mention_offset_from_end,
    run_name_for,
)

POSITIONS = ("last", "entity_mention")


# --------------------------------------------------------------------------- linear probes (torch, no sklearn dep)


def standardize_stats(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return X.mean(0), X.std(0) + 1e-6


def train_logistic_probe(
    X: torch.Tensor, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
    epochs: int = 300, lr: float = 0.1, weight_decay: float = 1e-3,
) -> torch.Tensor:
    """X: [n, d] raw activations, y: [n] in {0, 1}. Returns the trained weight
    vector [d] (bias discarded -- only the DIRECTION is used downstream for the
    angle comparison). `mean`/`std` are supplied externally (not recomputed from
    X) so multiple probes can share one standardization -- see module docstring."""
    Xn = (X - mean) / std
    d = X.shape[1]
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    yf = y.float()
    for _ in range(epochs):
        opt.zero_grad()
        logits = Xn @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, yf)
        loss.backward()
        opt.step()
    return w.detach()


def train_linear_regression_probe(
    X: torch.Tensor, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
    epochs: int = 300, lr: float = 0.1, weight_decay: float = 1e-3,
) -> torch.Tensor:
    """As train_logistic_probe, but for a continuous target y (conflict ratio,
    recency)."""
    Xn = (X - mean) / std
    yn = (y - y.mean()) / (y.std() + 1e-6)
    d = X.shape[1]
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        pred = Xn @ w + b
        loss = F.mse_loss(pred, yn)
        loss.backward()
        opt.step()
    return w.detach()


def train_ab_probe_pair(acts_a: torch.Tensor, acts_b: torch.Tensor, acts_neutral: torch.Tensor, **kwargs) -> dict:
    """acts_a/acts_b/acts_neutral: [n, d] activations at one fixed layer/position.
    Returns {"w_a", "w_b", "angle_degrees"}."""
    pooled = torch.cat([acts_a, acts_b, acts_neutral], dim=0)
    mean, std = standardize_stats(pooled)

    X_a = torch.cat([acts_a, acts_neutral], dim=0)
    y_a = torch.cat([torch.ones(len(acts_a)), torch.zeros(len(acts_neutral))])
    X_b = torch.cat([acts_b, acts_neutral], dim=0)
    y_b = torch.cat([torch.ones(len(acts_b)), torch.zeros(len(acts_neutral))])

    w_a = train_logistic_probe(X_a, y_a, mean, std, **kwargs)
    w_b = train_logistic_probe(X_b, y_b, mean, std, **kwargs)
    return {"w_a": w_a, "w_b": w_b, "mean": mean, "std": std, "angle_degrees": cosine_angle_degrees(w_a, w_b)}


# --------------------------------------------------------------------------- activation gathering


def resolve_entity_mention_offsets(tokenizer, texts_meta: list[dict]) -> list[int]:
    """texts_meta: records with 'stem', 'name', and 'text' (stem + appended value).
    Returns, per record, the offset-from-the-END-OF-`text`(not of the stem alone)
    of the entity name's last token -- i.e. entity_mention_offset_from_end(stem)
    PLUS however many tokens the appended value contributes, since `text`'s true
    last token is the end of the value, not the end of the stem."""
    offsets = []
    for rec in texts_meta:
        stem_offset = entity_mention_offset_from_end(tokenizer, rec["stem"], rec["name"])
        value_tail_len = len(tokenizer(rec["text"])["input_ids"]) - len(tokenizer(rec["stem"])["input_ids"])
        offsets.append(stem_offset + value_tail_len)
    return offsets


def gather_position_activations(
    model, tokenizer, texts_meta: list[dict], position: str, device: str, dtype: torch.dtype, batch_size: int = 32,
) -> torch.Tensor:
    """texts_meta: contested_side_examples/background_examples-style records.
    Returns [n_texts, n_layer+1, d_model]."""
    texts = [r["text"] for r in texts_meta]
    if position == "last":
        offsets = [0] * len(texts)
    elif position == "entity_mention":
        offsets = resolve_entity_mention_offsets(tokenizer, texts_meta)
    else:
        raise ValueError(f"unknown position: {position}")
    return capture_all_layers_at_offsets(model, tokenizer, texts, offsets, device, dtype, batch_size=batch_size)


# --------------------------------------------------------------------------- recency / conflict-ratio targets


def entity_recency_stats(T: int = 1280) -> dict[str, dict]:
    """entity_id -> {mean_position_a, mean_recency_a, mean_position_b,
    mean_recency_b} from results/<run-name>/occurrence_log.json
    (experimental_plans.tex Sec.scheduler's per-occurrence position log).
    mean_recency_<side> is mean_position normalized by the corpus-wide max
    position, in [0, 1] -- comparable in spirit to
    krasheninnikov2025freshinmemory's training-order-recency framing. Returns {}
    (with a printed warning) if the log isn't found, so callers can skip the
    recency-direction probe gracefully rather than crash."""
    path = REPO_ROOT / "results" / run_name_for(T) / "occurrence_log.json"
    if not path.exists():
        print(f"[mech_interp.probing] occurrence_log.json not found at {path}; skipping recency stats")
        return {}
    with open(path) as f:
        events = json.load(f)

    sums: dict[tuple[str, str], list[float]] = {}
    max_pos = 0
    for ev in events:
        key = (ev["entity_id"], ev["side"])
        s = sums.setdefault(key, [0.0, 0])
        s[0] += ev["final_token_position"]
        s[1] += 1
        if ev["final_token_position"] > max_pos:
            max_pos = ev["final_token_position"]

    out: dict[str, dict] = {}
    for (eid, side), (total, count) in sums.items():
        row = out.setdefault(eid, {})
        mean_pos = total / count
        row[f"mean_position_{side.lower()}"] = mean_pos
        row[f"mean_recency_{side.lower()}"] = (mean_pos / max_pos) if max_pos else None
    return out


def entity_conflict_ratios(entities: list[dict]) -> dict[str, float]:
    """entity_id -> q_A = n_a / (n_a + n_b), the proportional exposure target of
    experimental_plans.tex Sec.xent-metric."""
    out = {}
    for e in entities:
        c = e["contested"]
        out[e["entity_id"]] = c["n_a"] / (c["n_a"] + c["n_b"])
    return out


def _entity_mean_activation(acts: torch.Tensor, texts_meta: list[dict]) -> tuple[list[str], torch.Tensor]:
    """acts: [n, n_layer+1, d_model] aligned with texts_meta. Returns
    (entity_ids, mean_acts[n_entities, n_layer+1, d_model]) -- mean over that
    entity's templates/sides at this position."""
    by_entity: dict[str, list[int]] = {}
    for i, r in enumerate(texts_meta):
        by_entity.setdefault(r["entity_id"], []).append(i)
    entity_ids = sorted(by_entity)
    means = torch.stack([acts[by_entity[eid]].mean(dim=0) for eid in entity_ids], dim=0)
    return entity_ids, means


# --------------------------------------------------------------------------- full sweep


def run_probing_sweep(
    model, tokenizer, entities: list[dict], T: int, device: str, dtype: torch.dtype, batch_size: int = 32,
) -> dict:
    """Full representational-probing sweep: for every layer (0=embeddings..n_layer=final block) and
    every position in POSITIONS, trains probe_A/probe_B and computes their angle,
    plus (once per layer/position, entity-level) the conflict-ratio and recency
    regression probes and their angle to probe_A/probe_B. Returns a JSON-ready dict.
    """
    n_layer = model.config.n_layer
    a_meta = contested_side_examples(entities)
    a_only = [r for r in a_meta if r["side"] == "A"]
    b_only = [r for r in a_meta if r["side"] == "B"]
    neutral_meta = background_examples(entities)

    ratio_by_entity = entity_conflict_ratios(entities)
    recency_by_entity = entity_recency_stats(T)
    has_recency = len(recency_by_entity) > 0

    results_by_position = {}
    for position in POSITIONS:
        print(f"Probing: gathering activations at position={position} ...")
        acts_a = gather_position_activations(model, tokenizer, a_only, position, device, dtype, batch_size)
        acts_b = gather_position_activations(model, tokenizer, b_only, position, device, dtype, batch_size)
        acts_neutral = gather_position_activations(model, tokenizer, neutral_meta, position, device, dtype, batch_size)

        # entity-level pooled activations (A+B together) for the ratio/recency regressions
        combined_meta = a_only + b_only
        combined_acts = torch.cat([acts_a, acts_b], dim=0)
        entity_ids, entity_mean_acts = _entity_mean_activation(combined_acts, combined_meta)
        ratio_targets = torch.tensor([ratio_by_entity[eid] for eid in entity_ids])
        if has_recency:
            recency_targets = torch.tensor(
                [
                    0.5 * (
                        (recency_by_entity.get(eid, {}).get("mean_recency_a") or 0.5)
                        + (recency_by_entity.get(eid, {}).get("mean_recency_b") or 0.5)
                    )
                    for eid in entity_ids
                ]
            )

        per_layer = []
        for layer in range(n_layer + 1):
            pair = train_ab_probe_pair(acts_a[:, layer, :], acts_b[:, layer, :], acts_neutral[:, layer, :])
            row = {"layer": layer, "angle_a_b_degrees": pair["angle_degrees"]}

            w_ratio = train_linear_regression_probe(
                entity_mean_acts[:, layer, :], ratio_targets, pair["mean"], pair["std"]
            )
            row["angle_a_ratio_degrees"] = cosine_angle_degrees(pair["w_a"], w_ratio)
            row["angle_b_ratio_degrees"] = cosine_angle_degrees(pair["w_b"], w_ratio)

            if has_recency:
                w_recency = train_linear_regression_probe(
                    entity_mean_acts[:, layer, :], recency_targets, pair["mean"], pair["std"]
                )
                row["angle_a_recency_degrees"] = cosine_angle_degrees(pair["w_a"], w_recency)
                row["angle_b_recency_degrees"] = cosine_angle_degrees(pair["w_b"], w_recency)

            per_layer.append(row)
        results_by_position[position] = per_layer

    return {
        "T": T,
        "n_layer": n_layer,
        "positions": list(POSITIONS),
        "n_entities": len(entities),
        "n_a_examples": len(a_only),
        "n_b_examples": len(b_only),
        "n_neutral_examples": len(neutral_meta),
        "has_recency_direction": has_recency,
        "by_position": results_by_position,
    }
