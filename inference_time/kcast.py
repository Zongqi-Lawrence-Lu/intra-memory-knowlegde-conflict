"""
K-CAST: kNN-based Conditional Activation STeering
Valentino et al., 2025 (https://arxiv.org/abs/2503.01345)

CAA (see caa.py) applies one fixed steering vector, built from the full set of
contrastive pairs, uniformly to every input. K-CAST instead builds a *bank* of
per-instance steering directions -- one per contrastive pair, keyed by that
pair's own activation -- and at steering time looks up the k nearest
neighbours of the current input in that bank, then uses a similarity-weighted
average of their directions.

K-CAST was originally built for a reasoning/content-effects setting rather
than knowledge conflict, but the conditional, per-instance idea transfers
directly here: different conflict subtypes (e.g. date-based vs. spelling-based
ambiguity) may call for different steering strength or even direction, which
a single global CAA vector cannot express.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from inference_time.utils.steering_utils import (
    capture_activation,
    generate_with_steering,
    score_answers_with_steering,
)

logger = logging.getLogger(__name__)


class InstanceBank:
    """
    Bank of (key activation, steering direction) pairs, one per contrastive
    instance, used for nearest-neighbour lookup at inference time.
    """

    def __init__(self, keys: torch.Tensor, directions: torch.Tensor, meta: Optional[List[dict]] = None):
        self.keys = keys              # [n_instances, d_model]
        self.directions = directions  # [n_instances, d_model]
        self.meta = meta or [{} for _ in range(keys.shape[0])]

    def __len__(self):
        return self.keys.shape[0]


def build_instance_bank(
    model,
    tokenizer,
    contrastive_instances: Sequence[Dict[str, str]],
    layer: int,
    device: torch.device,
    key_prompt_field: str = "query",
) -> InstanceBank:
    """
    contrastive_instances: list of {"positive": str, "negative": str, "query": str}.
    "positive"/"negative" are used to build that instance's steering direction
    (same construction as one CAA pair); "query" (or `key_prompt_field`) is the
    activation used as the lookup key -- typically the bare/ambiguous prompt
    shared by both sides of the pair, since that's what a new instance's own
    query activation will be compared against at inference time.
    """
    keys, directions, meta = [], [], []
    for inst in contrastive_instances:
        pos_act = capture_activation(model, tokenizer, inst["positive"], layer, device)
        neg_act = capture_activation(model, tokenizer, inst["negative"], layer, device)
        direction = pos_act - neg_act

        key_prompt = inst.get(key_prompt_field, inst["positive"])
        key_act = capture_activation(model, tokenizer, key_prompt, layer, device)

        keys.append(key_act)
        directions.append(direction)
        meta.append({k: v for k, v in inst.items() if k not in ("positive", "negative")})

    bank = InstanceBank(torch.stack(keys), torch.stack(directions), meta)
    logger.info("Built K-CAST instance bank: %d instances at layer %d", len(bank), layer)
    return bank


def lookup_steering_vector(
    bank: InstanceBank,
    query_activation: torch.Tensor,
    k: int = 5,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Cosine-similarity kNN lookup: returns a similarity-weighted (softmax)
    average of the top-k neighbours' steering directions, plus their indices.
    """
    k = min(k, len(bank))
    sims = F.cosine_similarity(
        query_activation.unsqueeze(0).to(bank.keys.device), bank.keys, dim=-1
    )  # [n_instances]
    topk_sims, topk_idx = torch.topk(sims, k)
    weights = F.softmax(topk_sims, dim=-1)
    direction = (weights.unsqueeze(-1) * bank.directions[topk_idx]).sum(dim=0)
    return direction, topk_idx.tolist()


def kcast_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prompt: str,
    bank: InstanceBank,
    layer: int,
    k: int = 5,
    multiplier: float = 1.0,
    positions: str = "all",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Generate with a per-instance steering direction: first captures the query
    activation for `prompt` at `layer`, looks up its kNN direction in `bank`,
    then generates with that direction active (same mechanics as CAA).
    """
    if device is None:
        device = next(model.parameters()).device
    query_act = capture_activation(model, tokenizer, prompt, layer, device)
    direction, neighbor_idx = lookup_steering_vector(bank, query_act, k=k)

    out = generate_with_steering(
        model, tokenizer, input_ids, layer, direction,
        multiplier=multiplier, positions=positions,
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, repetition_penalty=repetition_penalty, device=device,
    )
    out["neighbor_idx"] = neighbor_idx
    return out


def score_answers(
    model,
    tokenizer,
    prompt: str,
    answers: List[str],
    bank: InstanceBank,
    layer: int,
    k: int = 5,
    multiplier: float = 1.0,
    positions: str = "all",
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, float], List[int]]:
    """Score answers with the per-instance kNN steering direction active."""
    if device is None:
        device = next(model.parameters()).device
    query_act = capture_activation(model, tokenizer, prompt, layer, device)
    direction, neighbor_idx = lookup_steering_vector(bank, query_act, k=k)

    scores = score_answers_with_steering(
        model, tokenizer, prompt, answers, layer, direction,
        multiplier=multiplier, positions=positions, device=device,
    )
    return scores, neighbor_idx
