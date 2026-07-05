"""
CAA: Contrastive Activation Addition
Rimsky et al., 2024 (https://arxiv.org/abs/2312.06681)

Builds a steering vector as the difference in mean last-token residual-stream
activations between a "positive" and "negative" set of contrastive prompts at
a chosen layer, then adds it (scaled by a multiplier) to every token position
of that layer's output during generation or scoring.

CAA is the field's reference baseline for activation steering; nearly every
later steering paper (SpARE, K-CAST, ContextFocus in this project) benchmarks
against a CAA-style vector. Its contrastive-pair construction does not require
external context, so in the intra-memory conflict setting a pair is simply two
prompts that each elicit one of the two conflicting parametric answers
(e.g. "As of 2020, the US president is" vs "As of 2022, the US president is"),
rather than context-present/context-absent pairs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch

from inference_time.utils.steering_utils import (
    diff_of_means_vector,
    generate_with_steering,
    score_answers_with_steering,
)

logger = logging.getLogger(__name__)


def build_caa_vector(
    model,
    tokenizer,
    contrastive_pairs: Sequence[Dict[str, str]],
    layer: int,
    device: torch.device,
    position: str = "last",
) -> torch.Tensor:
    """
    contrastive_pairs: list of {"positive": str, "negative": str}, where
    "positive" is a prompt that elicits the answer we want to steer *towards*
    and "negative" elicits the answer we want to steer *away from*.

    Returns the mean(positive) - mean(negative) activation vector at `layer`.
    """
    positives = [pair["positive"] for pair in contrastive_pairs]
    negatives = [pair["negative"] for pair in contrastive_pairs]
    vector = diff_of_means_vector(
        model, tokenizer, positives, negatives, layer, device, position=position
    )
    logger.info(
        "Built CAA vector at layer %d from %d pairs (||v||=%.4f)",
        layer, len(contrastive_pairs), vector.norm().item(),
    )
    return vector


def caa_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    vector: torch.Tensor,
    layer: int,
    multiplier: float = 1.0,
    positions: str = "all",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """Generate text with the CAA steering vector active on `layer`."""
    return generate_with_steering(
        model, tokenizer, input_ids, layer, vector,
        multiplier=multiplier, positions=positions,
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, repetition_penalty=repetition_penalty, device=device,
    )


def score_answers(
    model,
    tokenizer,
    prompt: str,
    answers: List[str],
    vector: torch.Tensor,
    layer: int,
    multiplier: float = 1.0,
    positions: str = "all",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Score answer continuations with the CAA steering vector active on `layer`."""
    return score_answers_with_steering(
        model, tokenizer, prompt, answers, layer, vector,
        multiplier=multiplier, positions=positions, device=device,
    )
