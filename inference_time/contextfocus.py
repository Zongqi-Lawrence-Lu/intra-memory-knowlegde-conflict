"""
ContextFocus
Anand et al., 2026 (see references.bib: anand2026contextfocus)

ContextFocus's central empirical claim is that activation steering *composes*
with prompting rather than substituting for it -- a context-faithfulness
steering vector plus an explicit disambiguating cue in the prompt outperforms
either alone. Since it was built for the context-memory setting (steering
towards trusting retrieved context), applying it here requires substituting a
disambiguating cue for the missing external context, mirroring the same
substitution CAD/COIECD make in the decoding-contrastive family (cad.py):
the "cue" is a temporally- or otherwise-disambiguating prefix (e.g. "As of
2021,"), and the steering vector is built CAA-style from cued vs. bare prompt
pairs rather than context-present vs. context-absent pairs.

This module exposes three modes so the composition claim can be tested
directly as a baseline ablation, not assumed:
  - "prompting_only": prepend the cue text, no steering.
  - "steering_only":  apply the steering vector, bare prompt.
  - "both":           cue text AND steering vector together.
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

MODES = ("prompting_only", "steering_only", "both")


def build_contextfocus_vector(
    model,
    tokenizer,
    contrastive_pairs: Sequence[Dict[str, str]],
    layer: int,
    device: torch.device,
    position: str = "last",
) -> torch.Tensor:
    """
    contrastive_pairs: list of {"cued": str, "bare": str}, where "cued"
    prepends a disambiguating cue to the same underlying question as "bare".
    Reuses the CAA diff-of-means construction (mean(cued) - mean(bare)).
    """
    cued = [pair["cued"] for pair in contrastive_pairs]
    bare = [pair["bare"] for pair in contrastive_pairs]
    vector = diff_of_means_vector(model, tokenizer, cued, bare, layer, device, position=position)
    logger.info(
        "Built ContextFocus vector at layer %d from %d cued/bare pairs (||v||=%.4f)",
        layer, len(contrastive_pairs), vector.norm().item(),
    )
    return vector


def _apply_cue(prompt: str, cue: Optional[str], mode: str) -> str:
    if mode in ("prompting_only", "both") and cue:
        return f"{cue} {prompt}"
    return prompt


def contextfocus_generate(
    model,
    tokenizer,
    prompt: str,
    cue: Optional[str],
    vector: Optional[torch.Tensor],
    layer: int,
    mode: str = "both",
    multiplier: float = 1.0,
    positions: str = "all",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if device is None:
        device = next(model.parameters()).device

    effective_prompt = _apply_cue(prompt, cue, mode)
    input_ids = tokenizer(effective_prompt, return_tensors="pt")["input_ids"].to(device)
    steer_vector = vector if mode in ("steering_only", "both") else None

    out = generate_with_steering(
        model, tokenizer, input_ids, layer, steer_vector,
        multiplier=multiplier, positions=positions,
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, repetition_penalty=repetition_penalty, device=device,
    )
    out["effective_prompt"] = effective_prompt
    out["mode"] = mode
    return out


def score_answers(
    model,
    tokenizer,
    prompt: str,
    cue: Optional[str],
    answers: List[str],
    vector: Optional[torch.Tensor],
    layer: int,
    mode: str = "both",
    multiplier: float = 1.0,
    positions: str = "all",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    effective_prompt = _apply_cue(prompt, cue, mode)
    steer_vector = vector if mode in ("steering_only", "both") else None

    return score_answers_with_steering(
        model, tokenizer, effective_prompt, answers, layer, steer_vector,
        multiplier=multiplier, positions=positions, device=device,
    )
