"""
COIECD: Contextually Optimized Inference with Entropy-based Conflict Detection
Yuan et al., 2024 (https://arxiv.org/abs/2401.10225)

Improves on CAD by gating the contrastive adjustment with a per-token entropy
signal.  The adjustment is applied only when the entropy difference between the
cued and uncued distributions exceeds a threshold τ, i.e. when the model is
actually conflicted at that token position.  This avoids the degradation CAD
causes on non-conflicting tokens by leaving those tokens' distributions unchanged.

At each decoding step:
  1. Compute H_cued = entropy(p_cued) and H_uncued = entropy(p_uncued).
  2. If H_uncued − H_cued > τ  (cue reduces uncertainty → conflict detected):
       use CAD formula: (1+α) log p_cued − α log p_uncued
  3. Else:
       use bare cued distribution: log p_cued

Like CAD, the cued prompt substitutes a disambiguating cue for the missing
external context in the intra-memory conflict setting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from inference_time.utils.model_utils import join_prompt_answer

logger = logging.getLogger(__name__)


def _entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy from log-probabilities (last dimension = vocab)."""
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def _top_p_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, sampled)


def _apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    for tok in generated[0].tolist():
        if logits[0, tok] < 0:
            logits[0, tok] = logits[0, tok] * penalty
        else:
            logits[0, tok] = logits[0, tok] / penalty
    return logits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def coiecd_generate(
    model,
    tokenizer,
    cued_input_ids: torch.Tensor,
    uncued_input_ids: torch.Tensor,
    alpha: float = 1.0,
    tau: float = 0.1,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Generate text with COIECD entropy-gated contrastive decoding.

    Args:
        model: GPT2LMHeadModel (or compatible).
        tokenizer: corresponding tokenizer.
        cued_input_ids: [1, seq_len] — prompt with disambiguating cue.
        uncued_input_ids: [1, seq_len] — bare prompt without cue.
        alpha: CAD contrastive strength applied when conflict is detected.
        tau: entropy-difference threshold; contrast is applied only when
             H(p_uncued) − H(p_cued) > tau.
        max_new_tokens: tokens to generate.
        temperature: softmax temperature.
        top_p: nucleus sampling threshold (1.0 = greedy).
        repetition_penalty: penalty for repeated tokens.
        device: target device.

    Returns:
        dict with keys:
            output_ids          -- full cued sequence including prompt
            generated_ids       -- list of newly generated token ids
            generated_text      -- decoded string
            conflict_flags      -- bool per step: True if contrast was applied
    """
    if device is None:
        device = next(model.parameters()).device
    cued_ids = cued_input_ids.to(device)
    uncued_ids = uncued_input_ids.to(device)

    cued_generated = cued_ids.clone()
    uncued_generated = uncued_ids.clone()
    generated_ids: List[int] = []
    conflict_flags: List[bool] = []

    for step in range(max_new_tokens):
        cued_logits = model(cued_generated).logits[:, -1, :]
        uncued_logits = model(uncued_generated).logits[:, -1, :]

        cued_log_p = F.log_softmax(cued_logits / max(temperature, 1e-8), dim=-1)
        uncued_log_p = F.log_softmax(uncued_logits / max(temperature, 1e-8), dim=-1)

        h_cued = _entropy(cued_log_p)      # scalar
        h_uncued = _entropy(uncued_log_p)

        conflict = (h_uncued - h_cued).item() > tau
        conflict_flags.append(conflict)

        if conflict:
            contrast_logits = (1.0 + alpha) * cued_log_p - alpha * uncued_log_p
            logger.debug("step %d: conflict detected (ΔH=%.4f), applying CAD", step, (h_uncued - h_cued).item())
        else:
            contrast_logits = cued_log_p

        if repetition_penalty != 1.0:
            contrast_logits = _apply_repetition_penalty(
                contrast_logits, cued_generated, repetition_penalty
            )

        probs = F.softmax(contrast_logits, dim=-1)
        if top_p < 1.0:
            next_token = _top_p_sample(probs, top_p)
        else:
            next_token = probs.argmax(dim=-1, keepdim=True)

        generated_ids.append(next_token.item())
        cued_generated = torch.cat([cued_generated, next_token], dim=-1)
        uncued_generated = torch.cat([uncued_generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "output_ids": cued_generated,
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "conflict_flags": conflict_flags,
    }


@torch.no_grad()
def score_answers(
    model,
    tokenizer,
    cued_prompt: str,
    uncued_prompt: str,
    answers: List[str],
    alpha: float = 1.0,
    tau: float = 0.1,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Score answer continuations under the COIECD entropy-gated distribution.

    Returns mean per-token log-probability under the (conditionally) contrasted
    distribution for each answer string.
    """
    if device is None:
        device = next(model.parameters()).device

    scores: Dict[str, float] = {}

    for ans in answers:
        cued_full = join_prompt_answer(cued_prompt, ans)
        uncued_full = join_prompt_answer(uncued_prompt, ans)

        cued_enc = tokenizer(cued_full, return_tensors="pt").to(device)
        uncued_enc = tokenizer(uncued_full, return_tensors="pt").to(device)
        cued_prompt_len = tokenizer(cued_prompt, return_tensors="pt")["input_ids"].shape[1]
        uncued_prompt_len = tokenizer(uncued_prompt, return_tensors="pt")["input_ids"].shape[1]

        cued_logits_all = model(cued_enc["input_ids"]).logits
        uncued_logits_all = model(uncued_enc["input_ids"]).logits

        ans_tokens = cued_enc["input_ids"][0, cued_prompt_len:]
        n = ans_tokens.shape[0]
        log_prob_sum = 0.0

        for k in range(n):
            cp = cued_prompt_len - 1 + k
            up = uncued_prompt_len - 1 + k
            target = ans_tokens[k].item()

            cued_lp = F.log_softmax(cued_logits_all[0, cp] / max(temperature, 1e-8), dim=-1)
            uncued_lp = F.log_softmax(uncued_logits_all[0, up] / max(temperature, 1e-8), dim=-1)

            h_cued = _entropy(cued_lp.unsqueeze(0)).item()
            h_uncued = _entropy(uncued_lp.unsqueeze(0)).item()

            if h_uncued - h_cued > tau:
                contrast = (1.0 + alpha) * cued_lp - alpha * uncued_lp
            else:
                contrast = cued_lp

            contrast = contrast - torch.logsumexp(contrast, dim=-1)
            log_prob_sum += contrast[target].item()

        scores[ans] = log_prob_sum / max(n, 1)

    return scores
