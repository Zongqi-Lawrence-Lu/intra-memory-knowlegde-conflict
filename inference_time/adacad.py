"""
AdaCAD: Adaptive Context-Aware Decoding
Wang et al., 2024 (https://arxiv.org/abs/2405.11364)

Replaces CAD's fixed contrastive coefficient α with a per-step coefficient
derived from the Jensen-Shannon divergence between the cued and uncued
distributions.  When the two distributions agree (low JSD) the coefficient
approaches zero, preserving the original distribution; when they strongly
disagree (high JSD) the coefficient amplifies the shift.

Adaptive coefficient at step t:
    α_t = JSD(p_cued ‖ p_uncued)   (scaled to [0, 1] by definition of JSD)

Final distribution:
    log p_AdaCAD ∝ (1 + α_t) log p_cued − α_t log p_uncued

An optional global scaling factor β lets you up- or down-weight the adaptive
coefficient: α_t ← β · JSD(p_cued ‖ p_uncued).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scalar JSD between two probability vectors (last dim = vocab)."""
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    kl_qm = (q * (q.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


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
def adacad_generate(
    model,
    tokenizer,
    cued_input_ids: torch.Tensor,
    uncued_input_ids: torch.Tensor,
    beta: float = 1.0,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Generate text with AdaCAD adaptive contrastive decoding.

    Args:
        model: GPT2LMHeadModel (or compatible).
        tokenizer: corresponding tokenizer.
        cued_input_ids: [1, seq_len] — prompt with disambiguating cue.
        uncued_input_ids: [1, seq_len] — bare prompt without cue.
        beta: global scaling factor for the JSD-derived coefficient (default 1.0).
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
            alphas              -- adaptive coefficient α_t at each step
    """
    if device is None:
        device = next(model.parameters()).device
    cued_ids = cued_input_ids.to(device)
    uncued_ids = uncued_input_ids.to(device)

    cued_generated = cued_ids.clone()
    uncued_generated = uncued_ids.clone()
    generated_ids: List[int] = []
    alphas: List[float] = []

    for step in range(max_new_tokens):
        cued_logits = model(cued_generated).logits[:, -1, :]
        uncued_logits = model(uncued_generated).logits[:, -1, :]

        cued_log_p = F.log_softmax(cued_logits / max(temperature, 1e-8), dim=-1)
        uncued_log_p = F.log_softmax(uncued_logits / max(temperature, 1e-8), dim=-1)

        cued_p = cued_log_p.exp()
        uncued_p = uncued_log_p.exp()

        alpha_t = (beta * _js_divergence(cued_p, uncued_p)).clamp(min=0.0).item()
        alphas.append(alpha_t)

        contrast_logits = (1.0 + alpha_t) * cued_log_p - alpha_t * uncued_log_p

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

        logger.debug("step %d: α_t=%.4f", step, alpha_t)
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "output_ids": cued_generated,
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "alphas": alphas,
    }


@torch.no_grad()
def score_answers(
    model,
    tokenizer,
    cued_prompt: str,
    uncued_prompt: str,
    answers: List[str],
    beta: float = 1.0,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Score answer continuations under the AdaCAD adaptive distribution.

    Returns mean per-token log-probability for each answer string.
    """
    if device is None:
        device = next(model.parameters()).device

    scores: Dict[str, float] = {}

    for ans in answers:
        cued_full = cued_prompt + ans
        uncued_full = uncued_prompt + ans

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

            alpha_t = (beta * _js_divergence(
                cued_lp.exp().unsqueeze(0), uncued_lp.exp().unsqueeze(0)
            )).clamp(min=0.0).item()

            contrast = (1.0 + alpha_t) * cued_lp - alpha_t * uncued_lp
            contrast = contrast - torch.logsumexp(contrast, dim=-1)
            log_prob_sum += contrast[target].item()

        scores[ans] = log_prob_sum / max(n, 1)

    return scores
