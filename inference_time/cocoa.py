"""
CoCoA: Confidence-Aware Contextual Decoding
Khandelwal et al., 2025 (https://arxiv.org/abs/2502.09732)

Extends AdaCAD by composing three signals into the adaptive coefficient α_t:

  1. JSD term (same as AdaCAD): divergence between cued and uncued distributions.
  2. Entropy-gap signal: how much the cued distribution is sharper than uncued
     (H_uncued − H_cued), normalised to [0, 1].
  3. Contextual peakedness: max-probability of the cued distribution (how
     "peaked" / confident the cued model is).

Combined coefficient:
    α_t = β · [ γ_jsd · JSD(p_c, p_u)
               + γ_ent · (H_u − H_c) / log(V)
               + γ_peak · max(p_c) ]

where V is vocab size, normalisation ensures the entropy gap is in [0, 1],
and γ_{jsd,ent,peak} are per-signal weights (default equal weighting 1/3 each).
β is a global scale factor.

The three-signal arbiter is less likely than AdaCAD's single-signal arbiter to
mis-fire on ambiguous tokens where JSD is high but the cued model is diffuse.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from inference_time.utils.model_utils import join_prompt_answer

logger = logging.getLogger(__name__)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    kl_qm = (q * (q.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def _entropy(log_probs: torch.Tensor) -> torch.Tensor:
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


def _compute_alpha(
    cued_log_p: torch.Tensor,
    uncued_log_p: torch.Tensor,
    beta: float,
    gamma_jsd: float,
    gamma_ent: float,
    gamma_peak: float,
    vocab_size: int,
) -> float:
    """Compute the per-step adaptive coefficient from the three CoCoA signals."""
    cued_p = cued_log_p.exp()
    uncued_p = uncued_log_p.exp()

    jsd = _js_divergence(cued_p, uncued_p).item()

    h_cued = _entropy(cued_log_p).item()
    h_uncued = _entropy(uncued_log_p).item()
    log_v = math.log(vocab_size)
    ent_gap = max(h_uncued - h_cued, 0.0) / log_v   # normalised to [0, 1]

    peakedness = cued_p.max(dim=-1).values.item()    # already in [0, 1]

    alpha_t = beta * (gamma_jsd * jsd + gamma_ent * ent_gap + gamma_peak * peakedness)
    return float(alpha_t)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def cocoa_generate(
    model,
    tokenizer,
    cued_input_ids: torch.Tensor,
    uncued_input_ids: torch.Tensor,
    beta: float = 1.0,
    gamma_jsd: float = 1 / 3,
    gamma_ent: float = 1 / 3,
    gamma_peak: float = 1 / 3,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Generate text with CoCoA confidence-aware adaptive contrastive decoding.

    Args:
        model: GPT2LMHeadModel (or compatible).
        tokenizer: corresponding tokenizer.
        cued_input_ids: [1, seq_len] — prompt with disambiguating cue.
        uncued_input_ids: [1, seq_len] — bare prompt without cue.
        beta: global scale factor for combined signal.
        gamma_jsd: weight for JSD signal (default 1/3).
        gamma_ent: weight for entropy-gap signal (default 1/3).
        gamma_peak: weight for contextual-peakedness signal (default 1/3).
        max_new_tokens: tokens to generate.
        temperature: softmax temperature.
        top_p: nucleus sampling threshold (1.0 = greedy).
        repetition_penalty: penalty for repeated tokens.
        device: target device.

    Returns:
        dict with keys:
            output_ids      -- full cued sequence including prompt
            generated_ids   -- list of newly generated token ids
            generated_text  -- decoded string
            alphas          -- adaptive coefficient α_t at each step
    """
    if device is None:
        device = next(model.parameters()).device
    cued_ids = cued_input_ids.to(device)
    uncued_ids = uncued_input_ids.to(device)

    vocab_size = model.config.vocab_size
    cued_generated = cued_ids.clone()
    uncued_generated = uncued_ids.clone()
    generated_ids: List[int] = []
    alphas: List[float] = []

    for step in range(max_new_tokens):
        cued_logits = model(cued_generated).logits[:, -1, :]
        uncued_logits = model(uncued_generated).logits[:, -1, :]

        cued_log_p = F.log_softmax(cued_logits / max(temperature, 1e-8), dim=-1)
        uncued_log_p = F.log_softmax(uncued_logits / max(temperature, 1e-8), dim=-1)

        alpha_t = _compute_alpha(
            cued_log_p, uncued_log_p, beta, gamma_jsd, gamma_ent, gamma_peak, vocab_size
        )
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
    gamma_jsd: float = 1 / 3,
    gamma_ent: float = 1 / 3,
    gamma_peak: float = 1 / 3,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Score answer continuations under the CoCoA distribution.

    Returns mean per-token log-probability for each answer string.
    """
    if device is None:
        device = next(model.parameters()).device

    vocab_size = model.config.vocab_size
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

            alpha_t = _compute_alpha(
                cued_lp.unsqueeze(0), uncued_lp.unsqueeze(0),
                beta, gamma_jsd, gamma_ent, gamma_peak, vocab_size,
            )

            contrast = (1.0 + alpha_t) * cued_lp - alpha_t * uncued_lp
            contrast = contrast - torch.logsumexp(contrast, dim=-1)
            log_prob_sum += contrast[target].item()

        scores[ans] = log_prob_sum / max(n, 1)

    return scores
