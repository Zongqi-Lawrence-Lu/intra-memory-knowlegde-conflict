"""
ARR: Adaptive Regime Routing
Jiang et al., 2026 (https://arxiv.org/abs/2502.XXXXX)  [citation: jiang2026contextaware]

Generalises the CAD/COIECD/AdaCAD/CoCoA family by observing that all four
methods still default to trusting the cued (context) distribution whenever a
conflict is detected.  ARR instead routes each decoding step among three regimes:

  - TRUST_CONTEXT  : use cued distribution (context more informative)
  - TRUST_PRIOR    : use uncued distribution (parametric prior more stable)
  - AGREE          : distributions agree; use cued distribution as-is

Routing decision is made per step from two signals:
  1. JSD(p_cued, p_uncued)   — how much they diverge (conflict strength)
  2. H_cued vs H_uncued      — which distribution is more "confident"

Routing rules (thresholds τ_jsd and τ_ent are hyperparameters):
  - JSD < τ_jsd → AGREE (no conflict; pass through p_cued)
  - JSD ≥ τ_jsd AND H_cued < H_uncued − τ_ent → TRUST_CONTEXT (cued sharper)
  - JSD ≥ τ_jsd AND H_uncued < H_cued − τ_ent → TRUST_PRIOR (uncued sharper)
  - otherwise → blend (TRUST_CONTEXT with weight proportional to peakedness)

In each non-AGREE regime, the chosen distribution is optionally amplified with a
CAD-style contrastive step at a fixed strength α_ctx or α_prior respectively.

For the intra-memory conflict setting, TRUST_PRIOR corresponds to "believe the
more frequent / earlier training signal", while TRUST_CONTEXT corresponds to
"believe the disambiguating temporal cue injected into the prompt".
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from inference_time.utils.model_utils import join_prompt_answer

logger = logging.getLogger(__name__)

REGIME_AGREE = "agree"
REGIME_TRUST_CONTEXT = "trust_context"
REGIME_TRUST_PRIOR = "trust_prior"
REGIME_BLEND = "blend"


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    kl_qm = (q * (q.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    return (0.5 * (kl_pm + kl_qm)).item()


def _entropy(log_probs: torch.Tensor) -> float:
    return -(log_probs.exp() * log_probs).sum(dim=-1).item()


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


def _route(
    cued_log_p: torch.Tensor,
    uncued_log_p: torch.Tensor,
    tau_jsd: float,
    tau_ent: float,
    alpha_ctx: float,
    alpha_prior: float,
) -> Tuple[torch.Tensor, str]:
    """
    Apply ARR routing and return (contrast_logits, regime_label).
    """
    cued_p = cued_log_p.exp()
    uncued_p = uncued_log_p.exp()
    jsd = _js_divergence(cued_p, uncued_p)
    h_cued = _entropy(cued_log_p)
    h_uncued = _entropy(uncued_log_p)

    if jsd < tau_jsd:
        # Distributions agree — pass through cued
        return cued_log_p, REGIME_AGREE

    ent_diff = h_cued - h_uncued
    if ent_diff < -tau_ent:
        # Cued is sharper → trust context
        contrast = (1.0 + alpha_ctx) * cued_log_p - alpha_ctx * uncued_log_p
        return contrast, REGIME_TRUST_CONTEXT
    elif ent_diff > tau_ent:
        # Uncued is sharper → trust prior
        contrast = (1.0 + alpha_prior) * uncued_log_p - alpha_prior * cued_log_p
        return contrast, REGIME_TRUST_PRIOR
    else:
        # Ambiguous: blend, weighted by cued peakedness
        w = cued_p.max(dim=-1).values.item()   # in [0, 1]
        blend = w * cued_log_p + (1.0 - w) * uncued_log_p
        return blend, REGIME_BLEND


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def arr_generate(
    model,
    tokenizer,
    cued_input_ids: torch.Tensor,
    uncued_input_ids: torch.Tensor,
    tau_jsd: float = 0.05,
    tau_ent: float = 0.1,
    alpha_ctx: float = 1.0,
    alpha_prior: float = 1.0,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Generate text with ARR Adaptive Regime Routing.

    Args:
        model: GPT2LMHeadModel (or compatible).
        tokenizer: corresponding tokenizer.
        cued_input_ids: [1, seq_len] — prompt with disambiguating cue.
        uncued_input_ids: [1, seq_len] — bare prompt without cue.
        tau_jsd: JSD threshold below which distributions are considered to agree.
        tau_ent: entropy-difference threshold for assigning trust-context vs
                 trust-prior regime.
        alpha_ctx: CAD strength when routing to TRUST_CONTEXT.
        alpha_prior: CAD strength when routing to TRUST_PRIOR.
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
            regimes         -- regime label at each step
    """
    if device is None:
        device = next(model.parameters()).device
    cued_ids = cued_input_ids.to(device)
    uncued_ids = uncued_input_ids.to(device)

    cued_generated = cued_ids.clone()
    uncued_generated = uncued_ids.clone()
    generated_ids: List[int] = []
    regimes: List[str] = []

    for step in range(max_new_tokens):
        cued_logits = model(cued_generated).logits[:, -1, :]
        uncued_logits = model(uncued_generated).logits[:, -1, :]

        cued_log_p = F.log_softmax(cued_logits / max(temperature, 1e-8), dim=-1)
        uncued_log_p = F.log_softmax(uncued_logits / max(temperature, 1e-8), dim=-1)

        contrast_logits, regime = _route(
            cued_log_p, uncued_log_p, tau_jsd, tau_ent, alpha_ctx, alpha_prior
        )
        regimes.append(regime)
        logger.debug("step %d: regime=%s", step, regime)

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
        "regimes": regimes,
    }


@torch.no_grad()
def score_answers(
    model,
    tokenizer,
    cued_prompt: str,
    uncued_prompt: str,
    answers: List[str],
    tau_jsd: float = 0.05,
    tau_ent: float = 0.1,
    alpha_ctx: float = 1.0,
    alpha_prior: float = 1.0,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Score answer continuations under the ARR-routed distribution.

    Returns mean per-token log-probability for each answer string.
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

            contrast, _ = _route(
                cued_lp.unsqueeze(0), uncued_lp.unsqueeze(0),
                tau_jsd, tau_ent, alpha_ctx, alpha_prior,
            )
            contrast = contrast.squeeze(0)
            contrast = contrast - torch.logsumexp(contrast, dim=-1)
            log_prob_sum += contrast[target].item()

        scores[ans] = log_prob_sum / max(n, 1)

    return scores
