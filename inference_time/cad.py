"""
Context-Aware Decoding (CAD)
Shi et al., 2023 (https://arxiv.org/abs/2305.14739)

Runs the model twice per decoding step: once with a disambiguating-cue prompt
and once without.  The contrastive distribution amplifies the shift the cue
induces in the output distribution:

    log p_CAD(x | cued, uncued) ∝ (1 + α) log p(x | cued) − α log p(x | uncued)

In the original paper the "cued" prompt is a retrieved context passage and the
"uncued" prompt is the bare question.  In the intra-memory conflict setting
there is no external context, so the two prompts are instead a temporally- or
otherwise-disambiguated version of the question (cued) versus the bare question
(uncued).  The interface accepts both prompts explicitly so the caller controls
what constitutes a disambiguating cue.

The fixed α is the main weakness of CAD (degrades non-conflicting inputs);
see COIECD and AdaCAD for entropy-based and divergence-based fixes respectively.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
def cad_generate(
    model,
    tokenizer,
    cued_input_ids: torch.Tensor,
    uncued_input_ids: torch.Tensor,
    alpha: float = 1.0,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Generate text with CAD contrastive decoding.

    Args:
        model: GPT2LMHeadModel (or compatible).
        tokenizer: corresponding tokenizer.
        cued_input_ids: [1, seq_len_cued] — prompt with disambiguating cue.
        uncued_input_ids: [1, seq_len_uncued] — bare prompt without cue.
        alpha: contrastive strength (paper default 1.0).
        max_new_tokens: tokens to generate.
        temperature: softmax temperature.
        top_p: nucleus sampling threshold (1.0 = greedy).
        repetition_penalty: penalty for repeated tokens.
        device: target device (inferred from model if None).

    Returns:
        dict with keys:
            output_ids      -- full cued sequence including prompt
            generated_ids   -- list of newly generated token ids
            generated_text  -- decoded string
    """
    if device is None:
        device = next(model.parameters()).device
    cued_ids = cued_input_ids.to(device)
    uncued_ids = uncued_input_ids.to(device)

    # Track generated tokens separately so we can apply repetition penalty
    cued_generated = cued_ids.clone()
    uncued_generated = uncued_ids.clone()

    generated_ids: List[int] = []

    for step in range(max_new_tokens):
        cued_logits = model(cued_generated).logits[:, -1, :]       # [1, vocab]
        uncued_logits = model(uncued_generated).logits[:, -1, :]   # [1, vocab]

        cued_log_p = F.log_softmax(cued_logits / max(temperature, 1e-8), dim=-1)
        uncued_log_p = F.log_softmax(uncued_logits / max(temperature, 1e-8), dim=-1)

        # CAD formula: (1 + α) log p_cued − α log p_uncued
        contrast_logits = (1.0 + alpha) * cued_log_p - alpha * uncued_log_p

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
            logger.debug("EOS at step %d", step)
            break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "output_ids": cued_generated,
        "generated_ids": generated_ids,
        "generated_text": generated_text,
    }


@torch.no_grad()
def score_answers(
    model,
    tokenizer,
    cued_prompt: str,
    uncued_prompt: str,
    answers: List[str],
    alpha: float = 1.0,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Score answer continuations under the CAD contrastive distribution.

    Returns mean per-token log-probability under the contrasted distribution
    for each answer string.
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

        cued_ids = cued_enc["input_ids"]
        uncued_ids = uncued_enc["input_ids"]

        cued_out = model(cued_ids).logits     # [1, seq_c, vocab]
        uncued_out = model(uncued_ids).logits  # [1, seq_u, vocab]

        # Score only the answer tokens (positions after the cued prompt)
        ans_tokens = cued_ids[0, cued_prompt_len:]
        n = ans_tokens.shape[0]
        log_prob_sum = 0.0

        for k in range(n):
            cued_pos = cued_prompt_len - 1 + k
            uncued_pos = uncued_prompt_len - 1 + k
            target = ans_tokens[k].item()

            cued_lp = F.log_softmax(
                cued_out[0, cued_pos] / max(temperature, 1e-8), dim=-1
            )
            uncued_lp = F.log_softmax(
                uncued_out[0, uncued_pos] / max(temperature, 1e-8), dim=-1
            )

            contrast = (1.0 + alpha) * cued_lp - alpha * uncued_lp
            contrast = contrast - torch.logsumexp(contrast, dim=-1)
            log_prob_sum += contrast[target].item()

        scores[ans] = log_prob_sum / max(n, 1)

    return scores
