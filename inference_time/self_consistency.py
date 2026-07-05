"""
Self-Consistency for Knowledge Conflict Mitigation
Wang et al., 2022 (https://arxiv.org/abs/2203.11171)  [original CoT self-consistency]

Self-consistency was originally proposed for chain-of-thought reasoning: sample
multiple reasoning paths and take a majority vote over the final answer.
Transplanted to factual recall and intra-memory conflict, the idea becomes:

    1. Sample K independent continuations of the probe question.
    2. Parse or normalise each continuation to a candidate answer.
    3. Return the majority-vote answer; or abstain if no majority exists
       (high disagreement = detected conflict).

This is the most obvious black-box comparison point for the internal-access
methods in this project (DoLa, CAA, PH3, etc.) because it:
  - Requires no access to model internals or activations.
  - Works with any model through a text-generation API.
  - Has a well-understood cost profile (K forward passes per query vs 1–2 for
    the decoding-contrastive family and 1 for steered generation).

As of the survey, self-consistency has been used as a conflict *detection* signal
(high disagreement = conflict present) but not evaluated as a named *mitigation*
method for intra-memory conflict.  This implementation serves both roles:
  - `self_consistent_answer` returns the majority answer (mitigation).
  - `consistency_score` returns the fraction of samples agreeing on the top
    answer (detection; high score = low conflict).

Answer extraction
-----------------
For open-ended generation the "answer" is the first N tokens of the continuation.
For multiple-choice probes where the answer set is known, the continuation is
matched against the provided choices by (a) exact prefix match or (b) argmax over
per-choice log-probabilities of the sampled continuation (see `vote_by_scoring`).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampling (mirrors semantic_entropy.sample_answers but returns more metadata)
# ---------------------------------------------------------------------------

def _top_p_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, sampled)


@torch.no_grad()
def _single_sample(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    generated = input_ids.clone()
    prompt_len = input_ids.shape[1]
    for _ in range(max_new_tokens):
        logits = model(generated).logits[:, -1, :]
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
        if top_p < 1.0:
            next_tok = _top_p_sample(probs, top_p)
        else:
            next_tok = probs.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=-1)
        if next_tok.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[0, prompt_len:].tolist(), skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Open-ended majority vote
# ---------------------------------------------------------------------------

@torch.no_grad()
def self_consistent_answer(
    model,
    tokenizer,
    prompt: str,
    k: int = 10,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.9,
    abstain_threshold: float = 0.0,
    normalise: bool = True,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Sample K continuations and return the majority-vote answer.

    Args:
        model: GPT2LMHeadModel (or compatible).
        tokenizer: corresponding tokenizer.
        prompt: the probe question.
        k: number of samples.
        max_new_tokens: length cap per sample.
        temperature: sampling temperature.
        top_p: nucleus threshold.
        abstain_threshold: if the top-answer fraction is ≤ this, return abstain=True.
        normalise: lowercase + strip before counting votes.
        device: target device.

    Returns:
        dict with keys:
            answer          -- majority-vote answer (str) or None if abstaining
            abstain         -- bool
            vote_fraction   -- fraction of samples voting for the top answer
            vote_counts     -- Counter of all answer strings
            samples         -- list of raw sampled continuations
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    samples = [
        _single_sample(model, input_ids, tokenizer, max_new_tokens, temperature, top_p, device)
        for _ in range(k)
    ]

    keys = [s.lower().strip() if normalise else s for s in samples]
    counts = Counter(keys)
    top_key, top_count = counts.most_common(1)[0]
    top_frac = top_count / k

    # Map back to original case
    original = next(s for s, norm in zip(samples, keys) if norm == top_key)
    abstain = top_frac <= abstain_threshold

    return {
        "answer": None if abstain else original,
        "abstain": abstain,
        "vote_fraction": top_frac,
        "vote_counts": dict(counts),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Multiple-choice: vote by scoring each answer choice
# ---------------------------------------------------------------------------

@torch.no_grad()
def vote_by_scoring(
    model,
    tokenizer,
    prompt: str,
    answers: List[str],
    k: int = 10,
    max_new_tokens: int = 20,
    temperature_sample: float = 1.0,
    top_p: float = 0.9,
    temperature_score: float = 1.0,
    abstain_threshold: float = 0.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    For multiple-choice probes: sample K continuations then, for each sample,
    identify which supplied answer choice it is closest to by log-probability
    scoring.  Cast a vote for that choice and return the majority.

    The scoring step uses the standard next-token log-probability of the answer
    continuation given the prompt (same as score_answers in the decoding modules).

    Args:
        model, tokenizer: the model.
        prompt: the probe question.
        answers: list of candidate answer strings.
        k: number of samples.
        max_new_tokens: length cap per sample (for the sampling step).
        temperature_sample: temperature for the sampling step.
        top_p: nucleus threshold for sampling.
        temperature_score: temperature for the scoring step.
        abstain_threshold: abstain if top-vote fraction ≤ this.
        device: target device.

    Returns:
        dict with keys:
            answer          -- majority-vote answer (str) or None if abstaining
            abstain         -- bool
            vote_fraction   -- fraction of K samples voting for top answer
            vote_counts     -- Counter over answer strings
            answer_scores   -- mean log-prob scores for each answer (from scoring)
            samples         -- list of raw sampled continuations
    """
    if device is None:
        device = next(model.parameters()).device

    # Pre-compute scores for each answer (ground-truth scoring, independent of samples)
    answer_scores: Dict[str, float] = {}
    for ans in answers:
        full = prompt + ans
        enc = tokenizer(full, return_tensors="pt").to(device)
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        ids = enc["input_ids"]
        logits_all = model(ids).logits
        ans_tokens = ids[0, prompt_len:]
        n = ans_tokens.shape[0]
        log_prob_sum = 0.0
        for k_pos in range(n):
            pos = prompt_len - 1 + k_pos
            target = ans_tokens[k_pos].item()
            lp = F.log_softmax(logits_all[0, pos] / max(temperature_score, 1e-8), dim=-1)
            log_prob_sum += lp[target].item()
        answer_scores[ans] = log_prob_sum / max(n, 1)

    # Sample K continuations and vote by nearest-answer matching
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    samples = [
        _single_sample(
            model, input_ids, tokenizer, max_new_tokens,
            temperature_sample, top_p, device,
        )
        for _ in range(k)
    ]

    votes: List[str] = []
    for sample in samples:
        # Match sample to nearest answer by prefix or substring, else by score
        matched = None
        sample_lower = sample.lower().strip()
        for ans in answers:
            if sample_lower.startswith(ans.lower().strip()):
                matched = ans
                break
        if matched is None:
            # Fall back to highest log-prob answer
            matched = max(answer_scores, key=answer_scores.__getitem__)
        votes.append(matched)

    counts = Counter(votes)
    top_ans, top_count = counts.most_common(1)[0]
    top_frac = top_count / k
    abstain = top_frac <= abstain_threshold

    return {
        "answer": None if abstain else top_ans,
        "abstain": abstain,
        "vote_fraction": top_frac,
        "vote_counts": dict(counts),
        "answer_scores": answer_scores,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Consistency score (detection-only interface)
# ---------------------------------------------------------------------------

@torch.no_grad()
def consistency_score(
    model,
    tokenizer,
    prompt: str,
    k: int = 20,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.9,
    normalise: bool = True,
    device: Optional[torch.device] = None,
) -> float:
    """
    Return the fraction of K samples that agree on the most common answer.
    High score = low conflict (the model is self-consistent).
    Low score = high conflict (the model disagrees with itself).

    Equivalent to 1 − normalised_entropy (with max-agreement as the signal).
    Use semantic_entropy.conflict_score for a proper entropy-based signal.
    """
    result = self_consistent_answer(
        model, tokenizer, prompt,
        k=k, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
        normalise=normalise, device=device,
    )
    return result["vote_fraction"]
