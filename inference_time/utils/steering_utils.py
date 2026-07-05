"""
Shared utilities for residual-stream activation steering, used by the
Activation Steering Methods family (CAA, SpARE, K-CAST, ContextFocus) in
inference_time/.

All methods in this family reduce to the same two primitives:
  1. Capture a hidden-state activation at a chosen layer/position (used to
     build steering vectors and, for K-CAST, to look up nearest neighbours).
  2. Add a (scaled) vector to the residual stream at a chosen layer during
     a forward pass (used at both generation and scoring time).

Keeping these primitives here means each method module only needs to define
*how the vector is constructed*, not how it's captured or applied.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

def get_block(model, layer: int):
    """Return the transformer block module to hook for a given layer index."""
    return model.transformer.h[layer]


@torch.no_grad()
def capture_activation(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    device: torch.device,
    position: str = "last",
) -> torch.Tensor:
    """
    Run a single forward pass over `prompt` and return the residual-stream
    hidden state at `layer` (the *output* of block `layer`).

    position: "last" returns the final-token activation [d_model];
              "mean" returns the mean activation over all positions [d_model].
    """
    captured: Dict[str, torch.Tensor] = {}

    def _hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()

    handle = get_block(model, layer).register_forward_hook(_hook)
    try:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        model(input_ids)
    finally:
        handle.remove()

    hs = captured["hs"][0]  # [seq_len, d_model]
    if position == "last":
        return hs[-1]
    elif position == "mean":
        return hs.mean(dim=0)
    raise ValueError(f"Unknown position mode: {position}")


def mean_activation(
    model,
    tokenizer,
    prompts: Sequence[str],
    layer: int,
    device: torch.device,
    position: str = "last",
) -> torch.Tensor:
    """Average `capture_activation` over a list of prompts."""
    vecs = [
        capture_activation(model, tokenizer, p, layer, device, position=position)
        for p in prompts
    ]
    return torch.stack(vecs, dim=0).mean(dim=0)


def diff_of_means_vector(
    model,
    tokenizer,
    positive_prompts: Sequence[str],
    negative_prompts: Sequence[str],
    layer: int,
    device: torch.device,
    position: str = "last",
) -> torch.Tensor:
    """
    The core CAA construction: mean activation over `positive_prompts` minus
    mean activation over `negative_prompts`, at a given layer.
    """
    pos = mean_activation(model, tokenizer, positive_prompts, layer, device, position)
    neg = mean_activation(model, tokenizer, negative_prompts, layer, device, position)
    return pos - neg


# ---------------------------------------------------------------------------
# Residual-stream addition
# ---------------------------------------------------------------------------

def _addition_hook_factory(vector: torch.Tensor, multiplier: float, positions: str):
    def _hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        v = (multiplier * vector).to(hs.dtype).to(hs.device)
        if positions == "all":
            hs = hs + v
        elif positions == "last":
            hs = hs.clone()
            hs[:, -1, :] = hs[:, -1, :] + v
        else:
            raise ValueError(f"Unknown positions mode: {positions}")
        return (hs,) + output[1:] if is_tuple else hs

    return _hook


def register_addition_hook(
    model,
    layer: int,
    vector: torch.Tensor,
    multiplier: float = 1.0,
    positions: str = "all",
):
    """
    Register a forward hook on `model.transformer.h[layer]` that adds
    `multiplier * vector` to its output hidden state.  Returns the hook
    handle; caller is responsible for calling `.remove()`.
    """
    hook = _addition_hook_factory(vector, multiplier, positions)
    return get_block(model, layer).register_forward_hook(hook)


class steering:
    """
    Context manager that applies an addition hook for its duration, e.g.:

        with steering(model, layer=6, vector=v, multiplier=4.0):
            out = model(input_ids)
    """

    def __init__(self, model, layer: int, vector: torch.Tensor, multiplier: float = 1.0, positions: str = "all"):
        self.model = model
        self.layer = layer
        self.vector = vector
        self.multiplier = multiplier
        self.positions = positions
        self._handle = None

    def __enter__(self):
        self._handle = register_addition_hook(
            self.model, self.layer, self.vector, self.multiplier, self.positions
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


# ---------------------------------------------------------------------------
# Generation / scoring with steering active
# ---------------------------------------------------------------------------

def _top_p_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, sampled)


@torch.no_grad()
def generate_with_steering(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    layer: int,
    vector: Optional[torch.Tensor],
    multiplier: float = 1.0,
    positions: str = "all",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Greedy/top-p generation with a steering vector added to `layer`'s output
    at every step.  If `vector` is None, generation proceeds unsteered (useful
    as an ablation baseline sharing this exact loop).

    Mirrors the recompute-full-sequence-each-step loop used by dola/cad in
    this package (no KV cache) for consistency across baselines.
    """
    if device is None:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    generated = input_ids.clone()

    ctx = steering(model, layer, vector, multiplier, positions) if vector is not None else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for step in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :].clone()

            # Repetition penalty applied to raw logits (which can be either sign),
            # not post-softmax log-probabilities (which are always <= 0 and would
            # make the "divide" branch below dead code) -- matches the convention
            # used by dola.py/cad.py/etc. in this package.
            if repetition_penalty != 1.0:
                for tok in generated[0].tolist():
                    if logits[0, tok] < 0:
                        logits[0, tok] = logits[0, tok] * repetition_penalty
                    else:
                        logits[0, tok] = logits[0, tok] / repetition_penalty

            log_probs = F.log_softmax(logits / max(temperature, 1e-8), dim=-1)
            probs = F.softmax(log_probs, dim=-1)
            if top_p < 1.0:
                next_token = _top_p_sample(probs, top_p)
            else:
                next_token = probs.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                logger.debug("EOS at step %d", step)
                break
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    prompt_len = input_ids.shape[1]
    generated_ids = generated[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "output_ids": generated,
        "generated_ids": generated_ids,
        "generated_text": generated_text,
    }


@torch.no_grad()
def score_answers_with_steering(
    model,
    tokenizer,
    prompt: str,
    answers: List[str],
    layer: int,
    vector: Optional[torch.Tensor],
    multiplier: float = 1.0,
    positions: str = "all",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Score each answer continuation's mean per-token log-probability with the
    steering vector active on `layer`.  Shared scoring routine for CAA,
    K-CAST, ContextFocus, and SpARE (once SpARE resolves its selected SAE
    features to a residual-space vector).
    """
    if device is None:
        device = next(model.parameters()).device

    scores: Dict[str, float] = {}
    ctx = steering(model, layer, vector, multiplier, positions) if vector is not None else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for ans in answers:
            full_text = prompt + ans
            enc = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            input_ids = enc["input_ids"]

            logits = model(input_ids).logits  # [1, seq_len, vocab]
            seq_len = input_ids.shape[1]
            n_ans_tokens = seq_len - prompt_len
            log_prob_sum = 0.0
            for pos in range(prompt_len - 1, seq_len - 1):
                target = input_ids[0, pos + 1].item()
                lp = F.log_softmax(logits[0, pos], dim=-1)
                log_prob_sum += lp[target].item()
            scores[ans] = log_prob_sum / max(n_ans_tokens, 1)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    return scores
