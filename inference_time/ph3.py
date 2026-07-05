"""
PH3: Pruning Heads via PatH PatcHing
Jin et al., 2024 (https://arxiv.org/abs/2402.12897)

PH3 causally identifies the attention heads responsible for propagating a
memorised fact via path patching, then prunes or down-weights those heads at
inference time.  The mechanism is distinct from steering: rather than adding a
vector, it removes (or attenuates) the circuitry that carries the conflicting
memory.

This is one of only two inference-time methods in this survey that is genuinely
intra-memory-native (the other is DoLa): it needs no context-vs-parametric
contrast, only a "clean" run (the question with a disambiguating cue) and a
"corrupted" run (the question without a cue or with the opposing cue), from
which path patching identifies which heads carry the difference.

Two-phase workflow
------------------
Phase 1 — Causal attribution (offline):
    For each attention head h at each layer l, compute the path-patching
    indirect effect of that head on the logit difference between the two
    competing answers.  Specifically:
        1. Run "clean" forward pass (cued prompt) → cache all activations.
        2. Run "corrupted" forward pass (bare/opposing prompt) → cache all activations.
        3. For each head (l, h): re-run from the corrupted activations but
           patch *only* the output of head (l, h) from the clean cache, then
           measure the change in the logit difference at the final position.
        The indirect effect of (l, h) is this change; heads with large positive
        IE "help" the model output the clean answer.

Phase 2 — Head pruning (inference):
    At inference time, zero out (or scale down by a factor ρ) the output
    projections of the top-k highest-IE heads identified in Phase 1.

Implementation notes
--------------------
GPT-2's attention output at layer l is the *concatenation* of per-head outputs
passed through a single linear projection W_O (`GPT2Attention.c_proj`, a
`Conv1D`):
    MHA_out[l] = concat([head_0, ..., head_{H-1}]) @ W_O[l] + b_O[l]

Because W_O is linear over the concatenated heads, head h's contribution to
MHA_out is exactly `head_h @ W_O[l][h*head_dim:(h+1)*head_dim, :]` -- no
matrix inversion is needed to isolate it. `c_proj` is a real submodule, so a
`register_forward_pre_hook` on it gives direct read/write access to its input
(the pre-projection concatenated head tensor; last dim = n_heads * head_dim,
with heads contiguous in that order because of how GPT2Attention reshapes
before calling `c_proj`) -- no subclassing of the attention module required.
Patching head h means overwriting exactly the `[h*head_dim:(h+1)*head_dim]`
slice of that input with the clean run's corresponding slice (Phase 1), or
scaling it by rho (Phase 2); `c_proj`'s own forward pass then does the correct
linear combination for every other head, unaffected.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation cache helpers
# ---------------------------------------------------------------------------

def _head_dim(model) -> int:
    return model.config.n_embd // model.config.n_head


def _head_slice(head: int, head_dim: int) -> slice:
    return slice(head * head_dim, (head + 1) * head_dim)


def _cache_pre_proj_hook_factory(cache: Dict[int, torch.Tensor], layer: int):
    """Forward-pre-hook on `attn.c_proj`: caches its input (the pre-projection,
    per-head-concatenated tensor) for this layer."""
    def hook(module, args):
        cache[layer] = args[0].detach().clone()
    return hook


@torch.no_grad()
def _run_with_pre_proj_cache(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Full forward pass capturing each layer's pre-`c_proj` (i.e. pre-W_O)
    concatenated per-head attention output.

    Returns (final_logits, pre_proj_cache) where pre_proj_cache[l] is
    [1, seq_len, n_heads*head_dim] at layer l.
    """
    cache: Dict[int, torch.Tensor] = {}
    handles = []
    for l, block in enumerate(model.transformer.h):
        handles.append(
            block.attn.c_proj.register_forward_pre_hook(_cache_pre_proj_hook_factory(cache, l))
        )
    try:
        out = model(input_ids.to(device))
    finally:
        for h in handles:
            h.remove()
    return out.logits, cache


# ---------------------------------------------------------------------------
# Per-head patching
# ---------------------------------------------------------------------------

def _head_patch_pre_hook_factory(
    clean_cache: Dict[int, torch.Tensor],
    layer: int,
    head: int,
    head_dim: int,
):
    """Forward-pre-hook on `attn.c_proj` at `layer`: overwrites exactly head
    `head`'s slice of the pre-projection input with the clean run's value for
    that slice, leaving every other head's slice (and the rest of the
    sequence) untouched."""
    clean_pre = clean_cache[layer]  # [1, seq_len, n_heads*head_dim]
    sl = _head_slice(head, head_dim)

    def hook(module, args):
        x = args[0].clone()
        x[..., sl] = clean_pre[..., sl]
        return (x,) + args[1:]

    return hook


# ---------------------------------------------------------------------------
# Phase 1: causal attribution
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_head_scores(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    answer_pos: int,
    answer_neg: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the path-patching indirect effect of each attention head on the
    logit difference (answer_pos − answer_neg) at the final token position.

    Args:
        model: GPT2LMHeadModel.
        tokenizer: corresponding tokenizer.
        clean_prompt: the cued / "correct answer" prompt.
        corrupted_prompt: the bare / "opposing answer" prompt.
        answer_pos: token id of the target (positive) answer first token.
        answer_neg: token id of the negative answer first token.
        device: target device.

    Returns:
        scores: [n_layers, n_heads] tensor of indirect effects, one entry per
                individual (layer, head) pair (not a layer-level average).
                Positive = head helps produce the clean (positive) answer.
    """
    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corrupt_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]

    _, clean_cache = _run_with_pre_proj_cache(model, clean_ids, device)
    corrupt_logits, _ = _run_with_pre_proj_cache(model, corrupt_ids, device)

    def _logit_diff(logits: torch.Tensor) -> float:
        lp = logits[0, -1]
        return (lp[answer_pos] - lp[answer_neg]).item()

    baseline_diff = _logit_diff(corrupt_logits)
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    head_dim = _head_dim(model)
    scores = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        for head in range(n_heads):
            hook = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
                _head_patch_pre_hook_factory(clean_cache, layer, head, head_dim)
            )
            try:
                patched_out = model(corrupt_ids.to(device))
                patched_diff = _logit_diff(patched_out.logits)
            finally:
                hook.remove()

            scores[layer, head] = patched_diff - baseline_diff

        logger.debug("layer %d: head IEs=%s", layer, scores[layer].tolist())

    return scores


# ---------------------------------------------------------------------------
# Phase 2: inference with head pruning
# ---------------------------------------------------------------------------

def _pruning_pre_hook_factory(heads_at_layer: Set[int], head_dim: int, rho: float):
    """Forward-pre-hook on `attn.c_proj`: scales exactly the specified heads'
    slices of the pre-projection input by `rho` (0.0 = full ablation)."""
    def hook(module, args):
        if not heads_at_layer:
            return args
        x = args[0].clone()
        for head in heads_at_layer:
            sl = _head_slice(head, head_dim)
            x[..., sl] = x[..., sl] * rho
        return (x,) + args[1:]

    return hook


@contextmanager
def pruned_heads(model, head_indices: Set[Tuple[int, int]], rho: float = 0.0):
    """
    Context manager that attenuates a set of (layer, head) indices during the
    enclosed forward pass(es).

    Args:
        model: GPT2LMHeadModel.
        head_indices: set of (layer_idx, head_idx) tuples to prune.
        rho: scale factor applied to the pruned heads' contribution
             (0.0 = full ablation, 1.0 = no change).
    """
    head_dim = _head_dim(model)
    handles = []
    for layer in range(model.config.n_layer):
        heads_at_layer = {h for (l, h) in head_indices if l == layer}
        if heads_at_layer:
            handles.append(
                model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
                    _pruning_pre_hook_factory(heads_at_layer, head_dim, rho)
                )
            )
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _top_p_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, sampled)


@torch.no_grad()
def ph3_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    head_indices: Set[Tuple[int, int]],
    rho: float = 0.0,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Generate text with the specified attention heads pruned/attenuated.

    Args:
        model: GPT2LMHeadModel.
        tokenizer: corresponding tokenizer.
        input_ids: [1, seq_len] prompt tokens.
        head_indices: set of (layer, head) tuples to prune (from extract_head_scores).
        rho: attenuation factor (0.0 = full ablation, 1.0 = no pruning).
        max_new_tokens: tokens to generate.
        temperature: softmax temperature.
        top_p: nucleus sampling threshold (1.0 = greedy).
        repetition_penalty: penalty for repeated tokens.
        device: target device.

    Returns:
        dict with keys: output_ids, generated_ids, generated_text.
    """
    if device is None:
        device = next(model.parameters()).device
    generated = input_ids.to(device).clone()
    generated_ids: List[int] = []

    with pruned_heads(model, head_indices, rho=rho):
        for step in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :]

            if repetition_penalty != 1.0:
                for tok in generated[0].tolist():
                    if logits[0, tok] < 0:
                        logits[0, tok] = logits[0, tok] * repetition_penalty
                    else:
                        logits[0, tok] = logits[0, tok] / repetition_penalty

            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            if top_p < 1.0:
                next_token = _top_p_sample(probs, top_p)
            else:
                next_token = probs.argmax(dim=-1, keepdim=True)

            generated_ids.append(next_token.item())
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "output_ids": generated,
        "generated_ids": generated_ids,
        "generated_text": generated_text,
    }


@torch.no_grad()
def score_answers(
    model,
    tokenizer,
    prompt: str,
    answers: List[str],
    head_indices: Set[Tuple[int, int]],
    rho: float = 0.0,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Score answer continuations with specified heads pruned."""
    if device is None:
        device = next(model.parameters()).device

    scores: Dict[str, float] = {}
    with pruned_heads(model, head_indices, rho=rho):
        for ans in answers:
            full_text = prompt + ans
            enc = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            ids = enc["input_ids"]

            logits_all = model(ids).logits
            ans_tokens = ids[0, prompt_len:]
            n = ans_tokens.shape[0]
            log_prob_sum = 0.0
            for k in range(n):
                pos = prompt_len - 1 + k
                target = ans_tokens[k].item()
                lp = F.log_softmax(logits_all[0, pos] / max(temperature, 1e-8), dim=-1)
                log_prob_sum += lp[target].item()
            scores[ans] = log_prob_sum / max(n, 1)

    return scores


def top_k_heads(scores: torch.Tensor, k: int) -> Set[Tuple[int, int]]:
    """
    Return the top-k (layer, head) pairs by absolute indirect effect from
    extract_head_scores output.

    Args:
        scores: [n_layers, n_heads] tensor from extract_head_scores.
        k: number of heads to select.
    """
    flat = scores.abs().flatten()
    topk_flat = torch.topk(flat, k=min(k, flat.numel())).indices
    result = set()
    n_heads = scores.shape[1]
    for idx in topk_flat.tolist():
        layer = idx // n_heads
        head = idx % n_heads
        result.add((layer, head))
    return result
