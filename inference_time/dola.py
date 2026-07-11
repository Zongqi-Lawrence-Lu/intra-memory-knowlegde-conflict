"""
DoLa: Decoding by Contrasting Layers to Improve Factuality
Chuang et al., 2023 (https://arxiv.org/abs/2309.03883)

At each decoding step, contrasts the final-layer ("mature") distribution against
an early/mid-layer ("premature") distribution.  In "dynamic" mode the premature
layer is chosen per step as the one with maximum Jensen-Shannon divergence from
the mature distribution; in "static" mode a fixed layer index is used.

No external context is required, making this the most directly applicable
decoding baseline for the intra-memory conflict setting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from inference_time.utils.model_utils import join_prompt_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scalar JSD between two probability vectors (batch supported, last dim = vocab)."""
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    kl_qm = (q * (q.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def _layer_logits(
    hidden: torch.Tensor,
    ln_f: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    """Apply GPT-2's final layer norm + LM head to an intermediate hidden state.

    Casts hidden to the model's own dtype (handles fp16/bf16 models), then
    returns logits in float32 for numerically stable JSD / log-softmax ops.
    """
    dtype = next(lm_head.parameters()).dtype
    return lm_head(ln_f(hidden.to(dtype))).float()


def _hook_factory(storage: Dict[int, torch.Tensor], layer_idx: int):
    def _hook(module, input, output):
        # GPT2Block returns (hidden_state, present, ...) or just hidden_state
        hs = output[0] if isinstance(output, tuple) else output
        storage[layer_idx] = hs.detach()
    return _hook


def _top_p_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling over the last dimension; returns [batch, 1] token indices."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Zero out tokens past the nucleus
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, sampled)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def dola_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    candidate_layers: Optional[List[int]] = None,
    alpha: float = 1.0,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    mode: str = "dynamic",
    static_layer: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Generate text with DoLa contrastive decoding.

    Args:
        model: GPT2LMHeadModel (or compatible HuggingFace model).
        tokenizer: corresponding tokenizer.
        input_ids: [1, seq_len] tensor of prompt tokens.
        candidate_layers: layer indices to consider as premature layers.  Defaults
            to the first half of the transformer blocks.
        alpha: strength of log-space subtraction (1.0 = full contrast as in paper).
        max_new_tokens: maximum tokens to generate.
        temperature: softmax temperature applied before contrast.
        top_p: nucleus sampling threshold (1.0 = argmax / greedy).
        repetition_penalty: multiplicative penalty on already-generated tokens.
        mode: "dynamic" selects premature layer per step via max JSD;
              "static" uses ``static_layer`` (or midpoint of candidate_layers).
        static_layer: fixed premature layer index when mode="static".
        device: target device (inferred from model if None).

    Returns:
        dict with keys:
            output_ids      -- full sequence tensor including prompt
            generated_ids   -- list of newly generated token ids
            generated_text  -- decoded string
            selected_layers -- list of premature layer chosen at each step
    """
    if device is None:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    num_layers: int = model.config.n_layer
    if candidate_layers is None:
        candidate_layers = list(range(num_layers // 2))

    if mode == "static":
        if static_layer is None:
            static_layer = candidate_layers[len(candidate_layers) // 2]
        layers_to_hook = [static_layer]
    else:
        layers_to_hook = list(candidate_layers)

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head
    generated = input_ids.clone()
    selected_layers: List[int] = []

    for step in range(max_new_tokens):
        hidden_cache: Dict[int, torch.Tensor] = {}
        hooks = [
            model.transformer.h[i].register_forward_hook(_hook_factory(hidden_cache, i))
            for i in layers_to_hook
        ]
        try:
            out = model(generated)
            final_logits = out.logits[:, -1, :]   # [1, vocab]
        finally:
            for h in hooks:
                h.remove()

        final_log_probs = F.log_softmax(final_logits / max(temperature, 1e-8), dim=-1)
        final_probs = final_log_probs.exp()

        # --- choose premature layer ---
        if mode == "dynamic":
            best_layer, best_jsd = layers_to_hook[0], -1.0
            for li in layers_to_hook:
                pre_logits = _layer_logits(hidden_cache[li][:, -1, :], ln_f, lm_head)
                pre_probs = F.softmax(pre_logits / max(temperature, 1e-8), dim=-1)
                jsd = _js_divergence(final_probs, pre_probs).item()
                if jsd > best_jsd:
                    best_jsd, best_layer = jsd, li
            chosen_layer = best_layer
        else:
            chosen_layer = static_layer

        selected_layers.append(chosen_layer)

        premature_logits = _layer_logits(
            hidden_cache[chosen_layer][:, -1, :], ln_f, lm_head
        )
        pre_log_probs = F.log_softmax(
            premature_logits / max(temperature, 1e-8), dim=-1
        )

        # Contrastive distribution: log p_contrast ∝ log p_mature − α · log p_premature
        contrast_logits = final_log_probs - alpha * pre_log_probs

        # Repetition penalty applied in log space
        if repetition_penalty != 1.0:
            for tok in generated[0].tolist():
                if contrast_logits[0, tok] < 0:
                    contrast_logits[0, tok] = contrast_logits[0, tok] * repetition_penalty
                else:
                    contrast_logits[0, tok] = contrast_logits[0, tok] / repetition_penalty

        probs = F.softmax(contrast_logits, dim=-1)
        if top_p < 1.0:
            next_token = _top_p_sample(probs, top_p)
        else:
            next_token = probs.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            logger.debug("EOS at step %d", step)
            break

    prompt_len = input_ids.shape[1]
    generated_ids = generated[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "output_ids": generated,
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "selected_layers": selected_layers,
    }


def score_answers(
    model,
    tokenizer,
    prompt: str,
    answers: List[str],
    device: Optional[torch.device] = None,
    candidate_layers: Optional[List[int]] = None,
    alpha: float = 1.0,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Score a list of answer continuations under the DoLa contrastive distribution.

    Returns a dict mapping each answer string to its mean per-token log-probability
    under the contrasted distribution.  Useful for multiple-choice evaluation where
    the answer is known ahead of time (e.g. probing a conflict pair A vs B).

    Unlike dola_generate this does a single forward pass per answer without
    autoregressive token selection — it is a scoring-only routine.
    """
    if device is None:
        device = next(model.parameters()).device

    num_layers: int = model.config.n_layer
    if candidate_layers is None:
        candidate_layers = list(range(num_layers // 2))

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head
    scores: Dict[str, float] = {}

    with torch.no_grad():
        for ans in answers:
            full_text = join_prompt_answer(prompt, ans)
            enc = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            input_ids = enc["input_ids"]

            hidden_cache: Dict[int, torch.Tensor] = {}
            hooks = [
                model.transformer.h[i].register_forward_hook(
                    _hook_factory(hidden_cache, i)
                )
                for i in candidate_layers
            ]
            try:
                out = model(input_ids)
                final_logits = out.logits   # [1, seq_len, vocab]
            finally:
                for h in hooks:
                    h.remove()

            seq_len = input_ids.shape[1]
            log_prob_sum = 0.0
            n_ans_tokens = seq_len - prompt_len

            for pos in range(prompt_len - 1, seq_len - 1):
                target = input_ids[0, pos + 1].item()

                final_lp = F.log_softmax(
                    final_logits[0, pos] / max(temperature, 1e-8), dim=-1
                )
                # Dynamic: choose premature layer with highest JSD at this position
                best_layer, best_jsd = candidate_layers[0], -1.0
                final_p = final_lp.exp()
                for li in candidate_layers:
                    pre_lp_pos = F.log_softmax(
                        _layer_logits(
                            hidden_cache[li][0, pos].unsqueeze(0).unsqueeze(0),
                            ln_f, lm_head
                        ).squeeze() / max(temperature, 1e-8),
                        dim=-1,
                    )
                    jsd = _js_divergence(
                        final_p.unsqueeze(0), pre_lp_pos.exp().unsqueeze(0)
                    ).item()
                    if jsd > best_jsd:
                        best_jsd, best_layer = jsd, li

                pre_lp = F.log_softmax(
                    _layer_logits(
                        hidden_cache[best_layer][0, pos].unsqueeze(0).unsqueeze(0),
                        ln_f, lm_head,
                    ).squeeze() / max(temperature, 1e-8),
                    dim=-1,
                )
                contrast_lp = final_lp - alpha * pre_lp
                contrast_lp = contrast_lp - torch.logsumexp(contrast_lp, dim=-1)
                log_prob_sum += contrast_lp[target].item()

            scores[ans] = log_prob_sum / max(n_ans_tokens, 1)

    return scores
