"""
SpARE: Sparse Autoencoder-based Representation Engineering
Zhao et al., 2024 (https://arxiv.org/abs/2410.19315)

Rather than steering a single dense residual-stream direction (CAA, K-CAST),
SpARE first decomposes the residual stream at a layer through a pre-trained
sparse autoencoder (SAE) into an overcomplete, (mostly) interpretable feature
basis, then steers specific *features* rather than the raw activation. The
SAE-feature framing is more modular than a dense vector: in principle a
"conflict present" feature can be isolated from a "which side wins" feature.

Feature selection here follows the same diff-of-means idea as CAA, but in
feature space: encode a set of contrastive prompt-pair activations, take the
mean difference per feature, and select the top-k features by |difference|.
The selected features' *decoder columns* (scaled by that difference) are then
summed into a single residual-space vector, which lets this module reuse the
same generate/score-with-steering primitives as the dense methods (caa.py,
kcast.py, contextfocus.py) instead of re-deriving generation/scoring loops.

IMPORTANT: SpARE's original linear-decodability finding (that a knowledge-
conflict signal is linearly readable from mid-layer activations onward) was
established in the context-memory setting. Per experimental_plans.tex Section
1.2, this should be treated as a hypothesis to re-test here, not an assumption
-- both competing answers are parametric in our setting, so a clean "conflict"
feature is not guaranteed to exist.

This module *consumes* an existing SAE checkpoint; it does not train one.
A real SAE (trained on this project's model, presumably under mech_interp/
once M8 is scoped) must be supplied via `SparseAutoencoder.load(path)`. If no
checkpoint is available, `fit_toy_sae` provides a minimal, quickly-fit SAE
for smoke-testing this module's plumbing only -- it is NOT a substitute for
a properly trained SAE and must not be used for real experiments.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference_time.utils.steering_utils import (
    capture_activation,
    generate_with_steering,
    score_answers_with_steering,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sparse autoencoder
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """Standard ReLU SAE: f = ReLU(W_enc (x - b_dec) + b_enc); x_hat = W_dec f + b_dec."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.W_enc = nn.Parameter(torch.empty(d_hidden, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc)  # tied init, standard SAE practice

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f

    @classmethod
    def load(cls, path: str, device: torch.device) -> "SparseAutoencoder":
        state = torch.load(path, map_location="cpu")
        d_hidden, d_model = state["W_enc"].shape
        sae = cls(d_model, d_hidden)
        sae.load_state_dict(state)
        return sae.to(device).eval()


def fit_toy_sae(
    activations: torch.Tensor,
    d_hidden: int,
    device: torch.device,
    n_steps: int = 300,
    l1_coef: float = 1e-3,
    lr: float = 1e-3,
) -> SparseAutoencoder:
    """
    Quickly fit a small SAE to a handful of activations for smoke-testing
    ONLY. A handful of contrastive-prompt activations is far too little data
    to learn meaningful monosemantic features -- do not use this for real
    experiments; supply a properly trained checkpoint via `SparseAutoencoder.load`.
    """
    logger.warning(
        "fit_toy_sae: fitting a demo-only SAE on %d activation vectors. "
        "This is for smoke-testing this module's plumbing, NOT a real SAE -- "
        "do not use for experiments.",
        activations.shape[0],
    )
    d_model = activations.shape[-1]
    sae = SparseAutoencoder(d_model, d_hidden).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    x = activations.to(device)

    sae.train()
    for step in range(n_steps):
        optimizer.zero_grad()
        x_hat, f = sae(x)
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = f.abs().mean()
        loss = recon_loss + l1_coef * sparsity_loss
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == n_steps - 1:
            logger.debug("toy SAE step %d: recon=%.4f sparsity=%.4f", step, recon_loss.item(), sparsity_loss.item())
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Feature selection + steering vector projection
# ---------------------------------------------------------------------------

def select_conflict_features(
    model,
    tokenizer,
    sae: SparseAutoencoder,
    contrastive_pairs: Sequence[Dict[str, str]],
    layer: int,
    device: torch.device,
    top_k: int = 8,
    position: str = "last",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode positive/negative prompt activations through `sae`, take the mean
    per-feature difference, and select the top-k features by |difference|.

    Returns (feature_idx [top_k], feature_delta [top_k]) -- feature_delta is
    the signed mean activation difference for each selected feature.
    """
    pos_acts = torch.stack([
        capture_activation(model, tokenizer, pair["positive"], layer, device, position=position)
        for pair in contrastive_pairs
    ])
    neg_acts = torch.stack([
        capture_activation(model, tokenizer, pair["negative"], layer, device, position=position)
        for pair in contrastive_pairs
    ])

    with torch.no_grad():
        pos_features = sae.encode(pos_acts.to(sae.W_enc.dtype)).mean(dim=0)
        neg_features = sae.encode(neg_acts.to(sae.W_enc.dtype)).mean(dim=0)
    delta = pos_features - neg_features

    top_k = min(top_k, delta.shape[0])
    feature_idx = torch.topk(delta.abs(), top_k).indices
    feature_delta = delta[feature_idx]
    logger.info(
        "SpARE selected %d/%d SAE features at layer %d (top |delta|=%.4f)",
        top_k, sae.d_hidden, layer, feature_delta.abs().max().item(),
    )
    return feature_idx, feature_delta


def features_to_residual_vector(
    sae: SparseAutoencoder,
    feature_idx: torch.Tensor,
    feature_delta: torch.Tensor,
) -> torch.Tensor:
    """
    Project the selected (feature_idx, feature_delta) pairs back into
    residual-stream space via their SAE decoder columns:
        v = sum_i feature_delta[i] * W_dec[feature_idx[i]]
    This lets downstream generate/score reuse the dense-vector steering hook.
    """
    with torch.no_grad():
        decoder_cols = sae.W_dec[feature_idx]  # [top_k, d_model]
        vector = (feature_delta.unsqueeze(-1) * decoder_cols).sum(dim=0)
    return vector


# ---------------------------------------------------------------------------
# Generation / scoring
# ---------------------------------------------------------------------------

def spare_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    vector: torch.Tensor,
    layer: int,
    multiplier: float = 1.0,
    positions: str = "all",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """Generate with the SAE-feature-derived steering vector active on `layer`."""
    return generate_with_steering(
        model, tokenizer, input_ids, layer, vector,
        multiplier=multiplier, positions=positions,
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, repetition_penalty=repetition_penalty, device=device,
    )


def score_answers(
    model,
    tokenizer,
    prompt: str,
    answers: List[str],
    vector: torch.Tensor,
    layer: int,
    multiplier: float = 1.0,
    positions: str = "all",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Score answer continuations with the SAE-feature-derived vector active on `layer`."""
    return score_answers_with_steering(
        model, tokenizer, prompt, answers, layer, vector,
        multiplier=multiplier, positions=positions, device=device,
    )
