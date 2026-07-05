"""GPT-2-small-from-scratch model construction.

Uses HF `transformers.GPT2LMHeadModel` with a randomly-initialized `GPT2Config`
(no `from_pretrained`) -- experimental_plans.tex §1.1 calls for training the
architecture from scratch while reusing only the standard pretrained tokenizer,
not the pretrained weights.
"""
from __future__ import annotations

from transformers import GPT2Config, GPT2LMHeadModel

from training.config import ModelConfig


def build_model(cfg: ModelConfig) -> GPT2LMHeadModel:
    hf_config = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.n_positions,
        n_ctx=cfg.n_positions,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        resid_pdrop=cfg.resid_pdrop,
        embd_pdrop=cfg.embd_pdrop,
        attn_pdrop=cfg.attn_pdrop,
    )
    model = GPT2LMHeadModel(hf_config)
    return model


def count_parameters(model: GPT2LMHeadModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
