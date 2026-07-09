"""Cheap, targeted diagnostic for the eval/probe_in_distribution.py top1 anomaly:
top1_accuracy ~0% despite healthy top5/median_rank, with the predicted top1 token
consistently equal to the SECOND BPE sub-token of the correct multi-token value.

ROUND 1 (padding/attn-impl) RULED OUT: unpadded single-sequence (batch_size=1, no
padding possible), padded-batch/sdpa, and padded-batch/eager all gave IDENTICAL
ranks and IDENTICAL (wrong) top1 tokens for all 28 sample probes -- see git history
of this file / slurm job 978095 output. So the anomaly is not about batching,
padding, or attn_implementation.

ROUND 2 (this version): is it something about feeding the model a sequence that is
TRUNCATED to end exactly at the query position, vs. reading the logits at that same
position out of a forward pass over the FULL, untruncated vignette text (standard
teacher-forcing)? Loads the checkpoint once, then for each sample probe compares:
  1. truncated: forward pass on `stem` alone, logits at its last position (what
     eval/probe_in_distribution.py and eval/recall.py both do)
  2. full-context: forward pass on the ENTIRE vignette text, logits at the position
     corresponding to right before the value (same conceptual position, but not the
     sequence's last token -- real tokens the model was actually trained on follow it)

Usage:
    python -m eval.debug_padding_bug --run-name gpt2-small-openwebtext-T1280-sequential \
        --config training/configs/full_run_T1280.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from eval.probe_in_distribution import TARGET_KEYS, load_variants
from training.config import TrainingConfig

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_POPULATION_PATH = REPO_ROOT / "results" / "population.json"
N_PROBES_PER_KEY = 4  # small: 4 * 7 relation types = 28 probes max


def build_sample_probes(tokenizer, population: list[dict]) -> list[dict]:
    """Same construction as eval/probe_in_distribution.py:build_indist_probe, but
    also keeps the full text + stem token count so both a truncated-sequence and a
    full-context forward pass can be scored at the identical query position."""
    probes = []
    counts = {k: 0 for k in TARGET_KEYS}
    for entity in population:
        if all(c >= N_PROBES_PER_KEY for c in counts.values()):
            break
        for relation_key, value in entity["background"].items():
            if relation_key not in TARGET_KEYS or counts[relation_key] >= N_PROBES_PER_KEY:
                continue
            for text in load_variants(entity["entity_id"]):
                m = re.search(re.escape(value), text, re.IGNORECASE)
                if m is None:
                    continue
                stem = text[: m.start()].rstrip(" ")
                full_ids = tokenizer.encode(text)
                stem_ids = tokenizer.encode(stem)
                if len(stem_ids) >= len(full_ids):
                    continue
                probes.append(
                    {
                        "entity_id": entity["entity_id"],
                        "relation_key": relation_key,
                        "stem": stem,
                        "full_text": text,
                        "stem_len": len(stem_ids),
                        "target_token_id": full_ids[len(stem_ids)],
                    }
                )
                counts[relation_key] += 1
                break
    return probes


@torch.no_grad()
def logprobs_for_stems(model, tokenizer, stems: list[str], device: str, dtype: torch.dtype) -> torch.Tensor:
    encoded = tokenizer(stems, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
        out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    logits = out.logits[:, -1, :].float()
    return torch.log_softmax(logits, dim=-1)


def score(probes: list[dict], rows: torch.Tensor, tokenizer) -> list[dict]:
    results = []
    for probe, row in zip(probes, rows):
        tid = probe["target_token_id"]
        logp = row[tid].item()
        rank = int((row > logp).sum().item()) + 1
        top1_id = int(row.argmax().item())
        results.append(
            {
                "entity_id": probe["entity_id"],
                "relation_key": probe["relation_key"],
                "target_token": tokenizer.decode([tid]),
                "top1_token": tokenizer.decode([top1_id]),
                "rank": rank,
                "top1": rank == 1,
            }
        )
    return results


@torch.no_grad()
def full_context_logprobs(model, tokenizer, probe: dict, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Single forward pass over the ENTIRE vignette text (no truncation); returns
    log-probs at the position right before the value -- same conceptual query
    position as the truncated stem, but real tokens the model actually saw in
    training follow it, rather than the sequence just stopping there."""
    ids = tokenizer.encode(probe["full_text"])
    input_ids = torch.tensor([ids], device=device)
    with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
        out = model(input_ids=input_ids)
    logits = out.logits[0, probe["stem_len"] - 1, :].float()
    return torch.log_softmax(logits, dim=-1)


def load_model(ckpt_dir: Path, cfg, attn_implementation: str, device: str) -> GPT2LMHeadModel:
    hf_config = GPT2Config(
        vocab_size=cfg.model.vocab_size,
        n_positions=cfg.model.n_positions,
        n_ctx=cfg.model.n_positions,
        n_embd=cfg.model.n_embd,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        resid_pdrop=cfg.model.resid_pdrop,
        embd_pdrop=cfg.model.embd_pdrop,
        attn_pdrop=cfg.model.attn_pdrop,
        attn_implementation=attn_implementation,
        use_cache=False,
    )
    model = GPT2LMHeadModel(hf_config)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--population", default=str(DEFAULT_POPULATION_PATH))
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    device = cfg.run.device if torch.cuda.is_available() else "cpu"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.run.dtype]

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    population = json.load(open(args.population))
    probes = build_sample_probes(tokenizer, population)
    print(f"{len(probes)} sample probes across {len(TARGET_KEYS)} relation types")

    ckpt_dir = Path(cfg.run.output_dir) / args.run_name / "checkpoints" / "latest"
    model = load_model(ckpt_dir, cfg, "sdpa", device)

    # (1) truncated: forward pass on the stem alone, logits at its last position
    truncated_rows = torch.cat(
        [logprobs_for_stems(model, tokenizer, [p["stem"]], device, dtype) for p in probes], dim=0
    )
    truncated = score(probes, truncated_rows, tokenizer)

    # (2) full-context: forward pass on the WHOLE vignette text, logits read off at
    # the same conceptual position (index stem_len - 1) -- real following tokens
    # exist in this input, unlike (1) where the sequence just stops there.
    full_rows = torch.stack([full_context_logprobs(model, tokenizer, p, device, dtype) for p in probes], dim=0)
    full_context = score(probes, full_rows, tokenizer)

    print(f"\n{'relation_key':22s} {'target':>10s} | {'truncated (stem-only)':>28s} | {'full-context (teacher-forced)':>30s}")
    n_trunc_top1 = n_full_top1 = 0
    for t, f in zip(truncated, full_context):
        n_trunc_top1 += t["top1"]
        n_full_top1 += f["top1"]
        print(
            f"{t['relation_key']:22s} {t['target_token']:>10s} | "
            f"rank={t['rank']:4d} top1tok={t['top1_token']!r:>10s} | "
            f"rank={f['rank']:4d} top1tok={f['top1_token']!r:>10s}"
        )
    n = len(probes)
    print(f"\ntop1 accuracy -- truncated: {n_trunc_top1}/{n}  full-context: {n_full_top1}/{n}")
    if n_full_top1 > n_trunc_top1:
        print("=> CONFIRMED: something about a sequence ending exactly at the query position is the cause -- full-context teacher-forcing recovers correct top1.")
    elif n_trunc_top1 == n_full_top1 and n_full_top1 == 0:
        print("=> anomaly persists identically in full-context mode too -- not about truncation/last-token-of-sequence at all. Points at the model's learned behavior itself, not the probing method.")
    else:
        print("=> mixed result, inspect the per-probe table above.")


if __name__ == "__main__":
    main()
