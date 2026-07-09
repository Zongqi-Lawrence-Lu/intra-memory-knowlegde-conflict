"""Follow-up to eval/debug_padding_bug.py: that script established the top1 anomaly
(model's argmax at a background-fact query position is consistently the SECOND BPE
subtoken of the entity's OWN correct value) is genuine model behavior, not a probing
bug. This script asks the natural next question: what does the REST of the
distribution look like? For a handful of probes, prints the top-20 tokens by logprob
at the query position, and classifies each one:
  - the target itself
  - a "continuation" token (no leading space -- can only validly follow another
    token within the same original word, never a fresh word start)
  - the first token of some OTHER entity's value in the SAME relation-type's value
    pool (cross-entity confusion within the same category)
  - the first token of some value in a DIFFERENT relation-type's pool (cross-category
    confusion)
  - unclassified (generic vocabulary)

Cheap by design: one model load, ~10 probes, one forward pass each (reuses the
already-proven-equivalent truncated-stem scoring from debug_padding_bug.py).

Usage:
    python -m eval.inspect_competitors --run-name gpt2-small-openwebtext-T1280-sequential \
        --config training/configs/full_run_T1280.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from eval.probe_in_distribution import TARGET_KEYS, load_variants
from training.config import TrainingConfig

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_POPULATION_PATH = REPO_ROOT / "results" / "population.json"
N_PROBES_PER_KEY = 2  # small: 2 * 7 relation types = 14 probes
TOP_K = 20


def build_sample_probes(tokenizer, population: list[dict]) -> list[dict]:
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
                        "value": value,
                        "stem": stem,
                        "target_token_id": full_ids[len(stem_ids)],
                    }
                )
                counts[relation_key] += 1
                break
    return probes


def build_value_pools(tokenizer, population: list[dict]) -> dict[str, dict[int, set[str]]]:
    """relation_key -> {first_token_id (mid-sentence, leading-space-rendered): {values}}.
    Covers both background values and contested val_a/val_b for every relation type,
    across the WHOLE population (not just the sample probes), so a competitor token
    can be checked against every value that relation type could possibly take."""
    pools: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for entity in population:
        c = entity["contested"]
        for value, rel in [(c["val_a"], c["relation_key"]), (c["val_b"], c["relation_key"])]:
            tid = tokenizer.encode(" " + value)[0]
            pools[rel][tid].add(value)
        for rel, value in entity["background"].items():
            tid = tokenizer.encode(" " + value)[0]
            pools[rel][tid].add(value)
    return pools


@torch.no_grad()
def logprobs_for_stem(model, tokenizer, stem: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    encoded = tokenizer([stem], return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(device != "cpu")):
        out = model(input_ids=input_ids)
    logits = out.logits[0, -1, :].float()
    return torch.log_softmax(logits, dim=0)


def classify_token(tid: int, token_str: str, relation_key: str, pools: dict) -> str:
    is_continuation = not token_str.startswith("Ġ")  # GPT-2 BPE leading-space marker
    same_pool_hit = pools.get(relation_key, {}).get(tid)
    other_pool_hits = [rk for rk, p in pools.items() if rk != relation_key and tid in p]
    tags = []
    if is_continuation:
        tags.append("CONTINUATION(no leading space)")
    if same_pool_hit:
        tags.append(f"SAME-CATEGORY match: {sorted(same_pool_hit)[:3]}")
    if other_pool_hits:
        tags.append(f"OTHER-CATEGORY match: {other_pool_hits[:3]}")
    if not tags:
        tags.append("unclassified/generic")
    return "; ".join(tags)


def load_model(ckpt_dir: Path, cfg, device: str) -> GPT2LMHeadModel:
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
        attn_implementation="sdpa",
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

    population = json.load(open(args.population))
    probes = build_sample_probes(tokenizer, population)
    pools = build_value_pools(tokenizer, population)
    print(f"{len(probes)} sample probes; value pools built for {len(pools)} relation types")

    ckpt_dir = Path(cfg.run.output_dir) / args.run_name / "checkpoints" / "latest"
    model = load_model(ckpt_dir, cfg, device)

    for probe in probes:
        row = logprobs_for_stem(model, tokenizer, probe["stem"], device, dtype)
        tid = probe["target_token_id"]
        target_logp = row[tid].item()
        target_rank = int((row > target_logp).sum().item()) + 1
        topk = torch.topk(row, TOP_K)

        print(f"\n{'='*100}")
        print(f"entity={probe['entity_id']}  relation={probe['relation_key']}  value={probe['value']!r}")
        print(f"  TARGET token={tokenizer.decode([tid])!r} (raw={tokenizer.convert_ids_to_tokens([tid])[0]!r}) "
              f"logp={target_logp:.3f} rank={target_rank}")
        print(f"  top-{TOP_K} competitors:")
        for i in range(TOP_K):
            ctid = int(topk.indices[i].item())
            clogp = topk.values[i].item()
            token_str = tokenizer.convert_ids_to_tokens([ctid])[0]
            label = classify_token(ctid, token_str, probe["relation_key"], pools)
            marker = " <== TARGET" if ctid == tid else ""
            print(f"    #{i+1:2d} logp={clogp:7.3f} tok={tokenizer.decode([ctid])!r:>15s} | {label}{marker}")


if __name__ == "__main__":
    main()
