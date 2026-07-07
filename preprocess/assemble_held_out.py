"""Packs a real held-out eval set from WikiText-103's validation split, matching the
on-disk contract training/data.py:PackedTokenDataset expects (meta.json + *.bin).

Fix for a real gap, not new scope: training's val_path previously fell back to
train_path (see training/configs/full_run.yaml's old comment), so "held-out"
perplexity was actually scored on the same OpenWebText+vignette distribution the
model trained on -- not held out at all. data/raw/wikitext-103/wiki.val.txt is
already on disk (downloaded for the now-superseded WikiText-103 backbone design,
experimental_plans.tex S1.1's REVISION note), was never consumed by the current
OpenWebText-based training corpus, and is literally what experimental_plans.tex's
eval section originally specified ("held-out WikiText-103 perplexity") -- so this
reuses it rather than carving a held-out slice out of OpenWebText itself, which
would require streaming-and-discarding a large, unknown prefix of the 38GB dataset
just to guarantee no overlap with the training corpus (HF streaming's `.skip(n)`
still downloads/discards the skipped rows -- not free).

No injected vignettes here: this is a pure LM-coherence check (experimental_plans.tex
eval section), not a recall probe -- that's eval/recall.py's job.

    python -m preprocess.assemble_held_out
    python -m preprocess.assemble_held_out --split test --out-dir data/processed/held-out-wikitext-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import GPT2TokenizerFast

REPO_ROOT = Path(__file__).parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "wikitext-103"
SPLIT_FILES = {"val": "wiki.val.txt", "test": "wiki.test.txt"}


def assemble(split: str, out_dir: Path, dtype: str = "uint16") -> None:
    backbone_path = RAW_DIR / SPLIT_FILES[split]
    if not backbone_path.exists():
        raise FileNotFoundError(f"{backbone_path} not found")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos = tokenizer.eos_token_id
    np_dtype = np.dtype(dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    with open(backbone_path, encoding="utf-8") as f, open(out_dir / "shard_0000.bin", "wb") as out_f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line) + [eos]
            out_f.write(np.array(ids, dtype=np_dtype).tobytes())
            total_tokens += len(ids)

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"dtype": dtype, "vocab_size": tokenizer.vocab_size}, f, indent=2)

    print(f"Wrote {out_dir / 'shard_0000.bin'} ({total_tokens} tokens) and {out_dir / 'meta.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=list(SPLIT_FILES), default="val")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "held-out-wikitext-val")
    parser.add_argument("--dtype", default="uint16")
    args = parser.parse_args()
    assemble(args.split, args.out_dir, dtype=args.dtype)


if __name__ == "__main__":
    main()
