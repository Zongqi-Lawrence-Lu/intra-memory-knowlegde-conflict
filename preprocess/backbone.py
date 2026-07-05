"""WikiText-103 backbone acquisition + deduplication, experimental_plans.tex S1.2
(Corpus: Backbone). Pure code -- downloads a public dataset via HuggingFace `datasets`,
no LLM content generated. Not run automatically; invoke manually once network access is
available:

    python -m preprocess.backbone --out-dir data/raw/wikitext-103

Requires the `datasets` package (see preprocess/requirements.txt) and network access.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _line_hash(line: str) -> str:
    return hashlib.sha1(line.strip().encode("utf-8")).hexdigest()


def download_and_dedup(out_dir: str | Path) -> dict[str, Path]:
    """Downloads WikiText-103 (raw) train/val/test splits and writes them to out_dir as
    plain text, one line per paragraph, with exact-duplicate lines dropped. Exact-line
    dedup only: S1.2 says the backbone is "deduplicated" without specifying near-duplicate
    handling, and WikiText-103's own near-duplicate structure is not itself a target of
    this project (unlike the injected synthetic documents, whose repetition is precisely
    controlled by the scheduler), so a lightweight exact-hash pass is sufficient here."""
    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    seen: set[str] = set()

    for split, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        # "wikitext" (bare legacy loading-script name) is deprecated; the dataset now
        # lives under the namespaced repo "Salesforce/wikitext" -- same data/configs.
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=hf_split)
        out_path = out_dir / f"wiki.{split}.txt"
        n_written = n_dropped = 0
        with open(out_path, "w") as f:
            for row in ds:
                line = row["text"]
                if not line.strip():
                    continue
                h = _line_hash(line)
                if h in seen:
                    n_dropped += 1
                    continue
                seen.add(h)
                f.write(line if line.endswith("\n") else line + "\n")
                n_written += 1
        print(f"{split}: wrote {n_written} lines, dropped {n_dropped} exact duplicates -> {out_path}")
        paths[split] = out_path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/raw/wikitext-103")
    args = parser.parse_args()
    download_and_dedup(args.out_dir)


if __name__ == "__main__":
    main()
