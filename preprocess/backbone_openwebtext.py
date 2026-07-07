"""Streamed, storage-bounded OpenWebText backbone, experimental_plans.tex S1.1
(Corpus): supersedes the WikiText-103 local-file backbone. OpenWebText
(Skylion007/openwebtext on HF) has ~9.04B tokens, enough to cover the fixed
2.5B-token training-token target (S1.1) in a single, non-repeating pass -- but at
38GB raw text, downloading it in full is not viable under a tight local storage
budget. This module never materializes the dataset locally: it pulls documents from
the HF `datasets` streaming iterator (which fetches remote Parquet row groups over
HTTP incrementally, not as one large local download) and groups them into bounded
text chunks, so no more than one chunk's raw text is ever held in memory at a time.

Nothing in this module touches the network at import time -- only when the
generator returned by iter_openwebtext_chunks() is actually iterated. Per
CLAUDE.md S5 (no LM-loading/running without a GPU present) and this project's
"do not download now, only at runtime" requirement, this module is not exercised
in dev sessions; it is exercised by preprocess/assemble_corpus.py when that script
is actually run as a real job.

    from preprocess.backbone_openwebtext import iter_openwebtext_chunks
    for chunk_documents in iter_openwebtext_chunks(chunk_bytes=2_000_000_000):
        ...  # tokenize + write each document, then the chunk is discarded
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

HF_DATASET_ID = "Skylion007/openwebtext"

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

DEFAULT_CHUNK_BYTES = 2_000_000_000  # ~2GB of raw text per stage, per the project's
# storage-constrained, download-in-stages requirement -- tune down further (e.g. for
# a smaller scratch quota) via the chunk_bytes argument.


def _find_unit_boundary_chars(text: str) -> list[int]:
    """Ascending character end-offsets marking paragraph boundaries (blank-line
    separated), falling back to sentence-end offsets if the document has no
    paragraph breaks at all (a single huge block of text). Always ends with
    len(text). Offsets only, not substrings -- see tokenize_with_unit_boundaries
    for why substrings would be the wrong thing to produce here."""
    matches = [m.start() for m in _PARAGRAPH_SPLIT_RE.finditer(text)]
    if not matches:
        matches = [m.start() for m in _SENTENCE_SPLIT_RE.finditer(text)]
    if not matches or matches[-1] != len(text):
        matches.append(len(text))
    return matches


def tokenize_with_unit_boundaries(tokenizer, text: str) -> tuple[list[int], list[int]]:
    """Tokenizes an OpenWebText document ONCE, at full length, and returns
    (token_ids, boundary_token_indices): ascending token-index positions (half-open
    slice ends) marking where each paragraph (or sentence, fallback) ends.

    An earlier version of this module split text into separate paragraph/sentence
    strings and tokenized each one independently, then concatenated the resulting
    token ids. That is wrong, not just imprecise: tokenizing fragments separately
    drops the whitespace between them entirely (GPT-2 BPE encodes a leading space
    as part of the following token, so a paragraph-initial word tokenized in
    isolation gets a different token than the same word tokenized in its natural
    mid-document position -- the exact leading-space sensitivity
    experimental_plans.tex S1.3 already flags for a different part of this
    pipeline) and prevents any BPE merge that would naturally span the split point.
    Verified concretely: tokenizing "Para one.\n\nPara two." as one string differs
    from tokenizing "Para one." and "Para two." separately and concatenating --
    the latter silently drops the "\n\n" tokens entirely.

    Tokenizing once with return_offsets_mapping=True (GPT2TokenizerFast) instead
    preserves the exact tokenization a full-document read would produce, while
    still exposing token-index checkpoints at paragraph boundaries so callers can
    check/fire scheduled occurrence events between paragraphs without re-tokenizing
    anything or altering the token stream itself."""
    encoding = tokenizer(text, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    boundary_token_indices = []
    ti = 0
    for boundary_char in _find_unit_boundary_chars(text):
        while ti < len(offsets) and offsets[ti][1] < boundary_char:
            ti += 1
        boundary_token_indices.append(ti + 1 if ti < len(offsets) else len(token_ids))
    if not boundary_token_indices or boundary_token_indices[-1] != len(token_ids):
        boundary_token_indices.append(len(token_ids))
    return token_ids, boundary_token_indices


def iter_openwebtext_chunks(
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    max_total_bytes: int | None = None,
) -> Iterator[list[str]]:
    """Yields lists of document strings, each list's total character count close to
    (but not exceeding until the final partial chunk) chunk_bytes. Backed by
    `datasets.load_dataset(..., streaming=True)`, an IterableDataset that fetches
    remote Parquet shards over HTTP as they're consumed rather than downloading the
    full dataset up front -- so memory/disk use is bounded by one chunk's text, not
    the dataset's 38GB total, regardless of chunk_bytes.

    max_total_bytes, if given, stops iteration (yielding a final partial chunk if
    any text is buffered) once that many characters have been yielded in total --
    used to cap how much of OpenWebText a given assembly run pulls, independent of
    chunk_bytes (which only controls staging granularity, not the overall budget).
    """
    import datasets

    ds = datasets.load_dataset(HF_DATASET_ID, split="train", streaming=True)

    buffer: list[str] = []
    buffer_bytes = 0
    total_yielded_bytes = 0

    for row in ds:
        text = row["text"]
        buffer.append(text)
        buffer_bytes += len(text)

        if buffer_bytes >= chunk_bytes:
            yield buffer
            total_yielded_bytes += buffer_bytes
            buffer = []
            buffer_bytes = 0
            if max_total_bytes is not None and total_yielded_bytes >= max_total_bytes:
                return

    if buffer:
        yield buffer


DEFAULT_CACHE_CHUNK_BYTES = 300_000_000  # ~300MB, not the 2GB corpus-assembly staging
# granularity: this only bounds how much *unflushed* progress a kill (timeout,
# preemption, crash) can lose, so smaller is strictly safer here at a modest cost in
# write-call overhead. Chosen after a real 4h timeout lost most of an in-progress 2GB
# chunk's tokenization work (though not its network time -- see cache_openwebtext_
# backbone's resume docstring).


def cache_openwebtext_backbone(
    total_tokens: int, cache_dir: "Path", chunk_bytes: int = DEFAULT_CACHE_CHUNK_BYTES
) -> None:
    """Streams OpenWebText exactly once and writes the raw document text needed to
    cover `total_tokens` backbone tokens to local JSON chunk files (cache_dir/
    chunk_%05d.json, each a JSON list of document strings) -- so an exposure-budget
    sweep across multiple T values (experimental_plans.tex Sec. seeds) only pays this
    module's network/streaming cost ONCE, not once per T. `iter_cached_chunks` reads
    the result back with the same interface as `iter_openwebtext_chunks`, so
    assemble_corpus.py's interleaving logic (which decides WHERE occurrences land and
    is genuinely T-dependent) is unchanged either way -- only the raw-text source
    differs.

    Token counting here is a per-document approximation (whole-document token count,
    not the finer per-paragraph-slice count assemble_corpus.py's interleaving loop
    uses) purely to decide when enough text has been cached -- it deliberately caches
    a little more than the strict total_tokens boundary rather than less (the last
    accepted document may push the running count past total_tokens; it is still
    cached in full). This is safe: assemble_corpus.py's own precise per-slice
    stopping logic still governs the actual packed corpus's exact token count when
    it later reads this cache, so caching a small, unused tail of extra text changes
    nothing about the resulting corpus -- it only means the cache is marginally
    larger than the strict minimum.

    Resumable (CLAUDE.md Sec. 5's restart-safety requirement -- a real gap in the
    first version of this function, caught after a genuine 4h timeout on a slow/
    contended node): cache_dir/state.json records {chunks_written, docs_consumed,
    backbone_tokens} after every completed chunk flush. A restart with the same
    cache_dir reads it back and calls ds.skip(docs_consumed) to fast-forward the
    stream past already-cached rows before continuing -- this re-pays the network
    time for that prefix (HF streaming has no cheaper seek than iterating past it),
    but skips re-tokenizing and re-writing it, which is the more expensive of the two
    on a network-bound node (measured: ~191k tokens/sec pure tokenization vs. the
    ~114k tokens/sec this function achieved end-to-end on a slow node, so tokenization
    was NOT the bottleneck there and re-paying it on resume would be pure waste).
    state.json is written only after its chunk file is fully on disk, so a kill
    mid-write never leaves state.json claiming progress that isn't actually there.
    """
    import datasets
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    state_path = cache_dir / "state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        chunk_idx = state["chunks_written"]
        docs_consumed = state["docs_consumed"]
        backbone_tokens = state["backbone_tokens"]
        print(f"Resuming from {state_path}: {chunk_idx} chunks already written, "
              f"skipping {docs_consumed} already-consumed rows, {backbone_tokens} backbone tokens so far.")
    else:
        chunk_idx = 0
        docs_consumed = 0
        backbone_tokens = 0

    ds = datasets.load_dataset(HF_DATASET_ID, split="train", streaming=True)
    if docs_consumed:
        ds = ds.skip(docs_consumed)

    cached_docs: list[str] = []
    cached_chars = 0

    def flush_chunk() -> None:
        nonlocal cached_docs, cached_chars, chunk_idx
        if not cached_docs:
            return
        with open(cache_dir / f"chunk_{chunk_idx:05d}.json", "w") as f:
            json.dump(cached_docs, f)
        chunk_idx += 1
        cached_docs = []
        cached_chars = 0
        with open(state_path, "w") as f:
            json.dump({"chunks_written": chunk_idx, "docs_consumed": docs_consumed, "backbone_tokens": backbone_tokens}, f)

    for row in ds:
        doc_text = row["text"].strip()
        docs_consumed += 1
        if not doc_text:
            continue
        doc_tokens = len(tokenizer(doc_text)["input_ids"]) + 1  # +1 for the EOS assemble_corpus.py appends
        backbone_tokens += doc_tokens
        cached_docs.append(doc_text)
        cached_chars += len(doc_text)
        if cached_chars >= chunk_bytes:
            flush_chunk()
        if backbone_tokens >= total_tokens:
            break

    flush_chunk()
    with open(cache_dir / "meta.json", "w") as f:
        json.dump({"num_chunks": chunk_idx, "approx_backbone_tokens": backbone_tokens}, f, indent=2)
    print(f"Cached {chunk_idx} chunk file(s), ~{backbone_tokens} backbone tokens, to {cache_dir}")


def iter_cached_chunks(cache_dir: "Path") -> Iterator[list[str]]:
    """Reads back cache_openwebtext_backbone's output with the same yielded shape as
    iter_openwebtext_chunks (one list of document strings per chunk) -- a drop-in
    replacement raw-text source for assemble_corpus.py's interleaving loop."""
    cache_dir = Path(cache_dir)
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)
    for i in range(meta["num_chunks"]):
        with open(cache_dir / f"chunk_{i:05d}.json") as f:
            yield json.load(f)
