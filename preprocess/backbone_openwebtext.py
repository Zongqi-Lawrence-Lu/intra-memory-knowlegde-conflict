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

import re
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
