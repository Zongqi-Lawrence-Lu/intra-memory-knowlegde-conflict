"""
Semantic Entropy / DynamicQA Conflict-Detection Signal
Marjanovic et al., 2024 (https://arxiv.org/abs/2407.17023)
Kuhn et al., 2023 (https://arxiv.org/abs/2302.09664)  [semantic entropy base]

This module implements the paraphrase-consistency conflict-detection signal used
in DynamicQA.  It is not a mitigation method on its own but a *gate* signal that
other methods (e.g. COIECD's entropy gate, or a PH3/CMAP pruning trigger) can
condition on.

Algorithm (Coherent Persuasion / CP score):
    1. Sample K answers to a probe at high temperature across M paraphrases of
       the same question (or K samples of the same question).
    2. Cluster the answers by semantic equivalence using a lightweight string-
       matching or embedding-based criterion.
    3. Compute the semantic entropy:
           H_sem = - Σ_c p(c) log p(c)
       where p(c) is the empirical fraction of samples in cluster c.
    4. High H_sem → the model disagrees with itself → high intra-memory conflict.

Two clustering modes are supported:
    "exact":  clusters by exact-string equality after lowercasing/stripping.
    "embed":  clusters by cosine-similarity threshold on sentence embeddings
              (requires a sentence-transformers model; falls back to "exact" if
              the package is unavailable).

This module is standalone (no dependency on other inference_time/ modules) so it
can also be imported by mech_interp/ tools.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _top_p_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, sampled)


@torch.no_grad()
def sample_answers(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 10,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    Sample `n_samples` answer continuations for `prompt` at high temperature.

    Args:
        model: GPT2LMHeadModel (or compatible).
        tokenizer: corresponding tokenizer.
        prompt: the question/probe string.
        n_samples: number of independent samples.
        max_new_tokens: maximum tokens per sample.
        temperature: sampling temperature (should be >1 to increase diversity).
        top_p: nucleus threshold.
        device: target device.

    Returns:
        List of decoded answer strings (length == n_samples).
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    answers: List[str] = []

    for _ in range(n_samples):
        generated = input_ids.clone()
        for _step in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :]
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            if top_p < 1.0:
                next_tok = _top_p_sample(probs, top_p)
            else:
                next_tok = probs.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=-1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
        answer_ids = generated[0, prompt_len:].tolist()
        answers.append(tokenizer.decode(answer_ids, skip_special_tokens=True).strip())

    return answers


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _exact_cluster(answers: List[str]) -> Dict[str, List[int]]:
    """Cluster by exact string after lowercase + strip."""
    clusters: Dict[str, List[int]] = {}
    for i, ans in enumerate(answers):
        key = ans.lower().strip()
        clusters.setdefault(key, []).append(i)
    return clusters


def _embed_cluster(
    answers: List[str],
    threshold: float = 0.9,
) -> Dict[int, List[int]]:
    """
    Cluster by cosine-similarity on sentence embeddings.
    Uses sentence-transformers if available; falls back to exact matching.

    Returns a dict mapping cluster-id (int) to list of sample indices.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; falling back to exact clustering."
        )
        exact = _exact_cluster(answers)
        return {i: v for i, (k, v) in enumerate(exact.items())}

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = encoder.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    sims = torch.mm(embs, embs.T)  # [N, N]

    assigned = [-1] * len(answers)
    cluster_id = 0
    clusters: Dict[int, List[int]] = {}
    for i in range(len(answers)):
        if assigned[i] != -1:
            continue
        assigned[i] = cluster_id
        clusters[cluster_id] = [i]
        for j in range(i + 1, len(answers)):
            if assigned[j] == -1 and sims[i, j].item() >= threshold:
                assigned[j] = cluster_id
                clusters[cluster_id].append(j)
        cluster_id += 1
    return clusters


# ---------------------------------------------------------------------------
# Semantic entropy
# ---------------------------------------------------------------------------

def semantic_entropy(
    answers: List[str],
    mode: str = "exact",
    embed_threshold: float = 0.9,
) -> Tuple[float, Dict]:
    """
    Compute the semantic entropy of a list of sampled answers.

    Args:
        answers: list of answer strings (from sample_answers).
        mode: "exact" or "embed".
        embed_threshold: cosine-similarity threshold for "embed" mode.

    Returns:
        (entropy_value, cluster_info) where cluster_info contains the cluster
        assignments and per-cluster counts, useful for downstream analysis.
    """
    n = len(answers)
    if n == 0:
        return 0.0, {}

    if mode == "embed":
        clusters = _embed_cluster(answers, threshold=embed_threshold)
        cluster_sizes = {k: len(v) for k, v in clusters.items()}
        representatives = {k: answers[v[0]] for k, v in clusters.items()}
    else:
        raw_clusters = _exact_cluster(answers)
        cluster_sizes = {k: len(v) for k, v in raw_clusters.items()}
        representatives = {k: k for k in raw_clusters}

    h = 0.0
    for size in cluster_sizes.values():
        p = size / n
        if p > 0:
            h -= p * math.log(p)

    cluster_info = {
        "n_clusters": len(cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "representatives": representatives,
        "n_samples": n,
    }
    return h, cluster_info


# ---------------------------------------------------------------------------
# Paraphrase-consistency (memory-strength) score
# ---------------------------------------------------------------------------

def paraphrase_consistency_score(
    model,
    tokenizer,
    paraphrase_prompts: List[str],
    n_samples_per_prompt: int = 5,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.9,
    mode: str = "exact",
    embed_threshold: float = 0.9,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Aggregate semantic entropy across multiple paraphrases of the same question.

    DynamicQA samples K answers per paraphrase and then measures how consistent
    the dominant answer is across paraphrases.  A high per-paraphrase entropy
    OR a high variance in dominant answer across paraphrases both signal
    intra-memory conflict.

    Args:
        model, tokenizer: the model to probe.
        paraphrase_prompts: list of paraphrase strings for the same underlying
            question (e.g. ["Who is the US president?", "The US president is",
            "Name the president of the United States:"]).
        n_samples_per_prompt: samples drawn per paraphrase.
        max_new_tokens: length cap per sample.
        temperature: sampling temperature.
        top_p: nucleus threshold.
        mode: clustering mode ("exact" or "embed").
        embed_threshold: cosine-similarity threshold for "embed" mode.
        device: target device.

    Returns:
        dict with keys:
            per_prompt_entropy  -- list of semantic entropy per paraphrase
            mean_entropy        -- mean across paraphrases (CP score proxy)
            dominant_answers    -- most common answer per paraphrase
            answer_consistency  -- fraction of paraphrases sharing the same dominant answer
            all_samples         -- all sampled answers, grouped by paraphrase
    """
    if device is None:
        device = next(model.parameters()).device

    per_prompt_entropy: List[float] = []
    dominant_answers: List[str] = []
    all_samples: List[List[str]] = []

    for prompt in paraphrase_prompts:
        samples = sample_answers(
            model, tokenizer, prompt,
            n_samples=n_samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        h, info = semantic_entropy(samples, mode=mode, embed_threshold=embed_threshold)
        per_prompt_entropy.append(h)
        all_samples.append(samples)

        # Dominant answer: most common cluster representative
        if info:
            dom_key = max(info["cluster_sizes"], key=info["cluster_sizes"].__getitem__)
            dominant_answers.append(info["representatives"][dom_key])
        else:
            dominant_answers.append("")

    mean_entropy = sum(per_prompt_entropy) / max(len(per_prompt_entropy), 1)

    # Consistency: fraction of paraphrases whose dominant answer equals the most common dominant answer
    if dominant_answers:
        most_common_dom = Counter(a.lower().strip() for a in dominant_answers).most_common(1)[0][0]
        consistency = sum(
            1 for a in dominant_answers if a.lower().strip() == most_common_dom
        ) / len(dominant_answers)
    else:
        consistency = 1.0

    return {
        "per_prompt_entropy": per_prompt_entropy,
        "mean_entropy": mean_entropy,
        "dominant_answers": dominant_answers,
        "answer_consistency": consistency,
        "all_samples": all_samples,
    }


# ---------------------------------------------------------------------------
# Convenience: single-prompt conflict signal
# ---------------------------------------------------------------------------

def conflict_score(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 20,
    max_new_tokens: int = 20,
    temperature: float = 1.5,
    top_p: float = 0.9,
    mode: str = "exact",
    embed_threshold: float = 0.9,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Single-prompt wrapper: sample at high temperature, return semantic entropy
    as a scalar conflict signal.

    Returns:
        dict with keys: entropy, n_clusters, dominant_answer, samples.
    """
    if device is None:
        device = next(model.parameters()).device

    samples = sample_answers(
        model, tokenizer, prompt,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    h, info = semantic_entropy(samples, mode=mode, embed_threshold=embed_threshold)

    dominant = ""
    if info and info["cluster_sizes"]:
        dom_key = max(info["cluster_sizes"], key=info["cluster_sizes"].__getitem__)
        dominant = info["representatives"][dom_key]

    return {
        "entropy": h,
        "n_clusters": info.get("n_clusters", 0),
        "dominant_answer": dominant,
        "samples": samples,
    }
