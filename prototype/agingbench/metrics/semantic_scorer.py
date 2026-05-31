"""
agingbench/metrics/semantic_scorer.py — Semantic similarity scoring.

Provides cosine-similarity-based scoring alongside keyword matching.
Handles paraphrasing ("$340K" vs "$340,000" vs "three hundred forty thousand")
and provides partial credit instead of binary 0/1.

Uses sentence-transformers if available, falls back to keyword-only scoring.
"""

from __future__ import annotations

from typing import Optional
import re

_model = None
_model_load_attempted = False


def _present(needle: str, haystack_lower: str) -> bool:
    """Digit-flank-safe substring presence: a numeric keyword (e.g. ``73``)
    will not match inside a longer number (``731``/``$1,732``). Word keywords
    keep plain substring semantics. ``haystack_lower`` must be lowercased."""
    if not needle:
        return False
    return re.search(r"(?<!\d)" + re.escape(needle.lower()) + r"(?!\d)",
                     haystack_lower) is not None


def _get_model():
    """Lazy-load sentence-transformers model."""
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        _model = None
    return _model


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def semantic_score(text: str, reference: str, threshold: float = 0.65) -> float:
    """
    Score text against reference using semantic similarity.
    Returns a float in [0, 1].

    If sentence-transformers is available: cosine similarity.
    Otherwise: keyword overlap ratio.
    """
    model = _get_model()
    if model is not None:
        embs = model.encode([text[:512], reference[:512]])
        sim = cosine_similarity(embs[0], embs[1])
        return max(0.0, sim)

    # Fallback: word overlap
    words_text = set(re.findall(r'\w+', text.lower()))
    words_ref = set(re.findall(r'\w+', reference.lower()))
    if not words_ref:
        return 0.0
    overlap = len(words_text & words_ref) / len(words_ref)
    return overlap


def score_probe_hybrid(
    text: str,
    probe: dict,
    semantic_threshold: float = 0.65,
    keyword_weight: float = 0.5,
) -> float:
    """
    Hybrid scoring: combines keyword match and semantic similarity.

    Returns a float in [0, 1]:
    - 1.0 if keyword match succeeds
    - semantic_score if keyword match fails but semantic similarity is high
    - 0.0 if both fail

    This handles paraphrasing ("$340K" vs "$340,000") that keyword matching misses.
    """
    text_lower = text.lower()

    # Keyword match (original binary approach)
    keywords = probe.get("keywords", [])
    keyword_hit = any(kw.lower() in text_lower for kw in keywords)
    if keyword_hit:
        return 1.0

    # Semantic similarity (handles paraphrasing)
    reference = probe.get("canonical_answer", "") or probe.get("question", "")
    if reference:
        sim = semantic_score(text, reference, semantic_threshold)
        if sim >= semantic_threshold:
            return sim  # partial credit

    return 0.0


def score_fact_survival(
    memory_text: str,
    fact: str,
    keywords: list[str],
    semantic_threshold: float = 0.60,
) -> float:
    """
    Score whether a fact survives in memory, using both keyword and semantic matching.

    Returns float in [0, 1]:
    - 1.0 if any keyword found
    - semantic similarity score if above threshold
    - 0.0 otherwise
    """
    text_lower = memory_text.lower()

    # Keyword match (digit-flank-safe so "73" does not match inside "731").
    if any(_present(kw, text_lower) for kw in keywords):
        return 1.0

    # Value-bearing gate: if the fact carries numeric gold keywords (dollar
    # amounts, counts, dates) and NONE of those exact values survived above,
    # the specific value is gone. Sentence-level cosine similarity must NOT
    # re-credit a stale/wrong number — "Contingency reserve set at $X" stays
    # ~0.9 similar to the gold sentence regardless of X, which would mask the
    # compression/revision the metric exists to detect. The semantic fallback
    # is reserved for non-numeric (paraphrasable) facts only.
    numeric_gold = [kw for kw in keywords if any(ch.isdigit() for ch in kw)]
    if numeric_gold:
        return 0.0

    # Semantic: check each sentence in memory for similarity to the fact
    model = _get_model()
    if model is not None:
        # Split memory into sentences and check each
        sentences = [s.strip() for s in memory_text.split('.') if len(s.strip()) > 10]
        if sentences:
            fact_emb = model.encode([fact])
            # Check in batches to avoid encoding all sentences at once
            best_sim = 0.0
            batch_size = 32
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                sent_embs = model.encode(batch)
                for emb in sent_embs:
                    sim = cosine_similarity(fact_emb[0], emb)
                    best_sim = max(best_sim, sim)
            if best_sim >= semantic_threshold:
                return best_sim

    return 0.0


def compute_fidelity_semantic(
    memory_text: str,
    gold_facts: list[dict],
    semantic_threshold: float = 0.60,
) -> tuple[float, dict]:
    """
    Compute summarization fidelity using hybrid keyword+semantic scoring.

    Returns (overall_fidelity, per_fact_scores).
    """
    per_fact = {}
    for fact_entry in gold_facts:
        fact_id = fact_entry.get("id", fact_entry.get("fact", ""))
        fact_text = fact_entry.get("fact", "")
        keywords = fact_entry.get("keywords", [])

        score = score_fact_survival(
            memory_text, fact_text, keywords, semantic_threshold
        )
        per_fact[fact_id] = score

    overall = sum(per_fact.values()) / len(per_fact) if per_fact else 1.0
    return overall, per_fact
