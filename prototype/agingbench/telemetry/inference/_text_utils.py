"""
inference/_text_utils.py — Lightweight, NLP-dep-free text helpers shared
across mechanism inference modules.

All helpers are intentionally simple (regex + sets) so the package
doesn't grow a spaCy/transformers/nltk dependency just for telemetry.
For higher-quality NLP, downstream consumers can override these.
"""
from __future__ import annotations

import re
from typing import Iterable


# Conservative English stopword set. Kept short so it stays useful
# without becoming a maintenance burden. Domain-specific terms (code
# identifiers, etc.) can be filtered by callers via additional filters.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
    "at", "by", "for", "with", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "must", "can",
    "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "them", "his", "her", "its", "our", "their", "my",
    "me", "us", "him", "your", "yours", "what", "when", "where", "why",
    "how", "which", "who", "whom", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "now", "then",
    "there", "here", "into", "through", "about", "between", "out", "up",
    "down", "off", "over", "under", "again", "further", "once", "ok",
    "okay", "yes", "please", "thanks", "thank",
})

_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}(?:\s+[A-Z][a-zA-Z0-9]+)*\b")

# Specific-value patterns for P3 (argument specificity)
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?(?:Z|[+-]\d{2}:?\d{2})?$")
_VERSION_RE = re.compile(r"^v?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?$")
_PATH_RE = re.compile(r"^(?:/|\.{1,2}/|[a-zA-Z]:[\\/])[\w./\\-]+$")
_GENERIC_TERMS = frozenset({
    "true", "false", "null", "none", "nil", "yes", "no",
    "data", "info", "test", "value", "name", "id", "thing",
    "item", "x", "y", "z", "foo", "bar", "baz", "tbd", "todo",
    "recent", "current", "latest", "all", "any", "some",
})


def is_specific_value(v) -> bool:
    """Heuristic: does this tool-call arg value look 'specific' (UUID, ISO
    timestamp, version, path, large int ID) vs 'generic' (null, common word,
    short noun)?

    Used by P3 (`_tool_argument_specificity_trajectory`) to detect
    compression eating fact-specificity in agent outputs over sessions.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        # Treat large numeric IDs as specific; small ints / counts are generic.
        return abs(v) >= 100
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False
    lo = s.lower()
    if lo in _GENERIC_TERMS:
        return False
    if _UUID_RE.match(s):
        return True
    if _ISO_TS_RE.match(s):
        return True
    if _VERSION_RE.match(s):
        return True
    if _PATH_RE.match(s):
        return True
    # Numeric strings: same threshold as numeric types
    if s.lstrip("-").isdigit():
        return abs(int(s)) >= 100
    # Long alphanumeric strings (likely IDs, hashes, names)
    if len(s) >= 8 and any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
        return True
    # Default: treat short or word-like as generic
    return False


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens of length >= 3, alphanumeric only."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def significant_terms(text: str, extra_filter: set[str] | None = None) -> set[str]:
    """Return the set of non-stopword tokens of length >= 3.
    Used as a coarse 'topic vocabulary' for similarity comparisons.
    """
    extra = extra_filter or set()
    return {t for t in tokenize(text)
            if t not in _STOPWORDS and t not in extra}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity of two iterables. Returns 0.0 if both empty."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def extract_capitalised_entities(text: str) -> set[str]:
    """Return the set of capitalised multi-word strings — a proxy for
    proper-noun entities. Lower-cased for downstream comparison.
    """
    if not text:
        return set()
    return {m.group(0).lower() for m in _ENTITY_RE.finditer(text)}


def ols_slope(ys: list[float]) -> float | None:
    """Ordinary-least-squares slope of ys vs index. None if too few points."""
    if not ys or len(ys) < 2:
        return None
    n = len(ys)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None


def cluster_by_similarity(
    items: list[tuple[int, int, str]],
    threshold: float = 0.75,
) -> list[list[tuple[int, int, str]]]:
    """Greedy single-pass clustering by sentence-transformer cosine similarity.

    Each item is (session_idx, record_idx, text). Returns a list of clusters,
    where each cluster is a list of items whose texts are pairwise similar
    (cosine sim ≥ threshold to the cluster's first member, the centroid).

    Falls back to Jaccard if sentence-transformers is unavailable. Items
    with empty/whitespace text are skipped.
    """
    nonempty = [(s, r, t.strip()) for s, r, t in items if t and t.strip()]
    if not nonempty:
        return []

    # Try the encoder path first.
    try:
        from ...metrics.semantic_scorer import _get_model, cosine_similarity
        model = _get_model()
    except ImportError:
        model = None

    if model is not None:
        import numpy as np
        embs = model.encode([t[:512] for _, _, t in nonempty])
        return _greedy_cluster(nonempty, embs, threshold, cosine_similarity)

    # Fallback: Jaccard on significant terms.
    return _greedy_cluster_jaccard(nonempty, threshold)


def _greedy_cluster(items, embs, threshold, sim_fn):
    clusters = []
    cluster_centroids = []
    for item, emb in zip(items, embs):
        assigned = False
        for ci, centroid in enumerate(cluster_centroids):
            if sim_fn(emb, centroid) >= threshold:
                clusters[ci].append(item)
                assigned = True
                break
        if not assigned:
            clusters.append([item])
            cluster_centroids.append(emb)
    return clusters


def _greedy_cluster_jaccard(items, threshold):
    clusters = []
    cluster_centroids = []
    for s, r, t in items:
        terms = significant_terms(t)
        assigned = False
        for ci, centroid in enumerate(cluster_centroids):
            if jaccard(terms, centroid) >= threshold:
                clusters[ci].append((s, r, t))
                assigned = True
                break
        if not assigned:
            clusters.append([(s, r, t)])
            cluster_centroids.append(terms)
    return clusters
