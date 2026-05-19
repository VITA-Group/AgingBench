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
