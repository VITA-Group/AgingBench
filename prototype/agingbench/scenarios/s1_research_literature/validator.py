"""
agingbench/scenarios/s1_research_literature/validator.py — P2 constraint scorer.

Design: keyword presence check on the summary text.
  - No LLM judge involved → fully deterministic, zero additional API cost.
  - Each probe has a list of `keywords`; score = 1 if any keyword appears
    as a case-insensitive substring in the summary text, else 0.
  - Aggregate m(k) = mean of all probe scores = fraction of constraints retained.

This operationalizes "constraint still present" as "exact named value still
appears as a substring." Conservative: may penalize rephrased-but-correct values,
but gives a clean lower bound on information retention which is what P2 measures.

Unit test:
  validator.score_all(source_doc_text, probes) should return (all_ones, 1.0)
"""

from __future__ import annotations

import re


def _normalize_for_match(text: str) -> str:
    """Format-tolerant normalization so that ``73.9%`` matches ``73.9 percent``,
    ``40,922`` matches ``40922``, ``156ms`` matches ``156 milliseconds`` etc.

    Numeric equivalence is preserved exactly (no rounding or approximation).
    """
    t = text.lower()
    t = t.replace("%", " ").replace("$", " ")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = re.sub(r"(\d(?:\.\d+)?)\s*ms\b", r"\1 milliseconds", t)
    t = re.sub(r"(\d(?:\.\d+)?)\s*gb\b", r"\1 gigabytes", t)
    t = re.sub(r"(\d(?:\.\d+)?)\s*mb\b", r"\1 megabytes", t)
    t = re.sub(r"(\d(?:\.\d+)?)\s*kb\b", r"\1 kilobytes", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def score_probe(summary_text: str, probe: dict) -> int:
    """
    Returns 1 if any keyword from probe["keywords"] is found in summary_text
    after format normalization (% ↔ percent, $ ↔ dollars, commas in numbers,
    common unit suffixes), else 0.

    If the probe carries ``forbidden_keywords`` (e.g. trend dep probes whose
    pre-revision value the agent must NOT cite), the score is forced to 0 when
    any forbidden keyword appears, regardless of whether expected keywords are
    present.
    """
    norm_text = _normalize_for_match(summary_text)
    forbidden = probe.get("forbidden_keywords") or []
    if forbidden and any(_normalize_for_match(fk) in norm_text for fk in forbidden):
        return 0
    return int(any(_normalize_for_match(kw) in norm_text for kw in probe["keywords"]))


def score_all(summary_text: str, probes: list[dict]) -> tuple[list[int], float]:
    """
    Score all probes against summary_text.

    Returns
    -------
    probe_scores : list[int]  — 0 or 1 per probe
    m            : float      — fraction correct in [0, 1]
    """
    scores = [score_probe(summary_text, p) for p in probes]
    m = sum(scores) / len(scores) if scores else 0.0
    return scores, m


def report(summary_text: str, probes: list[dict]) -> None:
    """Pretty-print which constraints are retained vs. lost."""
    scores, m = score_all(summary_text, probes)
    print(f"\nConstraint retention: {sum(scores)}/{len(scores)}  (m = {m:.3f})")
    for probe, s in zip(probes, scores):
        status = "OK" if s else "LOST"
        print(f"  [{status}] {probe['probe_id']}  {probe['canonical_answer']}")
