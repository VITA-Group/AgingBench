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


def score_probe(summary_text: str, probe: dict) -> int:
    """
    Returns 1 if any keyword from probe["keywords"] is found in summary_text
    (case-insensitive substring match), else 0.
    """
    lower = summary_text.lower()
    return int(any(kw.lower() in lower for kw in probe["keywords"]))


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
