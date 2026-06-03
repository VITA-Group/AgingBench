"""
S4 Software Engineering Agent — Validator

Scores agent modifications against ground-truth impact sets and test suites.
Produces G4 metrics: CFR, LA.
"""

from __future__ import annotations


def compute_la(predicted_impact: set[str], actual_impact: set[str]) -> float:
    """G4-M4: Lookahead Accuracy — Jaccard similarity."""
    if not predicted_impact and not actual_impact:
        return 1.0
    intersection = predicted_impact & actual_impact
    union = predicted_impact | actual_impact
    return len(intersection) / len(union) if union else 0.0


def compute_cfr(passing_before: set[str], passing_after: set[str]) -> float:
    """G4-M3: Cascading Failure Rate — fraction of previously passing tests now failing."""
    if not passing_before:
        return 0.0
    broken = passing_before - passing_after
    return len(broken) / len(passing_before)
