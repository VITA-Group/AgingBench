"""
agingbench/metrics/g4_metrics.py — G4: Efficiency, Robustness, and Planning-Ahead.

Metrics:
  FASR(t)         First-Attempt Success Rate — fraction of tasks completed
                  without revision at cycle t.
  RR(t)           Revision Rate — average edit cycles per task at cycle t.
  CFR(t)          Cascading Failure Rate — fraction of previously passing
                  checks broken after agent change.
  LA(t)           Lookahead Accuracy — Jaccard similarity between predicted
                  and actual impact sets.
  shock Δm(ℓ)     Performance drop at life event ℓ.
  recovery R(ℓ)   Steps until performance recovers to α · m(t_ℓ−).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# FASR — First-Attempt Success Rate
# ---------------------------------------------------------------------------

def compute_fasr(task_results: list[dict]) -> float:
    """
    Fraction of tasks completed in exactly one attempt (no revisions).

    Parameters
    ----------
    task_results : list[dict]
        Each dict must have:
          - "task_id": str
          - "attempts": int  (number of edit cycles to completion; 1 = first attempt)

    Returns
    -------
    float  FASR in [0, 1].  1.0 = every task succeeded on first attempt.
    """
    if not task_results:
        return 0.0
    first_attempt = sum(1 for t in task_results if t.get("attempts", 1) == 1)
    return first_attempt / len(task_results)


# ---------------------------------------------------------------------------
# RR — Revision Rate
# ---------------------------------------------------------------------------

def compute_rr(task_results: list[dict]) -> float:
    """
    Average number of edit cycles per task.

    Parameters
    ----------
    task_results : list[dict]
        Each dict must have:
          - "task_id": str
          - "attempts": int  (total edit cycles including first attempt)

    Returns
    -------
    float  RR >= 1.0.  1.0 = no revisions needed.
    """
    if not task_results:
        return 1.0
    total_attempts = sum(t.get("attempts", 1) for t in task_results)
    return total_attempts / len(task_results)


# ---------------------------------------------------------------------------
# CFR — Cascading Failure Rate
# ---------------------------------------------------------------------------

def compute_cfr(
    passing_before: set[str],
    passing_after: set[str],
) -> float:
    """
    Fraction of previously passing checks that broke after the agent's change.

    Parameters
    ----------
    passing_before : set[str]
        Test/check IDs that were passing before the agent acted.
    passing_after : set[str]
        Test/check IDs that are passing after the agent acted.

    Returns
    -------
    float  CFR in [0, 1].  0 = no regressions.
    """
    if not passing_before:
        return 0.0
    broken = passing_before - passing_after
    return len(broken) / len(passing_before)


# ---------------------------------------------------------------------------
# LA — Lookahead Accuracy (Jaccard)
# ---------------------------------------------------------------------------

def compute_la(
    predicted_impact: set[str],
    actual_impact: set[str],
) -> float:
    """
    Jaccard similarity between the agent's predicted impact set and the
    ground-truth impact set.

    Parameters
    ----------
    predicted_impact : set[str]
        Files/modules the agent stated would be affected.
    actual_impact : set[str]
        Files/modules actually affected (from static analysis / test results).

    Returns
    -------
    float  LA in [0, 1].  1.0 = perfect prediction.
    """
    if not predicted_impact and not actual_impact:
        return 1.0
    intersection = predicted_impact & actual_impact
    union = predicted_impact | actual_impact
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Life event: shock Δm and recovery R
# ---------------------------------------------------------------------------

def compute_shock(
    m_before: float,
    m_after: float,
) -> float:
    """
    Performance drop at a life event.

    Δm = m(t_ℓ+) − m(t_ℓ−)

    A negative value indicates degradation.
    """
    return m_after - m_before


def compute_recovery(
    scores_after_event: list[float],
    m_before: float,
    alpha: float = 0.9,
) -> int | float:
    """
    Number of steps after the life event until m(t) recovers to α · m(t_ℓ−).

    Parameters
    ----------
    scores_after_event : list[float]
        Metric values at each step after the life event (starting from t_ℓ+).
        scores_after_event[0] is m(t_ℓ+), the post-event checkpoint.
    m_before : float
        m(t_ℓ−), the pre-event performance.
    alpha : float
        Recovery threshold fraction (default 0.9).

    Returns
    -------
    int   Number of steps to recovery. float('inf') if never recovered.
    """
    target = alpha * m_before
    for i, score in enumerate(scores_after_event):
        if score >= target:
            return i  # 0 = immediate recovery at first post-event checkpoint
    return float("inf")


# ---------------------------------------------------------------------------
# Convenience: compute all G4 metrics for one session
# ---------------------------------------------------------------------------

def score_session_g4(
    task_results: list[dict],
    passing_before: set[str] | None = None,
    passing_after: set[str] | None = None,
    predicted_impact: set[str] | None = None,
    actual_impact: set[str] | None = None,
) -> dict[str, float]:
    """
    Compute all applicable G4 per-session metrics (excluding life event
    metrics which are computed across sessions).
    """
    result: dict[str, float] = {}

    result["FASR"] = compute_fasr(task_results)
    result["RR"] = compute_rr(task_results)

    if passing_before is not None and passing_after is not None:
        result["CFR"] = compute_cfr(passing_before, passing_after)

    if predicted_impact is not None and actual_impact is not None:
        result["LA"] = compute_la(predicted_impact, actual_impact)

    return result


def score_life_event(
    curve_scores: list[float],
    event_index: int,
    alpha: float = 0.9,
) -> dict[str, float]:
    """
    Compute life event metrics from a full aging curve.

    Parameters
    ----------
    curve_scores : list[float]
        Full time series m(0), m(1), ..., m(T).
    event_index : int
        Index of the life event (t_ℓ). m(event_index) is the last
        pre-event score; m(event_index + 1) is the first post-event score.
    alpha : float
        Recovery threshold fraction.

    Returns
    -------
    dict with "shock_delta_m" and "recovery_R".
    """
    if event_index < 0 or event_index + 1 >= len(curve_scores):
        raise ValueError(
            f"event_index {event_index} out of range for curve of "
            f"length {len(curve_scores)}"
        )

    m_before = curve_scores[event_index]
    m_after = curve_scores[event_index + 1]
    scores_after = curve_scores[event_index + 1:]

    return {
        "shock_delta_m": compute_shock(m_before, m_after),
        "recovery_R": compute_recovery(scores_after, m_before, alpha),
    }
