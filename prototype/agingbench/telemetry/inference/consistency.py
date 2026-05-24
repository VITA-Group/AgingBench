"""
inference/consistency.py — Cross-session task-consistency probe.

Operationalises "is the agent getting worse at the same job over time?"
without requiring gold outcomes. Clusters user turns by sentence-
transformer similarity into repeat-task buckets, then measures how the
agent's behaviour (tool-call path + response embedding) drifts between
the first and Nth occurrence of each task.

Role in the card:
  1. Headline metric source (`behavior_drift_at_repeat`) when no
     OutcomeEvents are present — the outcome-free analogue of half_life.
  2. Aging-happened detector — the card opens with this signal.
  3. Audit block — mechanism diagnoses *explain* what consistency
     surfaces.
"""
from __future__ import annotations

from typing import Optional

from ..schema import TelemetryRecord, CoverageReport
from ._text_utils import cluster_by_similarity, ols_slope
from ._verdict import degradation_verdict


def infer_consistency(
    sessions: list[list[TelemetryRecord]],
    *,
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 2,
) -> dict:
    """Run the cross-session consistency probe.

    Returns a dict suitable for `card.trace_audit.consistency`. Keys:
      n_repeated_tasks_detected    : int    — clusters with ≥ min_cluster_size
      cluster_sizes                : list[int]
      behavior_drift_at_repeat     : float  — aggregate drift in [0, 1]
      tool_path_jaccard_drop_mean  : float
      response_cosine_drop_mean    : float
      outcome_consistency_rate     : float | None  (auxiliary, when outcomes linked)
      consistency_drop_trajectory  : list[(int, float)]  — per-session cumulative drift
      consistency_drop_slope       : float | None
      consistency_drop_verdict     : str
      coverage                     : dict (from CoverageReport)
      derived_from                 : "cross_session_task_consistency"
    """
    if not sessions:
        return _empty()

    # Collect user-role records by (session_idx, record_idx, prompt_text)
    user_items: list[tuple[int, int, str]] = []
    for sidx, s in enumerate(sessions):
        for ridx, r in enumerate(s):
            if r.role == "user" and r.prompt_preview:
                user_items.append((sidx, ridx, r.prompt_preview))

    clusters = cluster_by_similarity(user_items, threshold=similarity_threshold)
    sized_clusters = [c for c in clusters if len(c) >= min_cluster_size]

    if not sized_clusters:
        return _empty(
            coverage=CoverageReport(
                verdict="no_test_fired",
                n_observations=len(user_items),
            )
        )

    # Per-cluster drift: compare first occurrence to last occurrence.
    cluster_drifts: list[dict] = []
    for cluster in sized_clusters:
        cluster.sort(key=lambda it: it[0])  # by session_idx
        first_sidx, first_ridx, _first_text = cluster[0]
        last_sidx, last_ridx, _last_text = cluster[-1]

        first_record = sessions[first_sidx][first_ridx]
        last_record = sessions[last_sidx][last_ridx]

        # Get the agent's response right after each user turn (in same session)
        first_resp = _agent_response_after(sessions[first_sidx], first_ridx)
        last_resp = _agent_response_after(sessions[last_sidx], last_ridx)

        # Tool-path similarity: sequence of tool names called by agent
        first_tools = _tool_path(sessions[first_sidx], first_ridx)
        last_tools = _tool_path(sessions[last_sidx], last_ridx)
        tool_jaccard = _seq_jaccard(first_tools, last_tools)

        # Response cosine similarity
        resp_cos = _response_similarity(first_resp, last_resp)

        # Outcome consistency, when outcomes linked
        outcome_match: Optional[bool] = None
        if first_record.outcome is not None and last_record.outcome is not None:
            outcome_match = (
                first_record.outcome.is_success == last_record.outcome.is_success
            )

        cluster_drifts.append({
            "first_session": first_sidx,
            "last_session": last_sidx,
            "n_occurrences": len(cluster),
            "tool_path_jaccard": tool_jaccard,
            "response_cosine": resp_cos,
            "outcome_match": outcome_match,
        })

    # Aggregate drift = 1 - mean(similarity across tool + response components).
    similarities = []
    for cd in cluster_drifts:
        if cd["tool_path_jaccard"] is not None:
            similarities.append(cd["tool_path_jaccard"])
        if cd["response_cosine"] is not None:
            similarities.append(cd["response_cosine"])

    behavior_drift = (1.0 - sum(similarities) / len(similarities)) if similarities else 0.0

    tool_drops = [1.0 - cd["tool_path_jaccard"]
                  for cd in cluster_drifts if cd["tool_path_jaccard"] is not None]
    resp_drops = [1.0 - cd["response_cosine"]
                  for cd in cluster_drifts if cd["response_cosine"] is not None]
    tool_drop_mean = sum(tool_drops) / len(tool_drops) if tool_drops else 0.0
    resp_drop_mean = sum(resp_drops) / len(resp_drops) if resp_drops else 0.0

    outcome_matches = [cd["outcome_match"] for cd in cluster_drifts
                       if cd["outcome_match"] is not None]
    outcome_rate = (
        sum(outcome_matches) / len(outcome_matches) if outcome_matches else None
    )

    # Per-session trajectory: cumulative drift up to session t.
    # For each session t, drift = mean of (1 - sim) for clusters whose last
    # occurrence has been observed by t.
    drop_trajectory = _per_session_drop_trajectory(cluster_drifts, len(sessions))
    drop_slope = ols_slope([v for v in drop_trajectory if v is not None]) \
        if len([v for v in drop_trajectory if v is not None]) >= 3 else None
    drop_verdict = degradation_verdict(
        drop_trajectory, drop_slope,
        rising_is_bad=True, slope_eps=0.01,
    )

    # Coverage: strong if ≥3 clusters with ≥3 each; adequate if ≥2 with ≥2; else underpowered.
    n_strong_clusters = sum(1 for c in sized_clusters if len(c) >= 3)
    if n_strong_clusters >= 3:
        cov_verdict = "strong"
    elif len(sized_clusters) >= 2:
        cov_verdict = "adequate"
    else:
        cov_verdict = "underpowered"

    coverage = CoverageReport(
        verdict=cov_verdict,
        n_observations=sum(len(c) for c in sized_clusters),
    )

    return {
        "n_repeated_tasks_detected":   len(sized_clusters),
        "cluster_sizes":               sorted((len(c) for c in sized_clusters), reverse=True),
        "behavior_drift_at_repeat":    round(behavior_drift, 4),
        "tool_path_jaccard_drop_mean": round(tool_drop_mean, 4),
        "response_cosine_drop_mean":   round(resp_drop_mean, 4),
        "outcome_consistency_rate":    (round(outcome_rate, 4) if outcome_rate is not None else None),
        "consistency_drop_trajectory": [(i, round(v, 4) if v is not None else None)
                                        for i, v in enumerate(drop_trajectory)],
        "consistency_drop_slope":      (round(drop_slope, 6) if drop_slope is not None else None),
        "consistency_drop_verdict":    drop_verdict,
        "coverage":                    coverage.as_dict(),
        "derived_from":                "cross_session_task_consistency",
    }


def _empty(coverage: Optional[CoverageReport] = None) -> dict:
    if coverage is None:
        coverage = CoverageReport(
            verdict="no_test_fired",
            n_observations=0,
        )
    return {
        "n_repeated_tasks_detected":   0,
        "cluster_sizes":               [],
        "behavior_drift_at_repeat":    0.0,
        "tool_path_jaccard_drop_mean": 0.0,
        "response_cosine_drop_mean":   0.0,
        "outcome_consistency_rate":    None,
        "consistency_drop_trajectory": [],
        "consistency_drop_slope":      None,
        "consistency_drop_verdict":    "no_signal",
        "coverage":                    coverage.as_dict(),
        "derived_from":                "cross_session_task_consistency",
    }


def _is_human_user_boundary(r: TelemetryRecord) -> bool:
    """Distinguish a human user turn from a tool-result record.

    Some adapters (notably claude_code) encode tool results as role='user'
    records with no `prompt_preview`. Without this guard, the walk would
    stop at the first tool-result boundary and miss the agent's actual
    multi-turn response. Real human turns carry a non-empty prompt.
    """
    return r.role == "user" and bool(r.prompt_preview)


def _agent_response_after(session: list[TelemetryRecord], user_idx: int) -> str:
    """Concatenate agent response_previews up to the next *human* user turn."""
    parts = []
    for r in session[user_idx + 1:]:
        if _is_human_user_boundary(r):
            break
        if r.role == "agent" and r.response_preview:
            parts.append(r.response_preview)
    return " ".join(parts)


def _tool_path(session: list[TelemetryRecord], user_idx: int) -> list[str]:
    """Sequence of tool-call names up to the next *human* user turn."""
    names = []
    for r in session[user_idx + 1:]:
        if _is_human_user_boundary(r):
            break
        for tc in r.tool_calls or []:
            if tc.name:
                names.append(tc.name)
    return names


def _seq_jaccard(a: list[str], b: list[str]) -> Optional[float]:
    """Jaccard similarity of two tool-call sequences (as sets)."""
    if not a and not b:
        return None
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None
    return len(sa & sb) / len(sa | sb) if (sa or sb) else None


def _response_similarity(a: str, b: str) -> Optional[float]:
    """Cosine similarity of two response texts via sentence-transformers, with Jaccard fallback."""
    if not a or not b:
        return None
    try:
        from ...metrics.semantic_scorer import _get_model, cosine_similarity
    except ImportError:
        return None
    model = _get_model()
    if model is None:
        # Fallback to Jaccard of significant terms
        from ._text_utils import significant_terms, jaccard
        return jaccard(significant_terms(a), significant_terms(b))
    embs = model.encode([a[:512], b[:512]])
    return float(cosine_similarity(embs[0], embs[1]))


def _per_session_drop_trajectory(cluster_drifts: list[dict], n_sessions: int) -> list[Optional[float]]:
    """Per-session cumulative drift: mean of (1 - sim) over clusters whose last
    occurrence has been observed by session t."""
    out: list[Optional[float]] = []
    for t in range(n_sessions):
        observed = [cd for cd in cluster_drifts if cd["last_session"] <= t]
        sims = []
        for cd in observed:
            if cd["tool_path_jaccard"] is not None:
                sims.append(cd["tool_path_jaccard"])
            if cd["response_cosine"] is not None:
                sims.append(cd["response_cosine"])
        if sims:
            out.append(1.0 - sum(sims) / len(sims))
        else:
            out.append(None)
    return out
