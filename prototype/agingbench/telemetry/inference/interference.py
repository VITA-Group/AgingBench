"""
inference/interference.py — Tool-distribution drift over sessions.

Primary signal: KL divergence of per-session tool-call distribution
relative to a baseline (early-session) distribution. Drift > 0 means
the agent is using tools differently than at start = behavioral drift
proxy for interference.

This is NOT scenario interference_resistance (which is the score on
constructed confusable pairs). The block flags it as
`derived_from: 'tool_distribution_drift'`.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Optional

from ..schema import TelemetryRecord, CoverageReport


def infer_interference(
    sessions: list[list[TelemetryRecord]],
    baseline_window_n: int = 3,
    epsilon: float = 0.001,
) -> dict:
    if len(sessions) < baseline_window_n + 1:
        return _empty(len(sessions))

    # Per-session tool-name distribution
    dists: list[dict[str, float]] = []
    all_tools: set[str] = set()
    for s in sessions:
        names = [tc.name for r in s for tc in r.tool_calls]
        if not names:
            dists.append({})
            continue
        c = Counter(names)
        total = sum(c.values())
        dists.append({k: v / total for k, v in c.items()})
        all_tools.update(c.keys())

    n_with_tools = sum(1 for d in dists if d)
    if n_with_tools == 0 or not all_tools:
        return _empty(len(sessions))

    # Smooth + project all distributions onto the union of tools
    smoothed = []
    for d in dists:
        denom = sum(d.values()) + epsilon * len(all_tools)
        smoothed.append({t: (d.get(t, 0) + epsilon) / denom for t in all_tools})

    baseline = _avg_distributions(smoothed[:baseline_window_n])
    kl_traj = [
        round(_kl(d, baseline), 4)
        for d in smoothed
    ]

    post_baseline = kl_traj[baseline_window_n:]
    kl_mean = round(statistics.mean(post_baseline), 4) if post_baseline else None
    kl_slope = _ols_slope(post_baseline) if len(post_baseline) >= 3 else None

    # NEW (long-horizon trajectory): goal-anchor drift.
    # Extract the topic vocabulary of session 0 (the original task framing),
    # then for each later session compute Jaccard overlap of agent
    # responses' topic vocabulary vs the anchor. Declining = the agent's
    # output has drifted from the original goal — a general, task-agnostic
    # proxy for "goal drift" / "off the main track."
    goal_drift_traj = _goal_anchor_drift_trajectory(sessions)
    goal_drift_slope = _ols_skipping_none(goal_drift_traj) if len(goal_drift_traj) >= 3 else None

    coverage = _interference_coverage(sessions, n_with_tools)

    # goal_anchor_drift is bounded [0, 1]; falling = degradation.
    # Floor at 0.02 means the agent has effectively lost touch with the
    # session-0 vocabulary regardless of any tiny remaining slope.
    from ._verdict import degradation_verdict
    goal_anchor_verdict = degradation_verdict(
        goal_drift_traj, goal_drift_slope,
        rising_is_bad=False, floor_threshold=0.02,
        ceiling_threshold=None, slope_eps=0.005,
    )

    return {
        "tool_kl_trajectory":             kl_traj,
        "tool_kl_mean_post_baseline":     kl_mean,
        "tool_kl_slope":                  (round(kl_slope, 6) if kl_slope is not None else None),
        "baseline_window_size":           baseline_window_n,
        "n_distinct_tools":               len(all_tools),
        "goal_anchor_drift_trajectory":   [round(x, 4) if x is not None else None for x in goal_drift_traj],
        "goal_anchor_drift_slope":        (round(goal_drift_slope, 4) if goal_drift_slope is not None else None),
        "goal_anchor_drift_verdict":      goal_anchor_verdict,
        "coverage":                       coverage.as_dict(),
        "derived_from":                   "tool_distribution_drift",
    }


def _goal_anchor_drift_trajectory(sessions: list) -> list:
    """Per session: Jaccard similarity of agent output vocabulary vs the
    session-0 goal-anchor vocabulary. Declining trajectory = the agent
    has drifted from the original task framing.

    Returns None per session that has no agent output to compare.
    """
    from ._text_utils import jaccard, significant_terms

    if not sessions:
        return []

    first_user = " ".join(
        r.prompt_preview or "" for r in sessions[0] if r.role == "user"
    )
    anchor = significant_terms(first_user)
    if not anchor:
        # No anchor to compare against; can't compute drift
        return [None] * len(sessions)

    out = []
    for s in sessions:
        agent_text = " ".join(r.response_preview or "" for r in s if r.role == "agent")
        if not agent_text:
            out.append(None)
            continue
        agent_vocab = significant_terms(agent_text)
        out.append(jaccard(anchor, agent_vocab))
    return out


def _ols_skipping_none(ys):
    pairs = [(i, y) for i, y in enumerate(ys) if y is not None]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    den = sum((p[0] - mx) ** 2 for p in pairs)
    return num / den if den else None


def _empty(n_sess: int) -> dict:
    return {
        "tool_kl_trajectory":          [],
        "tool_kl_mean_post_baseline":  None,
        "tool_kl_slope":               None,
        "baseline_window_size":        0,
        "n_distinct_tools":            0,
        "coverage":                    CoverageReport(n_sess, 0.0, "no_test_fired").as_dict(),
        "derived_from":                "tool_distribution_drift",
    }


def _avg_distributions(dists: list[dict[str, float]]) -> dict[str, float]:
    if not dists:
        return {}
    keys = set().union(*[d.keys() for d in dists])
    return {k: sum(d.get(k, 0) for d in dists) / len(dists) for k in keys}


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    s = 0.0
    for k, pv in p.items():
        qv = q.get(k, 1e-12)
        if pv > 0 and qv > 0:
            s += pv * math.log(pv / qv)
    return s


def _ols_slope(ys: list[float]) -> Optional[float]:
    if len(ys) < 2:
        return None
    n = len(ys)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None


def _interference_coverage(sessions, n_with_tools: int) -> CoverageReport:
    n_sess = len(sessions)
    if n_sess == 0:
        return CoverageReport(0, 0.0, "no_test_fired")
    cov = n_with_tools / n_sess
    if cov > 0.5:
        verdict = "strong"
    elif cov > 0.2:
        verdict = "adequate"
    elif cov > 0:
        verdict = "weak"
    else:
        verdict = "no_test_fired"
    return CoverageReport(n_with_tools, round(cov, 3), verdict)
