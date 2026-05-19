"""
inference/maintenance.py — Pre/post-shock delta around detected events.

Inputs:
  - sessions:  list[list[TelemetryRecord]]
  - shocks:    list[MemoryEvent]   (from memory_reconstruction.detect_shocks)
  - outcomes:  optional list[OutcomeEvent]  (only then can we measure capability delta)

Always reports structural deltas (latency, response length, tool KL).
Reports outcome delta only when OutcomeEvents are linked.
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Optional

from ..schema import TelemetryRecord, MemoryEvent, OutcomeEvent, CoverageReport


def infer_maintenance(
    sessions: list[list[TelemetryRecord]],
    shocks: list[MemoryEvent],
    outcomes: Optional[list[OutcomeEvent]] = None,
    window: int = 3,
) -> dict:
    if not sessions:
        return _empty()

    per_shock = []
    for sh in shocks:
        s_idx = sh.session_idx
        pre = sessions[max(0, s_idx - window):s_idx]
        post = sessions[s_idx + 1:s_idx + 1 + window]
        if not pre or not post:
            continue

        delta = {
            "shock_session": s_idx,
            "type": sh.type,
            "confidence": sh.confidence,
            "latency_p50_delta_ms":  _safe_diff(_median_latency(post), _median_latency(pre)),
            "avg_response_tokens_delta": _safe_diff(
                _avg_output_tokens(post), _avg_output_tokens(pre)
            ),
        }

        # Outcome delta if outcomes are linked
        if outcomes:
            sess_ids_pre = {r.session_id for s in pre for r in s if r.session_id}
            sess_ids_post = {r.session_id for s in post for r in s if r.session_id}
            pre_rate = _outcome_rate(outcomes, sess_ids_pre)
            post_rate = _outcome_rate(outcomes, sess_ids_post)
            if pre_rate is not None and post_rate is not None:
                delta["outcome_rate_pre"] = round(pre_rate, 4)
                delta["outcome_rate_post"] = round(post_rate, 4)
                delta["outcome_rate_delta"] = round(post_rate - pre_rate, 4)

        per_shock.append(delta)

    median_outcome_delta = _median_or_none(
        [d["outcome_rate_delta"] for d in per_shock if "outcome_rate_delta" in d]
    )
    median_latency_delta = _median_or_none(
        [d["latency_p50_delta_ms"] for d in per_shock if d["latency_p50_delta_ms"] is not None]
    )

    # NEW (long-horizon trajectory): per-session intervention rate.
    # Rate of human-steering events (fail/user_rejected/abandoned outcomes)
    # per agent action. Rising trajectory = the agent needs more
    # handholding to stay on track over time — a direct proxy for
    # "human intervention becomes increasingly necessary."
    intervention_traj = _intervention_rate_trajectory(sessions, outcomes)
    intervention_slope = _ols(intervention_traj) if len(intervention_traj) >= 3 else None

    coverage = _maintenance_coverage(shocks, outcomes is not None and len(outcomes) > 0)

    # intervention_rate is bounded [0, 1]; rising = degradation.
    # Floor at 0.01 indicates the workload didn't produce any human-steering
    # signal (telemetry no_signal-equivalent) — distinct from "agent is healthy."
    from ._verdict import degradation_verdict
    intervention_verdict = degradation_verdict(
        intervention_traj, intervention_slope,
        rising_is_bad=True, floor_threshold=0.01,
        ceiling_threshold=0.5, slope_eps=0.005,
    )

    return {
        "shock_events":                   [_shock_to_dict(sh) for sh in shocks],
        "per_shock_deltas":               per_shock,
        "median_outcome_rate_delta":      median_outcome_delta,
        "median_latency_p50_delta_ms":    median_latency_delta,
        "n_shocks":                       len(shocks),
        "intervention_rate_trajectory":   [round(x, 4) for x in intervention_traj],
        "intervention_rate_slope":        (round(intervention_slope, 6) if intervention_slope is not None else None),
        "intervention_rate_verdict":      intervention_verdict,
        "coverage":                       coverage.as_dict(),
        "derived_from":                   "telemetry",
    }


def _intervention_rate_trajectory(
    sessions: list, outcomes: Optional[list]
) -> list:
    """Per session: ratio of human-steering events / agent actions.
    Steering events = outcomes with outcome ∈ {fail, user_rejected, abandoned}.
    """
    if not outcomes:
        return [0.0] * len(sessions)
    intervention_outcomes = {"fail", "user_rejected", "abandoned"}
    out = []
    for s in sessions:
        n_agent = sum(1 for r in s if r.role == "agent")
        if n_agent == 0:
            out.append(0.0)
            continue
        sids = {r.session_id for r in s if r.session_id}
        n_intv = sum(
            1 for o in outcomes
            if o.session_id in sids and o.outcome in intervention_outcomes
        )
        out.append(n_intv / n_agent)
    return out


def _ols(ys):
    if not ys or len(ys) < 2:
        return None
    n = len(ys)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None


def _empty() -> dict:
    return {
        "shock_events":               [],
        "per_shock_deltas":           [],
        "median_outcome_rate_delta":  None,
        "median_latency_p50_delta_ms": None,
        "n_shocks":                   0,
        "coverage":                   CoverageReport(0, 0.0, "no_test_fired").as_dict(),
        "derived_from":               "telemetry",
    }


def _shock_to_dict(sh: MemoryEvent) -> dict:
    return {
        "session_idx": sh.session_idx,
        "type":        sh.type,
        "confidence":  sh.confidence,
        "detail":      sh.detail,
    }


def _median_latency(window: list[list[TelemetryRecord]]) -> Optional[float]:
    lats = [r.duration_ms for s in window for r in s if r.duration_ms is not None]
    return statistics.median(lats) if lats else None


def _avg_output_tokens(window: list[list[TelemetryRecord]]) -> Optional[float]:
    toks = [r.output_tokens for s in window for r in s if r.output_tokens > 0]
    return sum(toks) / len(toks) if toks else None


def _outcome_rate(outcomes: list[OutcomeEvent], session_ids: set) -> Optional[float]:
    if not session_ids:
        return None
    relevant = [o for o in outcomes if o.session_id in session_ids]
    if not relevant:
        return None
    succ = sum(1 for o in relevant if o.outcome == "success")
    return succ / len(relevant)


def _safe_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 3)


def _median_or_none(xs: list[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _maintenance_coverage(shocks: list[MemoryEvent], have_outcomes: bool) -> CoverageReport:
    n = len(shocks)
    if n == 0:
        return CoverageReport(0, 0.0, "no_test_fired")
    # Coverage = how many shocks we have to compute deltas around.
    # Strong only if we ALSO have outcomes; otherwise structural-only.
    if have_outcomes and n >= 3:
        verdict = "strong"
    elif have_outcomes:
        verdict = "adequate"
    elif n >= 3:
        verdict = "weak"        # structural delta only — no capability signal
    else:
        verdict = "underpowered"
    return CoverageReport(n, None, verdict)
