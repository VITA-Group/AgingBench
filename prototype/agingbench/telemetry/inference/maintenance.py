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

    # Long-horizon intervention-rate trajectory (outcome-derived).
    # Rate of human-steering events (fail/user_rejected/abandoned outcomes)
    # per agent action. Rising = the agent needs more handholding to stay
    # on track. Only meaningful when outcomes are linked; otherwise all-zero.
    intervention_traj = _intervention_rate_trajectory(sessions, outcomes)
    intervention_slope = _ols(intervention_traj) if len(intervention_traj) >= 3 else None

    # Long-horizon shock-damage trajectory (structural, universal).
    # Per-session cumulative damage from detected lifecycle shocks.
    # Conceptually a temporal extension of the paper's shock_delta:
    # rising slope = the agent is becoming less resilient to operational
    # events as deployment ages. Independent of outcomes — fires on any
    # trace where shocks were detected. Used as the card-surface signal
    # because every adapter that produces model_id / cache fields can
    # emit it.
    shock_damage_traj = _shock_damage_trajectory(sessions, per_shock)
    shock_damage_slope = _ols(shock_damage_traj) if len(shock_damage_traj) >= 3 else None

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

    # shock_damage is unbounded above (cumulative sum) — custom verdict:
    #   n_shocks == 0          → no_test_fired (mechanism didn't activate)
    #   slope > eps            → rising_degradation (later shocks bite harder
    #                            OR shocks accumulate faster late in deployment)
    #   trajectory all-zero    → no_test_fired
    #   else                   → flat (shocks fired, no temporal trend)
    shock_damage_verdict = _shock_damage_verdict(
        shock_damage_traj, shock_damage_slope, len(shocks),
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
        "shock_damage_trajectory":        [round(x, 4) for x in shock_damage_traj],
        "shock_damage_slope":             (round(shock_damage_slope, 6) if shock_damage_slope is not None else None),
        "shock_damage_verdict":           shock_damage_verdict,
        "coverage":                       coverage.as_dict(),
        "derived_from":                   "telemetry",
    }


def _shock_damage_trajectory(sessions: list, per_shock_deltas: list) -> list:
    """Per-session cumulative shock damage.

    For each detected shock with a non-trivial pre/post window, the
    "damage" magnitude is computed via a three-tier preference:

      1. outcome_rate_delta (capability drop) — when outcomes are linked
      2. avg_response_tokens_delta / 100 (response shape disturbance) —
         universal across adapters that produce output_tokens
      3. latency_p50_delta_ms / 1000 (positive only; agent slowing down)

    The trajectory is the cumulative sum up to session t, so a step
    appears at each shock session and the cumulative line never
    decreases.

    Cumulative form is intentional: OLS over a per-shock-spike series
    (zeros between shocks) gives misleading "falling" slopes after a
    single spike. Cumulating preserves "happened" information and lets
    the slope read as "average damage rate per session."
    """
    n = len(sessions)
    if n == 0 or not per_shock_deltas:
        return [0.0] * n
    per_session_damage = [0.0] * n
    for sh in per_shock_deltas:
        s_idx = sh.get("shock_session")
        if s_idx is None or not (0 <= s_idx < n):
            continue
        damage = _shock_damage_magnitude(sh)
        per_session_damage[s_idx] += damage
    out = []
    running = 0.0
    for v in per_session_damage:
        running += v
        out.append(running)
    return out


def _shock_damage_magnitude(sh: dict) -> float:
    """Pick the best available damage proxy for one shock.

    Returns 0.0 if no proxy is populated (e.g., a trace whose adapter
    emits neither output_tokens nor duration_ms).
    """
    # Tier 1: capability delta (best signal, when outcomes are linked)
    ord_ = sh.get("outcome_rate_delta")
    if ord_ is not None:
        return max(-ord_, 0.0)              # negative delta = capability drop
    # Tier 2: response-shape disturbance (universal — Claude Code, OpenAI,
    # OpenHands all carry output_tokens). Magnitude in "100-token units."
    tdelta = sh.get("avg_response_tokens_delta")
    if tdelta is not None:
        return abs(tdelta) / 100.0
    # Tier 3: latency damage (only when the adapter populates duration_ms)
    lat = sh.get("latency_p50_delta_ms")
    if lat is not None:
        return max(lat, 0.0) / 1000.0
    return 0.0


def _shock_damage_verdict(traj: list, slope, n_shocks: int) -> str:
    """Verdict tailored for the cumulative shock-damage signal.

    Distinct from `degradation_verdict` because shock_damage:
      - is monotone non-decreasing (cumulative), so "falling" is impossible
      - has no natural ceiling (latency-damage in seconds is unbounded)
      - reads no_test_fired when n_shocks == 0 (mechanism inactive),
        which is different from "the agent passed maintenance probes"
    """
    if n_shocks == 0:
        return "no_test_fired"
    if not traj or all(v == 0.0 for v in traj):
        # Shocks detected but no damage proxy populated (adapter doesn't
        # carry duration_ms, output_tokens, or linked outcomes). Honest
        # signal: the test didn't fire even though the trigger did.
        return "no_test_fired"
    if slope is None or n_shocks < 3:
        # A single shock produces a cumulative-form step function whose
        # OLS slope is positive but meaningless ("rising" because of the
        # step, not because shocks bite harder over time). Require ≥3
        # shocks before reading a slope as a trend.
        return "flat"
    # Damage units are mixed across tiers (capability drop in [0,1],
    # response-token deviation in "100-token units," latency in seconds).
    # Tier 2 dominates Claude Code traces and produces per-shock damage
    # ~0.1–1.0 unit; rising cumulative slope above ~0.05/session means
    # late-deployment shocks compound noticeably faster than early ones.
    if slope > 0.05:
        return "rising_degradation"
    return "flat"


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
        "intervention_rate_trajectory": [],
        "intervention_rate_slope":    None,
        "intervention_rate_verdict":  "no_test_fired",
        "shock_damage_trajectory":    [],
        "shock_damage_slope":         None,
        "shock_damage_verdict":       "no_test_fired",
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
