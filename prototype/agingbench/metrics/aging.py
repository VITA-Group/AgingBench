"""
agingbench/metrics/aging.py — Time-series metrics for aging curves (PDF §5.3).

Reported statistics per curve:
  m(0)          initial performance
  half_life     first t where m(t) <= 0.5 * m(0)  (linear interpolation)
  decay_slope   OLS slope of m on t  (negative = degradation)
  m_final       performance at last checkpoint
  hazard_proxy  probability of failure per unit exposure (survival analysis)
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class AgingCurve:
    """Time series of (exposure_t, metric_t) pairs for one SUT × scenario.

    `metric_name` identifies which metric this trajectory carries (e.g.
    "keyword_m", "constraint_precision", "version_accuracy"). Used by the
    plotting layer to pick appropriate axis labels and legend entries.
    Optional for backwards compatibility.
    """
    exposures: list[int]    # t-axis: cycle / session index
    scores: list[float]     # m(t) values in [0, 1]
    scenario: str
    sut_id: str
    metric_name: str = ""


def compute_half_life(curve: AgingCurve, threshold: float = 0.5) -> float:
    """
    First t where m(t) <= threshold * m(0), found by linear interpolation
    between checkpoint pairs. Returns float('inf') if never crossed.
    """
    if len(curve.scores) < 2:
        return float("inf")
    m0 = curve.scores[0]
    target = threshold * m0
    for i in range(1, len(curve.scores)):
        if curve.scores[i] <= target:
            # linear interpolation between t[i-1] and t[i]
            t0, t1 = curve.exposures[i - 1], curve.exposures[i]
            m_prev, m_cur = curve.scores[i - 1], curve.scores[i]
            if m_cur == m_prev:
                return float(t1)
            frac = (target - m_prev) / (m_cur - m_prev)
            return t0 + frac * (t1 - t0)
    return float("inf")


def compute_decay_slope(curve: AgingCurve) -> float:
    """OLS slope of m on t. Negative value = degradation."""
    if len(curve.scores) < 2:
        return 0.0
    xs = curve.exposures
    ys = curve.scores
    if _HAS_SCIPY:
        slope, *_ = _stats.linregress(xs, ys)
        return float(slope)
    # manual OLS fallback
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    den = sum((x - x_bar) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def compute_hazard_proxy(curve: AgingCurve, failure_threshold: float = 0.5) -> float:
    """
    Empirical hazard proxy: fraction of intervals where m(t) crosses below
    failure_threshold * m(0), divided by total intervals.

    Approximates the discrete hazard rate — probability of transitioning from
    'acceptable' to 'failed' per unit exposure. Returns 0.0 if the curve
    never crosses the threshold.
    """
    if len(curve.scores) < 2:
        return 0.0
    m0 = curve.scores[0]
    if m0 == 0:
        return 0.0
    target = failure_threshold * m0
    failure_events = 0
    at_risk_intervals = 0
    for i in range(1, len(curve.scores)):
        # only count intervals where we were above threshold at start
        if curve.scores[i - 1] > target:
            at_risk_intervals += 1
            if curve.scores[i] <= target:
                failure_events += 1
    if at_risk_intervals == 0:
        return 0.0
    return failure_events / at_risk_intervals


def count_response_tokens(llm, text: str) -> int:
    """
    Best-effort token count of an agent response, for truncation diagnostics.

    Returns the number of tokens, or -1 if the LLM backend doesn't support
    token counting. Intended for use in runners after each LLM call to
    populate session_results["response_tokens"] — lets post-hoc analysis
    detect runs where responses hit max_new_tokens and could confound
    the aging signal.
    """
    if not text:
        return 0
    try:
        return int(llm.count_tokens(text))
    except Exception:
        return -1


def flag_response_near_cap(response_tokens: list[int], max_tokens: int,
                           threshold: float = 0.98) -> dict:
    """
    Summarize how many responses in a run were at/near the generation cap.

    Args:
        response_tokens: per-response token counts from session_results
        max_tokens: the max_new_tokens setting used by the LLM
        threshold: fraction of max_tokens that counts as "near cap" (default 98%)

    Returns dict with:
        n_total: number of responses measured
        n_at_cap: number at or above threshold * max_tokens
        pct_at_cap: fraction of responses near the cap
        max_observed: largest response token count seen
        cap_confound_risk: "none" / "low" / "medium" / "high"
    """
    valid = [t for t in response_tokens if isinstance(t, int) and t >= 0]
    if not valid:
        return {"n_total": 0, "cap_confound_risk": "unknown"}
    n_at_cap = sum(1 for t in valid if t >= threshold * max_tokens)
    pct = n_at_cap / len(valid)
    if pct == 0:
        risk = "none"
    elif pct < 0.05:
        risk = "low"
    elif pct < 0.20:
        risk = "medium"
    else:
        risk = "high"
    return {
        "n_total": len(valid),
        "n_at_cap": n_at_cap,
        "pct_at_cap": round(pct, 4),
        "max_observed": max(valid),
        "max_tokens_setting": max_tokens,
        "cap_confound_risk": risk,
    }


def summarize(curve: AgingCurve) -> dict:
    """Return a flat dict of all reported statistics for one curve."""
    slope = compute_decay_slope(curve)
    m0 = round(curve.scores[0], 4) if curve.scores else None
    m_final = round(curve.scores[-1], 4) if curve.scores else None

    # Flag whether the curve shows a detectable aging signal.
    # A non-negative slope or m_final >= m0 means the primary metric
    # did not decline over the run — possibly because the model is
    # resilient, the session count is too low, or the metric does not
    # capture the relevant aging mechanism for this model×scenario.
    aging_detected = (slope < -0.005) if len(curve.scores) >= 3 else None

    return {
        "scenario": curve.scenario,
        "sut_id": curve.sut_id,
        "m0": m0,
        "m_final": m_final,
        "half_life": round(compute_half_life(curve), 2),
        "decay_slope": round(slope, 5),
        "hazard_proxy": round(compute_hazard_proxy(curve), 4),
        "n_checkpoints": len(curve.scores),
        "checkpoints": list(zip(curve.exposures, [round(s, 4) for s in curve.scores])),
        "aging_detected": aging_detected,
    }


def aggregate_curves(curves: list[AgingCurve]) -> dict:
    """
    Aggregate multiple seed runs into mean ± std per checkpoint.

    Returns dict with:
      exposures: list of checkpoint indices
      mean: list of mean scores
      std: list of std scores
      ci_lower: list of mean - 1.96*std (95% CI)
      ci_upper: list of mean + 1.96*std (95% CI)
      per_seed: list of per-seed score lists
      summary: dict with m0, m_final, slope mean ± std
    """
    if not curves:
        return {}

    # Align by exposure index
    max_len = max(len(c.exposures) for c in curves)
    exposures = curves[0].exposures[:max_len]

    # Collect scores per checkpoint across seeds
    per_checkpoint: list[list[float]] = []
    for i in range(max_len):
        vals = []
        for c in curves:
            if i < len(c.scores):
                vals.append(c.scores[i])
        per_checkpoint.append(vals)

    means = [sum(v) / len(v) for v in per_checkpoint]
    stds = []
    for vals in per_checkpoint:
        if len(vals) < 2:
            stds.append(0.0)
        else:
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            stds.append(variance ** 0.5)

    ci_lower = [max(0, m - 1.96 * s) for m, s in zip(means, stds)]
    ci_upper = [min(1, m + 1.96 * s) for m, s in zip(means, stds)]

    # Aggregate summary statistics across seeds
    summaries = [summarize(c) for c in curves]
    m0s = [s["m0"] for s in summaries if s["m0"] is not None]
    m_finals = [s["m_final"] for s in summaries if s["m_final"] is not None]
    slopes = [s["decay_slope"] for s in summaries]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    return {
        "exposures": exposures,
        "mean": means,
        "std": stds,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_seeds": len(curves),
        "per_seed": [c.scores[:max_len] for c in curves],
        "summary": {
            "m0_mean": round(_mean(m0s), 4),
            "m0_std": round(_std(m0s), 4),
            "m_final_mean": round(_mean(m_finals), 4),
            "m_final_std": round(_std(m_finals), 4),
            "slope_mean": round(_mean(slopes), 5),
            "slope_std": round(_std(slopes), 5),
        },
    }


def load_curve_from_trace(trace_path: str, sut_id: str = "") -> AgingCurve:
    """
    Reconstruct an AgingCurve by replaying probe_batch events from a trace file.
    Useful for re-computing metrics without re-running the experiment.
    """
    import json
    from pathlib import Path
    exposures, scores = [], []
    with open(trace_path) as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("event") == "probe_batch":
                exposures.append(ev["cycle"])
                scores.append(ev["m"])
    scenario = "unknown"
    with open(trace_path) as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("event") == "run_start":
                scenario = ev.get("scenario", "unknown")
                sut_id = sut_id or ev.get("sut_id", "unknown")
                break
    return AgingCurve(exposures=exposures, scores=scores, scenario=scenario, sut_id=sut_id)
