"""
inference/compression.py — Three observable signals for compression aging.

  1. Context saturation rate     (memory budget pressure over sessions)
  2. Self-contradiction rate     (NER-cross-reference; same entity, different value)
  3. Fact density slope          (specificity decay across sessions)

None of these is the same number as scenario keyword_m(t). The block
reports them separately + a coverage report so consumers know whether
the workload actually stressed compression.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict
from typing import Optional

from ..schema import TelemetryRecord, CoverageReport


# Lightweight "fact" pattern: numbers, dates, entity-ish capitalised tokens.
_NUM_RE   = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_DATE_RE  = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b")
_ENT_RE   = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*\b")
# (entity, attribute) ↔ value claim: "X's Y is Z" / "X has Y of Z"
_CLAIM_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)(?:'s|\s+has)?\s+([a-z]+)\s+(?:is|=|:|of)\s+([\w$.,-]+)",
    re.IGNORECASE,
)


def infer_compression(
    sessions: list[list[TelemetryRecord]],
    ctx_window: int = 200_000,
    saturation_threshold: float = 0.85,
) -> dict:
    if not sessions:
        return _empty()

    # 1. Saturation trajectory
    per_sess_sat: list[float] = []
    n_with_ctx = 0
    for s in sessions:
        loads = [r.context_window_size or r.input_tokens for r in s
                 if (r.context_window_size or r.input_tokens or 0) > 0]
        if loads:
            n_with_ctx += 1
            per_sess_sat.append(sum(loads) / len(loads) / ctx_window)

    sat_session_rate = (
        sum(1 for x in per_sess_sat if x > saturation_threshold) / len(per_sess_sat)
        if per_sess_sat else 0.0
    )
    sat_slope = _ols_slope(per_sess_sat) if len(per_sess_sat) >= 3 else None

    # 2. Self-contradiction across sessions
    claims = _extract_claims_per_session(sessions)
    contradiction_rate = _self_contradiction_rate(claims)

    # 3. Fact density slope
    per_sess_density: list[float] = []
    for s in sessions:
        agent_text = " ".join(r.response_preview or "" for r in s if r.role == "agent")
        n_tokens = sum(r.output_tokens for r in s if r.role == "agent")
        if n_tokens > 0:
            n_facts = (len(_NUM_RE.findall(agent_text))
                       + len(_DATE_RE.findall(agent_text))
                       + len(_ENT_RE.findall(agent_text)))
            per_sess_density.append(n_facts / n_tokens)
    density_slope = _ols_slope(per_sess_density) if len(per_sess_density) >= 3 else None

    # NEW (long-horizon trajectory): context-noise ratio per session.
    # Volume of input tokens carried in / count of distinct entities the
    # agent emits. Rising = agent hauling more cruft per unit of useful
    # output — a direct proxy for "effective signal density declines as
    # accumulated context grows."
    context_noise_traj = _context_noise_ratio_trajectory(sessions)
    context_noise_slope = _ols(context_noise_traj) if len(context_noise_traj) >= 3 else None

    coverage = _compression_coverage(sessions, per_sess_sat, saturation_threshold)

    # Saturation-aware verdict for the long-horizon trajectory.
    # context_noise_ratio is unbounded above; rising = degradation.
    from ._verdict import degradation_verdict
    context_noise_verdict = degradation_verdict(
        context_noise_traj, context_noise_slope,
        rising_is_bad=True, slope_eps=0.01,
    )

    return {
        "saturation_session_rate":         round(sat_session_rate, 4),
        "saturation_slope":                (round(sat_slope, 6) if sat_slope is not None else None),
        "saturation_trajectory":           [(i, round(x, 4)) for i, x in enumerate(per_sess_sat)],
        "self_contradiction_rate":         round(contradiction_rate, 4),
        "fact_density_slope":              (round(density_slope, 6) if density_slope is not None else None),
        "context_noise_ratio_trajectory":  [round(x, 3) if x is not None else None for x in context_noise_traj],
        "context_noise_slope":             (round(context_noise_slope, 4) if context_noise_slope is not None else None),
        "context_noise_verdict":           context_noise_verdict,
        "coverage":                        coverage.as_dict(),
        "derived_from":                    "telemetry",
    }


def _context_noise_ratio_trajectory(sessions: list) -> list:
    """Per session: input_tokens carried in / distinct entities emitted by agent.
    Rising trajectory = signal density falling as accumulated context grows.
    Returns None per session that has no agent output (avoids /0).
    """
    from ._text_utils import extract_capitalised_entities

    out = []
    for s in sessions:
        ctx_in = sum(r.input_tokens for r in s if r.role == "agent")
        agent_text = " ".join(r.response_preview or "" for r in s if r.role == "agent")
        ents = extract_capitalised_entities(agent_text)
        if not agent_text or not ents:
            out.append(None)
            continue
        out.append(ctx_in / len(ents))
    return out


def _ols(ys):
    """Slope tolerating None entries (those are skipped)."""
    pairs = [(i, y) for i, y in enumerate(ys) if y is not None]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    den = sum((p[0] - mx) ** 2 for p in pairs)
    return num / den if den else None


def _empty() -> dict:
    return {
        "saturation_session_rate":  None,
        "saturation_slope":         None,
        "saturation_trajectory":    [],
        "self_contradiction_rate":  None,
        "fact_density_slope":       None,
        "coverage":                 CoverageReport(0, 0.0, "no_test_fired").as_dict(),
        "derived_from":             "telemetry",
    }


def _extract_claims_per_session(
    sessions: list[list[TelemetryRecord]],
) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """Returns (entity, attribute) → [(session_idx, value), ...] claims."""
    out: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for s_idx, s in enumerate(sessions):
        for r in s:
            if r.role != "agent" or not r.response_preview:
                continue
            for m in _CLAIM_RE.finditer(r.response_preview):
                ent = m.group(1).strip().lower()
                attr = m.group(2).strip().lower()
                val = m.group(3).strip().rstrip(".,;:").lower()
                out[(ent, attr)].append((s_idx, val))
    return out


def _self_contradiction_rate(
    claims: dict[tuple[str, str], list[tuple[int, str]]],
) -> float:
    contradictions, total_pairs = 0, 0
    for occurrences in claims.values():
        for i in range(len(occurrences)):
            for j in range(i + 1, len(occurrences)):
                _, v_i = occurrences[i]
                _, v_j = occurrences[j]
                total_pairs += 1
                if v_i != v_j:
                    contradictions += 1
    return contradictions / max(total_pairs, 1)


def _ols_slope(ys: list[float]) -> Optional[float]:
    if len(ys) < 2:
        return None
    n = len(ys)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None


def _compression_coverage(
    sessions: list[list[TelemetryRecord]],
    per_sess_sat: list[float],
    threshold: float,
) -> CoverageReport:
    n_sess = len(sessions)
    n_pressured = sum(1 for x in per_sess_sat if x > threshold)
    if n_sess == 0:
        return CoverageReport(0, 0.0, "no_test_fired")
    cov = n_pressured / n_sess
    if cov > 0.3:
        verdict = "strong"
    elif cov > 0.1:
        verdict = "adequate"
    elif n_pressured > 0:
        verdict = "weak"
    else:
        verdict = "underpowered"
    return CoverageReport(n_observations=n_sess, coverage_fraction=round(cov, 3), verdict=verdict)
