"""
inference/revision.py — User-correction-pattern detection.

The strongest telemetry signal we have. When the user has to repeat a
correction, that's a real, user-felt revision-aging failure — directly
visible in trace text.

Detection: regex patterns matching common correction phrasings in user
turns. For each detected correction:
  - Did the agent adopt the new value in subsequent responses?
  - Did the user have to issue the same correction again?
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Optional

from ..schema import TelemetryRecord, CoverageReport


# Patterns ordered by specificity. Each captures (new_value, [old_value]).
_CORRECTION_PATTERNS = [
    re.compile(r"actually,?\s+(?:use|it'?s|that'?s)\s+([\w./-]+)\s+(?:instead\s+of|not)\s+([\w./-]+)", re.IGNORECASE),
    re.compile(r"no,?\s+([\w./-]+)\s+(?:not|instead\s+of)\s+([\w./-]+)", re.IGNORECASE),
    re.compile(r"use\s+([\w./-]+)\s+(?:not|rather\s+than|instead\s+of)\s+([\w./-]+)", re.IGNORECASE),
    re.compile(r"i\s+meant\s+([\w./-]+)(?:\s*,?\s+not\s+([\w./-]+))?", re.IGNORECASE),
    re.compile(r"correction[:.]?\s+([\w./-]+)", re.IGNORECASE),
    re.compile(r"(?:please\s+)?(?:always\s+)?(?:remember\s+to\s+)?use\s+([\w./-]+)\s+from\s+now\s+on", re.IGNORECASE),
    re.compile(r"let\s+me\s+correct(?:\s+that)?[:.]?\s+([\w./-]+)", re.IGNORECASE),
]


@dataclass
class _Correction:
    session_idx: int
    turn_idx: int
    new_value: str
    old_value: Optional[str]
    raw: str


def infer_revision(sessions: list[list[TelemetryRecord]]) -> dict:
    corrections = _detect_corrections(sessions)
    n_corrections = len(corrections)

    # Repetition: was the same correction (same new_value, same session-or-later) issued more than once?
    repeated = 0
    seen_in_session: dict[tuple[int, str], int] = {}
    for c in corrections:
        key = (c.session_idx, c.new_value.lower())
        if key in seen_in_session:
            repeated += 1
        else:
            seen_in_session[key] = c.turn_idx
        # Also: same new_value in a LATER session = the correction didn't stick
        for d in corrections:
            if d.session_idx > c.session_idx and d.new_value.lower() == c.new_value.lower():
                repeated += 1
                break  # count once per c

    # Time-to-adopt: turns between correction issue and first agent response containing new_value
    time_to_adopt: list[int] = []
    stale_after, total_after = 0, 0
    for c in corrections:
        adopted_at = None
        nv_low = c.new_value.lower()
        ov_low = (c.old_value or "").lower() if c.old_value else None
        # Walk forward in the same session for agent responses
        sess = sessions[c.session_idx]
        for k, r in enumerate(sess[c.turn_idx + 1:], start=c.turn_idx + 1):
            if r.role != "agent" or not r.response_preview:
                continue
            text = r.response_preview.lower()
            if nv_low in text:
                if adopted_at is None:
                    adopted_at = k - c.turn_idx
                total_after += 1
            elif ov_low and ov_low in text:
                stale_after += 1
                total_after += 1
        if adopted_at is not None:
            time_to_adopt.append(adopted_at)

    repetition_rate = repeated / n_corrections if n_corrections else 0.0
    stale_rate = stale_after / total_after if total_after else 0.0
    median_ttadopt = statistics.median(time_to_adopt) if time_to_adopt else None

    # NEW (long-horizon trajectory): per-session count of stale-value
    # citations. For each prior-session correction, count how many later
    # sessions still contain the OLD value in agent output. Rising
    # trajectory = constraint forgetting / belief-update failures
    # accumulating as the project progresses.
    violation_traj = _per_session_violation_trajectory(sessions, corrections)
    violation_slope = _ols(violation_traj) if len(violation_traj) >= 3 else None

    coverage = _revision_coverage(n_corrections, len(sessions))

    # violation count is unbounded above; rising = degradation.
    # If trajectory stays at 0 throughout, that's healthy floor.
    from ._verdict import degradation_verdict
    violation_verdict = degradation_verdict(
        violation_traj, violation_slope,
        rising_is_bad=True, floor_threshold=0.01, slope_eps=0.005,
    )

    return {
        "correction_repetition_rate":            round(repetition_rate, 4),
        "stale_value_citation_rate":             round(stale_rate, 4),
        "median_time_to_adopt_correction_turns": median_ttadopt,
        "n_corrections_detected":                n_corrections,
        "n_repeated_corrections":                repeated,
        "per_session_violation_trajectory":      violation_traj,
        "violation_trajectory_slope":            (round(violation_slope, 4) if violation_slope is not None else None),
        "violation_trajectory_verdict":          violation_verdict,
        "coverage":                              coverage.as_dict(),
        "derived_from":                          "user_correction_text_patterns",
    }


def _per_session_violation_trajectory(
    sessions: list, corrections: list
) -> list:
    """For each session, count agent responses that contain a previously-
    corrected OLD value (without also containing the NEW value).
    """
    out = []
    for s_idx, s in enumerate(sessions):
        agent_text = " ".join(
            (r.response_preview or "").lower()
            for r in s if r.role == "agent"
        )
        if not agent_text:
            out.append(0)
            continue
        violations = 0
        for c in corrections:
            if c.session_idx >= s_idx:
                continue
            old_val = (c.old_value or "").lower().strip()
            new_val = c.new_value.lower().strip()
            if not old_val or len(old_val) < 2:
                continue
            if old_val in agent_text and new_val not in agent_text:
                violations += 1
        out.append(violations)
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


def _detect_corrections(sessions: list[list[TelemetryRecord]]) -> list[_Correction]:
    out: list[_Correction] = []
    for s_idx, session in enumerate(sessions):
        for t_idx, r in enumerate(session):
            if r.role != "user" or not r.prompt_preview:
                continue
            text = r.prompt_preview
            for pat in _CORRECTION_PATTERNS:
                m = pat.search(text)
                if m:
                    out.append(_Correction(
                        session_idx=s_idx,
                        turn_idx=t_idx,
                        new_value=m.group(1).strip(),
                        old_value=(m.group(2).strip() if m.lastindex and m.lastindex >= 2 and m.group(2) else None),
                        raw=text[:200],
                    ))
                    break  # one correction per user turn
    return out


def _revision_coverage(n_corrections: int, n_sessions: int) -> CoverageReport:
    if n_sessions == 0:
        return CoverageReport(0, 0.0, "no_test_fired")
    if n_corrections == 0:
        return CoverageReport(0, 0.0, "no_test_fired")
    rate = n_corrections / n_sessions
    if n_corrections >= 20:
        verdict = "strong"
    elif n_corrections >= 5:
        verdict = "adequate"
    else:
        verdict = "weak"
    return CoverageReport(n_corrections, round(rate, 3), verdict)
