"""
inference/revision.py — Revision-aging detection across three tiers.

Three-tier fallback ladder (preferred → fallback):

  1. infer_revision_v2()          — tool_result_update_propagation
                                    Uses ToolCall.result_summary to anchor
                                    "world-said-otherwise". Strongest signal;
                                    requires the adapter to populate
                                    result_summary (generic, openai_assistants,
                                    langsmith do; claude_code, openhands,
                                    langfuse, otlp don't).

  2. infer_revision_args_only()   — tool_argument_self_reversion
                                    Tracks (arg_key, arg_value) across
                                    sessions; flags reversion to a prior value
                                    after the agent had moved past it.
                                    Structural; universal across the 7 adapters.

  3. infer_revision_legacy()      — user_correction_text_patterns
                                    Regex over user turns (the pre-v2 signal).
                                    Final fallback; visually downweighted on
                                    the card.

The public `infer_revision()` dispatcher picks the most informative tier the
trace supports. All three tiers emit the same dict shape, including legacy
field-name aliases (`per_session_violation_trajectory`, etc.) so the website
sparkline and existing tests keep working.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from typing import Optional

from ..schema import TelemetryRecord, CoverageReport


# ─── Tier-3 (legacy) patterns ────────────────────────────────────────────────

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


# ─── Public dispatcher ───────────────────────────────────────────────────────

def infer_revision(sessions: list[list[TelemetryRecord]]) -> dict:
    """Public entry point. Dispatches to the most-informative tier the trace
    supports, falling back as needed. Always returns the same dict shape."""
    if _trace_has_result_summary(sessions):
        return infer_revision_v2(sessions)
    if _trace_has_tool_args(sessions):
        return infer_revision_args_only(sessions)
    return infer_revision_legacy(sessions)


def _trace_has_result_summary(sessions) -> bool:
    for s in sessions:
        for r in s:
            for tc in r.tool_calls or []:
                if tc.result_summary:
                    return True
    return False


def _trace_has_tool_args(sessions) -> bool:
    for s in sessions:
        for r in s:
            for tc in r.tool_calls or []:
                if tc.args:
                    return True
    return False


# ─── Tier 1: tool-result update propagation (preferred) ──────────────────────

def infer_revision_v2(sessions: list[list[TelemetryRecord]]) -> dict:
    """Build an (entity_id, attribute) -> [(session_idx, value)] timeline from
    ToolCall.result_summary. Per session, count agent tool-call args citing a
    value older than the most-recent result for the same (entity, attribute).
    Rising trajectory = revision aging.

    Emits canonical field names (`value_supersession_*`) AND legacy aliases
    (`per_session_violation_*`) so the website sparkline renders without JS
    change.
    """
    # Timeline: key -> list of (session_idx, record_idx, value)
    timeline: dict[str, list[tuple[int, int, str]]] = {}

    for sidx, s in enumerate(sessions):
        for ridx, r in enumerate(s):
            for tc in r.tool_calls or []:
                if not tc.result_summary:
                    continue
                for k, v in _parse_kv_pairs(tc.result_summary).items():
                    timeline.setdefault(k, []).append((sidx, ridx, str(v)))

    # Walk forward: for each agent tool_call args, check if any value
    # references a key whose latest result is different.
    per_session_violations: list[int] = [0] * len(sessions)
    total_stale = 0
    for sidx, s in enumerate(sessions):
        for ridx, r in enumerate(s):
            if r.role != "agent":
                continue
            for tc in r.tool_calls or []:
                if not tc.args:
                    continue
                for arg_key, arg_val in _flatten_args(tc.args).items():
                    # Does this arg's value match a *prior* (older) state of any key?
                    for key, history in timeline.items():
                        prior_states = [(t_s, t_v) for (t_s, _r, t_v) in history if t_s <= sidx]
                        if len(prior_states) < 2:
                            continue
                        latest_value = prior_states[-1][1]
                        prior_values = {v for _, v in prior_states[:-1]}
                        if (str(arg_val) in prior_values
                                and str(arg_val) != latest_value):
                            per_session_violations[sidx] += 1
                            total_stale += 1
                            break  # one violation per arg

    violation_slope = _ols(per_session_violations) if len(per_session_violations) >= 3 else None
    from ._verdict import degradation_verdict
    violation_verdict = degradation_verdict(
        per_session_violations, violation_slope,
        rising_is_bad=True, floor_threshold=0.01, slope_eps=0.005,
    )

    coverage = _v2_coverage(timeline, len(sessions))

    return _emit(
        derived_from="tool_result_update_propagation",
        violations=per_session_violations,
        slope=violation_slope,
        verdict=violation_verdict,
        coverage=coverage,
        n_stale_propagations=total_stale,
        # Tier-3 fields zeroed out (kept for shape compat)
        repetition_rate=0.0, stale_rate=0.0, median_ttadopt=None,
        n_corrections=0, n_repeated_corrections=0,
        # Tier-1-specific
        n_entities_tracked=len(timeline),
    )


# ─── Tier 2: argument-only self-reversion ────────────────────────────────────

def infer_revision_args_only(sessions: list[list[TelemetryRecord]]) -> dict:
    """Track (arg_key, arg_value) across agent tool_call args. Pattern of
    interest: a key's value moves v1 -> v2 -> v1, indicating the agent
    reverted to a stale value it had previously moved past.

    Weaker than v2 (no external anchor confirming v2 is "the truth") but
    structural and universal — every adapter populates args. The signal still
    constrains a reasonable revision-aging story.
    """
    # Per-key timeline of values
    key_history: dict[str, list[tuple[int, str]]] = {}

    for sidx, s in enumerate(sessions):
        for r in s:
            if r.role != "agent":
                continue
            for tc in r.tool_calls or []:
                for k, v in _flatten_args(tc.args or {}).items():
                    key_history.setdefault(k, []).append((sidx, str(v)))

    per_session_violations: list[int] = [0] * len(sessions)
    total_stale = 0
    for k, hist in key_history.items():
        # Look for v1 -> v2 -> v1 pattern (a reversion)
        seen_values_by_session: dict[int, set[str]] = {}
        for sidx, v in hist:
            seen_values_by_session.setdefault(sidx, set()).add(v)
        if len(set(v for _, v in hist)) < 2:
            continue  # key didn't change values, no reversion possible
        # Compress to sequence of distinct values across time (preserving order)
        distinct_seq: list[tuple[int, str]] = []
        for sidx, v in hist:
            if not distinct_seq or distinct_seq[-1][1] != v:
                distinct_seq.append((sidx, v))
        # Now look for v_i = v_j where j > i + 1 (skipped at least one value)
        for j in range(2, len(distinct_seq)):
            sidx_j, v_j = distinct_seq[j]
            earlier_values = {v for _, v in distinct_seq[:j-1]}
            if v_j in earlier_values:
                per_session_violations[sidx_j] += 1
                total_stale += 1

    violation_slope = _ols(per_session_violations) if len(per_session_violations) >= 3 else None
    from ._verdict import degradation_verdict
    violation_verdict = degradation_verdict(
        per_session_violations, violation_slope,
        rising_is_bad=True, floor_threshold=0.01, slope_eps=0.005,
    )

    coverage = _args_only_coverage(key_history, len(sessions))

    return _emit(
        derived_from="tool_argument_self_reversion",
        violations=per_session_violations,
        slope=violation_slope,
        verdict=violation_verdict,
        coverage=coverage,
        n_stale_propagations=total_stale,
        repetition_rate=0.0, stale_rate=0.0, median_ttadopt=None,
        n_corrections=0, n_repeated_corrections=0,
        n_entities_tracked=len(key_history),
    )


# ─── Tier 3: legacy user-correction regex (final fallback) ───────────────────

def infer_revision_legacy(sessions: list[list[TelemetryRecord]]) -> dict:
    corrections = _detect_corrections(sessions)
    n_corrections = len(corrections)

    repeated = 0
    seen_in_session: dict[tuple[int, str], int] = {}
    for c in corrections:
        key = (c.session_idx, c.new_value.lower())
        if key in seen_in_session:
            repeated += 1
        else:
            seen_in_session[key] = c.turn_idx
        for d in corrections:
            if d.session_idx > c.session_idx and d.new_value.lower() == c.new_value.lower():
                repeated += 1
                break

    time_to_adopt: list[int] = []
    stale_after, total_after = 0, 0
    for c in corrections:
        adopted_at = None
        nv_low = c.new_value.lower()
        ov_low = (c.old_value or "").lower() if c.old_value else None
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

    violation_traj = _legacy_violation_trajectory(sessions, corrections)
    violation_slope = _ols(violation_traj) if len(violation_traj) >= 3 else None
    from ._verdict import degradation_verdict
    violation_verdict = degradation_verdict(
        violation_traj, violation_slope,
        rising_is_bad=True, floor_threshold=0.01, slope_eps=0.005,
    )

    coverage = _legacy_coverage(n_corrections, len(sessions))

    return _emit(
        derived_from="user_correction_text_patterns_fallback",
        violations=violation_traj,
        slope=violation_slope,
        verdict=violation_verdict,
        coverage=coverage,
        n_stale_propagations=sum(violation_traj),
        repetition_rate=repetition_rate,
        stale_rate=stale_rate,
        median_ttadopt=median_ttadopt,
        n_corrections=n_corrections,
        n_repeated_corrections=repeated,
        n_entities_tracked=0,
    )


# ─── Shared emit (canonical + legacy field names) ────────────────────────────

def _emit(
    *,
    derived_from: str,
    violations: list[int],
    slope: Optional[float],
    verdict: str,
    coverage: CoverageReport,
    n_stale_propagations: int,
    repetition_rate: float,
    stale_rate: float,
    median_ttadopt,
    n_corrections: int,
    n_repeated_corrections: int,
    n_entities_tracked: int,
) -> dict:
    slope_rounded = round(slope, 4) if slope is not None else None
    return {
        # Canonical new names (used by the selector + future website)
        "value_supersession_trajectory":         violations,
        "value_supersession_slope":              slope_rounded,
        "value_supersession_verdict":            verdict,
        # Legacy aliases (used by the existing website sparkline + tests)
        "per_session_violation_trajectory":      violations,
        "violation_trajectory_slope":            slope_rounded,
        "violation_trajectory_verdict":          verdict,
        # Counters
        "n_stale_propagations":                  n_stale_propagations,
        "n_entities_tracked":                    n_entities_tracked,
        # Tier-3 fields (zeroed for non-legacy tiers, populated for legacy)
        "correction_repetition_rate":            round(repetition_rate, 4),
        "stale_value_citation_rate":             round(stale_rate, 4),
        "median_time_to_adopt_correction_turns": median_ttadopt,
        "n_corrections_detected":                n_corrections,
        "n_repeated_corrections":                n_repeated_corrections,
        # Metadata
        "coverage":                              coverage.as_dict(),
        "derived_from":                          derived_from,
    }


# ─── Helpers ────────────────────────────────────────────────────────────────

def _parse_kv_pairs(text: str) -> dict[str, str]:
    """Extract (key, value) pairs from a result_summary string.

    Tries JSON first (covers structured tool results). Falls back to a
    coarse "key=value" / "key: value" / "X is Y" regex sweep so even
    semi-structured text yields some keys.
    """
    out: dict[str, str] = {}
    if not text:
        return out
    text = text.strip()
    # JSON path
    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (str, int, float, bool)):
                    out[str(k)] = str(v)
            return out
    # Regex sweep
    for m in re.finditer(r"['\"]?([\w.-]+)['\"]?\s*[:=]\s*['\"]?([\w./@-]+)['\"]?", text):
        out[m.group(1)] = m.group(2)
    return out


def _flatten_args(args: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested args into dotted keys for self-reversion tracking."""
    out: dict[str, str] = {}
    for k, v in (args or {}).items():
        full = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_args(v, prefix=full))
        elif isinstance(v, (str, int, float, bool)):
            out[full] = str(v)
    return out


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
                        old_value=(m.group(2).strip()
                                   if m.lastindex and m.lastindex >= 2 and m.group(2)
                                   else None),
                        raw=text[:200],
                    ))
                    break
    return out


def _legacy_violation_trajectory(sessions: list, corrections: list) -> list:
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


def _v2_coverage(timeline: dict, n_sessions: int) -> CoverageReport:
    n_updates = sum(max(0, len(v) - 1) for v in timeline.values())
    if n_updates == 0:
        return CoverageReport(0, 0.0, "no_test_fired")
    if n_updates >= 10:
        verdict = "strong"
    elif n_updates >= 3:
        verdict = "adequate"
    else:
        verdict = "weak"
    return CoverageReport(n_updates, round(n_updates / n_sessions, 3) if n_sessions else 0.0, verdict)


def _args_only_coverage(key_history: dict, n_sessions: int) -> CoverageReport:
    changing_keys = sum(1 for hist in key_history.values()
                        if len({v for _, v in hist}) >= 2)
    if changing_keys == 0:
        return CoverageReport(0, 0.0, "no_test_fired")
    if changing_keys >= 5:
        verdict = "adequate"
    elif changing_keys >= 1:
        verdict = "weak"
    else:
        verdict = "no_test_fired"
    return CoverageReport(changing_keys, round(changing_keys / n_sessions, 3) if n_sessions else 0.0, verdict)


def _legacy_coverage(n_corrections: int, n_sessions: int) -> CoverageReport:
    if n_sessions == 0 or n_corrections == 0:
        return CoverageReport(0, 0.0, "no_test_fired")
    rate = n_corrections / n_sessions
    if n_corrections >= 20:
        verdict = "strong"
    elif n_corrections >= 5:
        verdict = "adequate"
    else:
        verdict = "weak"
    return CoverageReport(n_corrections, round(rate, 3), verdict)
