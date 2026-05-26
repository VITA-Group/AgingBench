"""
session_detection.py — Group records into sessions.

Strategy (in priority order):
  1. Explicit session_id field (Langfuse, OpenAI threads, Claude Code)
  2. Explicit reset markers in user messages (/clear, /reset)
  3. Pure idle-gap split (last resort)
"""
from __future__ import annotations

from collections import defaultdict

from .schema import TelemetryRecord


_RESET_PATTERNS = ("/clear", "/reset", "/new", "/end", "new conversation")


def detect_sessions(
    records: list[TelemetryRecord],
    idle_gap_minutes: float = 30.0,
) -> tuple[list[list[TelemetryRecord]], str]:
    """Return (sessions, mode) where sessions is list of lists and mode is the
    detection strategy that fired.
    """
    if not records:
        return [], "no_records"

    # Strategy 1: explicit session_id
    if any(r.session_id for r in records):
        groups = defaultdict(list)
        for r in records:
            key = r.session_id or "_unknown"
            groups[key].append(r)
        sessions = [sorted(g, key=lambda r: r.timestamp) for g in groups.values()]
        return _stable_session_order(sessions), "explicit_id"

    sessions = _split_by_idle_gap(records, idle_gap_minutes)
    return sessions, "idle_gap"


def _split_by_idle_gap(
    records: list[TelemetryRecord], idle_gap_minutes: float
) -> list[list[TelemetryRecord]]:
    sorted_records = sorted(records, key=lambda r: r.timestamp)
    sessions: list[list[TelemetryRecord]] = []
    current: list[TelemetryRecord] = []
    last_t = None
    for r in sorted_records:
        # Reset markers in user messages cut a session boundary.
        text = (r.prompt_preview or "").lower()
        is_reset = (r.role == "user"
                    and any(p in text for p in _RESET_PATTERNS))
        if last_t is not None:
            gap_s = (r.timestamp - last_t).total_seconds()
            if gap_s > idle_gap_minutes * 60 or is_reset:
                if current:
                    sessions.append(current)
                current = []
        current.append(r)
        last_t = r.timestamp
    if current:
        sessions.append(current)
    return sessions


def _stable_session_order(sessions: list[list[TelemetryRecord]]) -> list[list[TelemetryRecord]]:
    """Order sessions by their earliest timestamp so downstream indexing is stable."""
    return sorted(sessions, key=lambda s: s[0].timestamp if s else None)
