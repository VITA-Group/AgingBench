"""
outcome_extractors.py — Derive OutcomeEvents from secondary sources.

The pipeline can ingest outcomes from three places:
  1. A JSONL file the user provides (outcomes_jsonl=)
  2. Extractors run against the same records the pipeline already has
  3. Extractors that read external sources (git log, CI, etc.)

This module hosts the extractor implementations. They all return
list[OutcomeEvent] which the pipeline merges before mechanism inference.

Extractor spec syntax (used by trace_to_card_v11's extract_outcomes
parameter):

    "claude_session_flags"            # bare name — runs against records
    "record_patterns"                 # bare name — runs against records
    "git_log:/path/to/repo"           # name:arg — runs against external source
    "git_log:/path/to/repo:since=30"  # name:arg:kwarg — kwargs after second colon

All extractors are best-effort: if the source is unavailable they emit
an empty list and a warning rather than crashing the pipeline.
"""
from __future__ import annotations

import re
import subprocess
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .schema import OutcomeEvent, TelemetryRecord


# -----------------------------------------------------------------------------
# 1. claude_session_flags — detect /clear, /reset, /new commands in user msgs
# -----------------------------------------------------------------------------

_CLAUDE_RESET_FLAGS = (
    re.compile(r"\s*/clear\b", re.IGNORECASE),
    re.compile(r"\s*/reset\b", re.IGNORECASE),
    re.compile(r"\s*/new\b",   re.IGNORECASE),
    re.compile(r"\s*/end\b",   re.IGNORECASE),
)


def extract_from_claude_session_flags(
    records: list[TelemetryRecord], **kwargs
) -> list[OutcomeEvent]:
    """Detect user-issued reset commands. Each fires an `abandoned` outcome
    for the task immediately preceding it — the user gave up on whatever
    the agent was doing and started over.
    """
    out: list[OutcomeEvent] = []
    last_task: Optional[str] = None
    last_session: Optional[str] = None
    for r in records:
        if r.role != "user" or not r.prompt_preview:
            if r.task_id:
                last_task = r.task_id
            if r.session_id:
                last_session = r.session_id
            continue
        text = r.prompt_preview
        for pat in _CLAUDE_RESET_FLAGS:
            if pat.search(text):
                if last_session:
                    out.append(OutcomeEvent(
                        session_id=last_session,
                        task_id=last_task or last_session,
                        outcome="abandoned",
                        timestamp=r.timestamp,
                        source="claude_session_flags",
                    ))
                break
    return out


# -----------------------------------------------------------------------------
# 2. record_patterns — implicit failure from user negation patterns
# -----------------------------------------------------------------------------

# A user message that contains one of these patterns near the start is
# strong evidence the prior agent response was unsatisfactory.
_NEGATIVE_USER_PATTERNS = (
    re.compile(r"^(?:no,?|that'?s\s+wrong|that'?s\s+not\s+right|try\s+again|"
               r"do\s+(?:it|that)\s+again|undo|revert)", re.IGNORECASE),
    re.compile(r"^(?:not\s+quite|incorrect|wrong)\b", re.IGNORECASE),
)
# Positive (success-ish) signals
_POSITIVE_USER_PATTERNS = (
    re.compile(r"^(?:thanks|thank\s+you|great|perfect|excellent|nice|"
               r"that\s+works|looks\s+good)\b", re.IGNORECASE),
)


def extract_from_record_patterns(
    records: list[TelemetryRecord], **kwargs
) -> list[OutcomeEvent]:
    """Detect implicit per-turn outcomes from user-message tone.

    Each (agent → user) pair where the user's next message starts with a
    negation/correction pattern fires a `fail` outcome attributed to the
    prior agent task. Positive acknowledgements fire `success`.
    """
    out: list[OutcomeEvent] = []
    last_agent_task: Optional[str] = None
    last_agent_session: Optional[str] = None
    last_agent_ts: Optional[datetime] = None

    for r in records:
        if r.role == "agent":
            last_agent_task = r.task_id or r.call_id
            last_agent_session = r.session_id
            last_agent_ts = r.timestamp
            continue
        if r.role != "user" or not r.prompt_preview or not last_agent_session:
            continue
        text = r.prompt_preview.strip()
        outcome_val: Optional[str] = None
        for pat in _NEGATIVE_USER_PATTERNS:
            if pat.search(text):
                outcome_val = "fail"
                break
        if outcome_val is None:
            for pat in _POSITIVE_USER_PATTERNS:
                if pat.search(text):
                    outcome_val = "success"
                    break
        if outcome_val is not None:
            out.append(OutcomeEvent(
                session_id=last_agent_session,
                task_id=last_agent_task or last_agent_session,
                outcome=outcome_val,
                timestamp=r.timestamp,
                user_signal="thumbs_down" if outcome_val == "fail" else "thumbs_up",
                source="record_patterns",
            ))
    return out


# -----------------------------------------------------------------------------
# 3. git_log — scan a repo for reverts as failure signals
# -----------------------------------------------------------------------------

_REVERT_RE = re.compile(r'^[Rr]evert(?:s|ing)?\s+["\']?(.+?)["\']?$')


def extract_from_git_log(
    records: list[TelemetryRecord],
    repo: Optional[str] = None,
    since_days: int = 90,
    **kwargs,
) -> list[OutcomeEvent]:
    """Scan a git repository for revert commits. Each detected revert is
    treated as a `revision_fail` outcome — evidence that an earlier agent
    contribution had to be undone.

    Tries to link the revert to a TelemetryRecord by matching commit
    timestamp ± 24h with the records the agent produced.
    """
    repo_path = Path(repo) if repo else None
    if repo_path is None or not (repo_path / ".git").exists():
        warnings.warn(
            f"git_log extractor: repo {repo!r} does not contain a .git/ directory; "
            f"emitting no events.",
            RuntimeWarning,
        )
        return []
    since = datetime.now(tz=timezone.utc) - timedelta(days=since_days)

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "log",
             f"--since={since.date().isoformat()}",
             "--format=%H%x00%ct%x00%s"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as e:
        warnings.warn(f"git_log extractor: subprocess failed: {e}", RuntimeWarning)
        return []
    if result.returncode != 0:
        warnings.warn(
            f"git_log extractor: git log returned {result.returncode}: "
            f"{result.stderr.strip()[:200]}",
            RuntimeWarning,
        )
        return []

    out: list[OutcomeEvent] = []
    for line in result.stdout.splitlines():
        parts = line.split("\x00")
        if len(parts) < 3:
            continue
        sha, commit_ts_str, subject = parts[0], parts[1], parts[2]
        m = _REVERT_RE.match(subject)
        if not m:
            continue
        try:
            commit_ts = datetime.fromtimestamp(int(commit_ts_str), tz=timezone.utc)
        except ValueError:
            continue

        # Try to find a TelemetryRecord that produced the reverted content.
        # Heuristic: agent record whose timestamp is within 24h before the
        # revert AND whose response mentions the reverted-commit subject.
        linked_record = _link_record_to_revert(records, commit_ts, m.group(1))
        out.append(OutcomeEvent(
            session_id=(linked_record.session_id if linked_record else f"git_revert::{sha[:8]}"),
            task_id=(linked_record.task_id or linked_record.call_id
                     if linked_record else f"git_revert::{sha[:8]}"),
            outcome="revision_fail",
            timestamp=commit_ts,
            correction_of=(linked_record.task_id or linked_record.call_id
                           if linked_record else None),
            gold_label={"git_sha": sha, "reverted_subject": m.group(1)},
            source="git_log",
        ))
    return out


def _link_record_to_revert(
    records: list[TelemetryRecord],
    revert_ts: datetime,
    reverted_subject: str,
) -> Optional[TelemetryRecord]:
    """Find the agent record most likely to have produced the now-reverted commit."""
    subject_key = reverted_subject.lower()[:40]
    candidates = []
    for r in records:
        if r.role != "agent" or not r.response_preview:
            continue
        if r.timestamp > revert_ts:
            continue
        if (revert_ts - r.timestamp) > timedelta(hours=24):
            continue
        if subject_key in r.response_preview.lower():
            candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.timestamp)


# -----------------------------------------------------------------------------
# Registry + dispatcher
# -----------------------------------------------------------------------------

EXTRACTORS: dict[str, Callable] = {
    "claude_session_flags": extract_from_claude_session_flags,
    "record_patterns":      extract_from_record_patterns,
    "git_log":              extract_from_git_log,
}


def list_extractors() -> list[str]:
    return sorted(EXTRACTORS.keys())


def parse_extractor_spec(spec: str) -> tuple[str, dict]:
    """Parse 'name' or 'name:arg' or 'name:arg:k=v[:k=v...]' into (name, kwargs).

    For backward-friendly simplicity:
      "claude_session_flags"             → ("claude_session_flags", {})
      "git_log:./my-repo"                → ("git_log", {"repo": "./my-repo"})
      "git_log:./my-repo:since_days=30"  → ("git_log", {"repo": "./my-repo", "since_days": 30})
    """
    parts = spec.split(":")
    name = parts[0]
    if name not in EXTRACTORS:
        raise ValueError(
            f"Unknown extractor {name!r}. Known: {list_extractors()}"
        )
    kwargs: dict = {}
    if len(parts) >= 2 and parts[1]:
        # If the first arg looks like "k=v", treat it as a kwarg; otherwise
        # name-specific positional (currently only git_log uses this slot for `repo`)
        first = parts[1]
        if "=" in first:
            k, v = first.split("=", 1)
            kwargs[k] = _coerce(v)
        else:
            # Positional → "repo" for git_log; ignored otherwise
            kwargs["repo"] = first
    for piece in parts[2:]:
        if "=" in piece:
            k, v = piece.split("=", 1)
            kwargs[k] = _coerce(v)
    return name, kwargs


def _coerce(s: str) -> Any:
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def run_extractor(
    spec: str, records: list[TelemetryRecord]
) -> list[OutcomeEvent]:
    """Run an extractor by spec string. Returns [] on error (with warning)."""
    try:
        name, kwargs = parse_extractor_spec(spec)
    except ValueError as e:
        warnings.warn(f"outcome extractor spec invalid: {e}", RuntimeWarning)
        return []
    fn = EXTRACTORS[name]
    try:
        return fn(records, **kwargs)
    except Exception as e:
        warnings.warn(f"outcome extractor {name!r} failed: {e}", RuntimeWarning)
        return []
