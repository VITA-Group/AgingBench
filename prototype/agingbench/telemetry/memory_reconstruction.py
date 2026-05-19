"""
memory_reconstruction.py — Detect memory-state shock events from a trace.

Maintenance-aging proxy: we can't see the agent's internal memory but we
can see observable transitions (model swap, large prompt-size drops,
system-message changes, cache spikes) that often correlate with shocks.

Each detection carries a confidence score so downstream maintenance
inference can downweight low-confidence events.
"""
from __future__ import annotations

from typing import Optional

from .schema import TelemetryRecord, MemoryEvent


_RESET_PATTERNS = ("/clear", "/reset", "/new")
_CTX_DROP_THRESHOLD   = 0.5   # 50% drop in input_tokens between consecutive calls
_CACHE_SPIKE_THRESHOLD = 0.3  # cache_creation > 30% of input_tokens


def detect_shocks(sessions: list[list[TelemetryRecord]]) -> list[MemoryEvent]:
    """Scan sessions for observable shock-event candidates.

    Detectors (each emits MemoryEvents independently):
      - model_swap:       model_id changes between consecutive records
      - context_reset:    input_tokens drops by >50% within a session
      - system_change:    a system-role record's content changes
      - compression_spike: cache_creation_tokens > 30% of input_tokens
      - explicit_command: user message contains /clear, /reset, /new
    """
    shocks: list[MemoryEvent] = []
    prev_model: Optional[str] = None
    prev_ctx: Optional[int] = None
    prev_system: Optional[str] = None

    for s_idx, session in enumerate(sessions):
        for r in session:
            # (a) Model swap (cross-session OR within session)
            if r.model_id and prev_model and r.model_id != prev_model:
                shocks.append(MemoryEvent(
                    session_idx=s_idx,
                    timestamp=r.timestamp,
                    type="model_swap",
                    confidence=1.0,
                    detail={"from": prev_model, "to": r.model_id},
                ))
            if r.model_id:
                prev_model = r.model_id

            # (b) Large context drop within session = forced compression / reset
            if r.input_tokens > 0 and prev_ctx and prev_ctx - r.input_tokens > _CTX_DROP_THRESHOLD * prev_ctx:
                shocks.append(MemoryEvent(
                    session_idx=s_idx,
                    timestamp=r.timestamp,
                    type="context_reset",
                    confidence=0.7,
                    detail={"from_tokens": prev_ctx, "to_tokens": r.input_tokens},
                ))
            if r.input_tokens > 0:
                prev_ctx = r.input_tokens

            # (c) System message change
            if r.role == "system" and r.prompt_preview:
                if prev_system is not None and r.prompt_preview != prev_system:
                    shocks.append(MemoryEvent(
                        session_idx=s_idx,
                        timestamp=r.timestamp,
                        type="system_change",
                        confidence=0.8,
                        detail={"new_system_preview": (r.prompt_preview or "")[:200]},
                    ))
                prev_system = r.prompt_preview

            # (d) Cache-creation spike (Anthropic prompt-cache pattern)
            if r.input_tokens > 0 and r.cache_creation_tokens > _CACHE_SPIKE_THRESHOLD * r.input_tokens:
                shocks.append(MemoryEvent(
                    session_idx=s_idx,
                    timestamp=r.timestamp,
                    type="compression_spike",
                    confidence=0.5,
                    detail={
                        "cache_creation_tokens": r.cache_creation_tokens,
                        "input_tokens": r.input_tokens,
                    },
                ))

            # (e) Explicit reset commands in user text
            if r.role == "user" and r.prompt_preview:
                low = r.prompt_preview.lower()
                if any(pat in low for pat in _RESET_PATTERNS):
                    shocks.append(MemoryEvent(
                        session_idx=s_idx,
                        timestamp=r.timestamp,
                        type="explicit_command",
                        confidence=0.95,
                        detail={"command_preview": r.prompt_preview[:120]},
                    ))

    return _dedupe_within_session(shocks)


def _dedupe_within_session(shocks: list[MemoryEvent]) -> list[MemoryEvent]:
    """Keep only one event per (session, type) pair (the highest-confidence one)."""
    by_key: dict[tuple[int, str], MemoryEvent] = {}
    for sh in shocks:
        key = (sh.session_idx, sh.type)
        if key not in by_key or sh.confidence > by_key[key].confidence:
            by_key[key] = sh
    return sorted(by_key.values(), key=lambda e: (e.session_idx, e.timestamp))
