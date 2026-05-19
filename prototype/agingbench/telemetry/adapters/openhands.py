"""
adapters/openhands.py — OpenHands event log → TelemetryRecord.

OpenHands persists agent runs as a JSONL event stream (one event per
line). The dominant event shapes are:

  - {"source": "user", "action": "message", "message": "..."}
  - {"source": "agent", "action": "run"|"edit"|"browse"|...,
                       "args": {...}}
  - {"source": "environment", "observation": "run"|"read"|...,
                              "content": "..."}
  - {"source": "agent", "message": "..."}              (assistant text)
  - {"source": "agent", "llm_metrics": {"prompt_tokens": ...,
                                        "completion_tokens": ...,
                                        "model": ...}}

Sessions are identified by `conversation_id` (preferred) or
`session_id` on each event. Timestamps are ISO-8601 strings; older
runs use epoch seconds.

This adapter reuses the same vocabulary the in-tree OpenHands bridge
emits when running scenario-mode evaluations (see
`agingbench/core/adapters/openhands_bridge.py`), so live-evaluation and
post-hoc telemetry analysis converge on the same canonical
`TelemetryRecord` shape.
"""
from __future__ import annotations

import uuid
from typing import Optional

from ..schema import TelemetryRecord, ToolCall
from .generic import _parse_ts


_AGENT_SOURCES = {"agent", "assistant"}
_USER_SOURCES = {"user", "human"}


def _session(ev: dict) -> Optional[str]:
    for k in ("conversation_id", "session_id", "sessionId", "thread_id"):
        v = ev.get(k)
        if v is not None:
            return str(v)
    return None


def _tool_call_from_action(ev: dict) -> Optional[ToolCall]:
    """Most OpenHands actions correspond directly to a tool invocation:
       run → bash; edit → str_replace; browse → browser; read → file_read."""
    action = ev.get("action")
    if not action or action == "message":
        return None
    args = ev.get("args") or {}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return ToolCall(
        name=str(action),
        args=args,
        result_summary=None,  # observation arrives in a separate event
    )


def normalize(ev: dict) -> Optional[TelemetryRecord]:
    if not isinstance(ev, dict):
        return None

    source = (ev.get("source") or "").lower()
    action = ev.get("action")
    obs = ev.get("observation")
    metrics = ev.get("llm_metrics") or {}
    msg = ev.get("message") or ev.get("content")

    sess = _session(ev)
    ts = _parse_ts(ev.get("timestamp") or ev.get("created_at") or ev.get("time"))
    cid = str(ev.get("id") or ev.get("event_id") or uuid.uuid4())

    # Aggregate metrics event (no message content; just usage and model)
    if metrics and not action and not msg:
        return TelemetryRecord(
            timestamp=ts,
            call_id=cid,
            role="agent",
            session_id=sess,
            input_tokens=int(metrics.get("prompt_tokens") or metrics.get("input_tokens") or 0),
            output_tokens=int(metrics.get("completion_tokens") or metrics.get("output_tokens") or 0),
            cost_usd=metrics.get("cost") if isinstance(metrics.get("cost"), (int, float)) else None,
            model_id=metrics.get("model"),
            source_format="openhands",
            raw=ev,
        )

    # User/agent message event
    if msg and source in _USER_SOURCES | _AGENT_SOURCES and not action:
        role = "user" if source in _USER_SOURCES else "agent"
        text = str(msg)[:1000]
        return TelemetryRecord(
            timestamp=ts,
            call_id=cid,
            role=role,
            session_id=sess,
            prompt_preview=text if role == "user" else None,
            response_preview=text if role == "agent" else None,
            source_format="openhands",
            raw=ev,
        )

    # Agent action event (run, edit, browse, read, ...)
    if source in _AGENT_SOURCES and action and action != "message":
        tc = _tool_call_from_action(ev)
        usage = ev.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return TelemetryRecord(
            timestamp=ts,
            call_id=cid,
            role="agent",
            session_id=sess,
            input_tokens=in_tok,
            output_tokens=out_tok,
            tool_calls=[tc] if tc else [],
            response_preview=str(ev.get("thought"))[:500] if ev.get("thought") else None,
            model_id=ev.get("model"),
            source_format="openhands",
            raw=ev,
        )

    # Action message variant: source=agent, action=message, message=...
    if source in _AGENT_SOURCES and action == "message" and msg:
        return TelemetryRecord(
            timestamp=ts,
            call_id=cid,
            role="agent",
            session_id=sess,
            response_preview=str(msg)[:1000],
            source_format="openhands",
            raw=ev,
        )

    # Environment observation — useful as tool-result context (not its own LLM call)
    if source == "environment" and obs:
        return TelemetryRecord(
            timestamp=ts,
            call_id=cid,
            role="tool",
            session_id=sess,
            response_preview=str(msg or "")[:500] if msg else None,
            source_format="openhands",
            raw=ev,
        )

    return None
