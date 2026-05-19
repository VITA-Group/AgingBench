"""
adapters/langfuse_v1.py — Langfuse span → TelemetryRecord.

v1.1 covers the GENERATION span schema (Langfuse's primary LLM-call
span). Other span types (TRACE, SPAN, EVENT) are ignored unless they
carry usage data.

Tolerates both camelCase (Langfuse SDK exports) and snake_case (Langfuse
REST-API JSON downloads), and falls back to `trace_id` for session
grouping when `sessionId` is absent — which is the common case for
traces collected via the OpenTelemetry exporter.
"""
from __future__ import annotations

import uuid
from typing import Optional

from ..schema import TelemetryRecord
from .generic import _parse_ts


def _g(ev: dict, *keys: str, default=None):
    for k in keys:
        if k in ev and ev[k] is not None:
            return ev[k]
    return default


def normalize(ev: dict) -> Optional[TelemetryRecord]:
    if not isinstance(ev, dict):
        return None
    span_type = (ev.get("type") or "").upper()
    usage = ev.get("usage") or ev.get("tokenUsage") or {}
    has_usage = bool(usage)

    # Filter non-LLM spans only when type is explicitly set AND no usage present.
    if span_type and span_type != "GENERATION" and not has_usage:
        return None

    sess = _g(ev, "sessionId", "session_id")
    trace = _g(ev, "traceId", "trace_id")

    in_tok = int(_g(usage, "input", "input_tokens", "promptTokens", "prompt_tokens", default=0) or 0)
    out_tok = int(_g(usage, "output", "output_tokens", "completionTokens", "completion_tokens", default=0) or 0)

    cost = _g(ev, "cost", "calculatedTotalCost", "total_cost")
    if not isinstance(cost, (int, float)):
        cost = None

    inp = ev.get("input")
    outp = ev.get("output")

    return TelemetryRecord(
        timestamp=_parse_ts(_g(ev, "startTime", "start_time", "timestamp")),
        call_id=str(ev.get("id") or uuid.uuid4()),
        role="agent",
        session_id=str(sess) if sess is not None else (str(trace) if trace is not None else None),
        task_id=str(trace) if trace is not None else None,
        input_tokens=in_tok,
        output_tokens=out_tok,
        duration_ms=_g(ev, "latencyMs", "latency_ms", "duration_ms"),
        cost_usd=cost,
        model_id=_g(ev, "model", "model_id"),
        prompt_preview=str(inp)[:1000] if inp else None,
        response_preview=str(outp)[:1000] if outp else None,
        source_format="langfuse",
        raw=ev,
    )
