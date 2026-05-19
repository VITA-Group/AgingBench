"""
adapters/otlp_v1.py — OpenTelemetry/OTLP span → TelemetryRecord.

v1.1 supports OTLP spans serialised as JSON (the JSON line-format used
by `otel-cli`, Jaeger JSON exports, and most observability backends).
Native protobuf decoding is v1.2.

Recognises both the new `gen_ai.*` semantic conventions
  https://opentelemetry.io/docs/specs/semconv/gen-ai/
and the legacy `llm.*` namespace still emitted by older OTel SDKs and
many observability backends (Phoenix/Arize, Logfire, Honeycomb, etc.).
The legacy namespace is the de-facto standard pre-semconv and remains
the most common in production traces today.
"""
from __future__ import annotations

import uuid
from typing import Optional

from ..schema import TelemetryRecord
from .generic import _parse_ts


# (semconv name, legacy name) — first present wins.
_INPUT_TOKEN_KEYS  = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens",
                      "llm.input_tokens", "llm.usage.prompt_tokens", "llm.token_count.prompt")
_OUTPUT_TOKEN_KEYS = ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens",
                      "llm.output_tokens", "llm.usage.completion_tokens", "llm.token_count.completion")
_MODEL_KEYS        = ("gen_ai.request.model", "gen_ai.response.model",
                      "llm.model", "llm.model_name", "llm.request.model")
_SYSTEM_KEYS       = ("gen_ai.system", "llm.system", "llm.vendor")
_PROMPT_KEYS       = ("gen_ai.prompt", "llm.prompts", "llm.input_messages")
_RESPONSE_KEYS     = ("gen_ai.completion", "llm.completions", "llm.output_messages")
_COST_KEYS         = ("gen_ai.usage.cost", "llm.usage.cost", "llm.cost")
_SESSION_KEYS      = ("session.id", "gen_ai.session.id", "llm.session.id", "conversation.id")


def _attr(ev: dict, key: str, default=None):
    """OTLP spans typically have an `attributes` dict (or a list of {key,value})."""
    attrs = ev.get("attributes") or {}
    if isinstance(attrs, dict):
        return attrs.get(key, default)
    if isinstance(attrs, list):
        for item in attrs:
            if isinstance(item, dict) and item.get("key") == key:
                v = item.get("value")
                if isinstance(v, dict):
                    return next(iter(v.values()), default)
                return v
    return default


def _first_attr(ev: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        v = _attr(ev, k)
        if v is not None:
            return v
    return default


def normalize(ev: dict) -> Optional[TelemetryRecord]:
    if not isinstance(ev, dict):
        return None
    # An OTLP span representing an LLM call carries either gen_ai.* or llm.* attributes.
    has_llm = (
        _first_attr(ev, _SYSTEM_KEYS) is not None
        or _first_attr(ev, _MODEL_KEYS) is not None
        or _first_attr(ev, _INPUT_TOKEN_KEYS) is not None
    )
    if not has_llm:
        return None

    in_tok = int(_first_attr(ev, _INPUT_TOKEN_KEYS, 0) or 0)
    out_tok = int(_first_attr(ev, _OUTPUT_TOKEN_KEYS, 0) or 0)
    model = _first_attr(ev, _MODEL_KEYS)
    cost = _first_attr(ev, _COST_KEYS)
    if not isinstance(cost, (int, float)):
        cost = None
    sess = _first_attr(ev, _SESSION_KEYS) or ev.get("sessionId")
    trace = ev.get("traceId") or ev.get("trace_id")

    prompt = _first_attr(ev, _PROMPT_KEYS)
    resp = _first_attr(ev, _RESPONSE_KEYS)

    return TelemetryRecord(
        timestamp=_parse_ts(ev.get("startTimeUnixNano") or ev.get("startTime")
                            or ev.get("start_time") or ev.get("timestamp")),
        call_id=str(ev.get("spanId") or ev.get("span_id") or trace or uuid.uuid4()),
        role="agent",
        session_id=str(sess) if sess is not None else (str(trace) if trace is not None else None),
        task_id=str(trace) if trace is not None else None,
        input_tokens=in_tok,
        output_tokens=out_tok,
        duration_ms=ev.get("durationMs") or ev.get("duration_ms"),
        cost_usd=cost,
        model_id=str(model) if model is not None else None,
        prompt_preview=str(prompt)[:1000] if prompt else None,
        response_preview=str(resp)[:1000] if resp else None,
        source_format="otlp",
        raw=ev,
    )
