"""
adapters/generic.py — Best-effort JSONL → TelemetryRecord.

Tolerates heterogeneous shapes. Picks token / timestamp / session fields
out of common alternative names. Returns None if the event has no
discernible token usage AND no message content (= probably not an LLM
call).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ..schema import TelemetryRecord, ToolCall


_TIMESTAMP_KEYS  = ("timestamp", "ts", "time", "startTime", "start_time", "@timestamp")
_INPUT_KEYS      = ("input_tokens", "prompt_tokens", "in_tokens")
_OUTPUT_KEYS     = ("output_tokens", "completion_tokens", "out_tokens")
_SESSION_KEYS    = ("session_id", "sessionId", "conversation_id", "thread_id")
_TASK_KEYS       = ("task_id", "taskId", "ticket_id", "issue_id", "pr_id")
_MODEL_KEYS      = ("model_id", "model", "modelId", "modelName")
_DURATION_KEYS   = ("duration_ms", "latency_ms", "latencyMs", "elapsed_ms")
_COST_KEYS       = ("cost_usd", "cost", "costUSD")
_PROMPT_KEYS     = ("prompt", "prompt_preview", "input", "messages")
_RESPONSE_KEYS   = ("response", "response_preview", "output", "completion")
# `content` is universal in OpenAI/Anthropic message format; routed
# to prompt vs response based on `role` below.
_CONTENT_KEYS    = ("content", "text", "message")
_USER_ROLES      = {"user", "system", "human", "input"}
_ROLE_KEYS       = ("role", "speaker", "actor")
_CTX_KEYS        = ("context_window_size", "context_size", "n_ctx")
_CACHE_CR_KEYS   = ("cache_creation_tokens", "cache_creation_input_tokens")
_CACHE_RD_KEYS   = ("cache_read_tokens", "cache_read_input_tokens")


def _first(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _nested_first(d: dict, parent_keys: tuple[str, ...], child_keys: tuple[str, ...]) -> Any:
    for pk in parent_keys:
        if pk in d and isinstance(d[pk], dict):
            for ck in child_keys:
                if ck in d[pk]:
                    return d[pk][ck]
    return None


def _parse_ts(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)):
        # epoch seconds or millis — heuristic
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, tz=timezone.utc)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    return datetime.now(tz=timezone.utc)


def _coerce_int(x: Any) -> int:
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        try:
            return int(float(x))
        except ValueError:
            return 0
    return 0


def _extract_tool_calls(ev: dict) -> list[ToolCall]:
    raw = ev.get("tool_calls") or ev.get("toolCalls") or ev.get("tools") or []
    out = []
    if isinstance(raw, list):
        for tc in raw:
            if isinstance(tc, dict):
                name = tc.get("name") or tc.get("tool") or tc.get("function") or "unknown"
                if isinstance(name, dict):
                    name = name.get("name", "unknown")
                out.append(ToolCall(
                    name=str(name),
                    args=tc.get("args") or tc.get("arguments") or {},
                    result_summary=str(tc.get("result", ""))[:200] or None,
                    duration_ms=tc.get("duration_ms"),
                    success=tc.get("success"),
                ))
    return out


def _summarise_text(v: Any, limit: int = 800) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v[:limit]
    if isinstance(v, list):
        # OpenAI / Anthropic message arrays
        parts = []
        for item in v:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content")
                if isinstance(t, str):
                    parts.append(t)
                elif isinstance(t, list):
                    for sub in t:
                        if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                            parts.append(sub["text"])
        return " ".join(parts)[:limit] if parts else None
    if isinstance(v, dict):
        for k in ("text", "content"):
            if k in v and isinstance(v[k], str):
                return v[k][:limit]
    return None


def normalize(ev: dict) -> Optional[TelemetryRecord]:
    """Best-effort map of a heterogeneous JSONL row to a TelemetryRecord.

    Returns None when the event has no token usage AND no recognisable
    message content (likely a metadata / log line that's not an LLM call).
    """
    if not isinstance(ev, dict):
        return None

    in_toks = _coerce_int(
        _first(ev, _INPUT_KEYS) or _nested_first(ev, ("usage", "tokenUsage"), _INPUT_KEYS) or 0
    )
    out_toks = _coerce_int(
        _first(ev, _OUTPUT_KEYS) or _nested_first(ev, ("usage", "tokenUsage"), _OUTPUT_KEYS) or 0
    )
    role_raw = _first(ev, _ROLE_KEYS) or ev.get("event") or "agent"
    role_str = str(role_raw).lower() if not isinstance(role_raw, dict) else "agent"

    prompt = _summarise_text(_first(ev, _PROMPT_KEYS))
    response = _summarise_text(_first(ev, _RESPONSE_KEYS))

    # `content` (OpenAI / Anthropic universal message field) is routed to
    # prompt or response based on role.
    content = _summarise_text(_first(ev, _CONTENT_KEYS))
    if content:
        if role_str in _USER_ROLES and not prompt:
            prompt = content
        elif not response:
            response = content

    tool_calls = _extract_tool_calls(ev)

    # If neither usage nor content nor tool calls visible, this isn't an
    # LLM call. Tool-call presence is itself a strong LLM-call signal.
    if in_toks == 0 and out_toks == 0 and not prompt and not response and not tool_calls:
        if not (ev.get("event") in {"tool_call", "tool_use", "outcome"} or ev.get("type")):
            return None

    cache_cr = _coerce_int(
        _first(ev, _CACHE_CR_KEYS) or _nested_first(ev, ("usage",), _CACHE_CR_KEYS) or 0
    )
    cache_rd = _coerce_int(
        _first(ev, _CACHE_RD_KEYS) or _nested_first(ev, ("usage",), _CACHE_RD_KEYS) or 0
    )

    role = role_str
    if role in {"llm_call", "generation", "completion", "assistant"}:
        role = "agent"
    elif role in {"human", "input"}:
        role = "user"

    sess = _first(ev, _SESSION_KEYS)
    task = _first(ev, _TASK_KEYS)

    return TelemetryRecord(
        timestamp=_parse_ts(_first(ev, _TIMESTAMP_KEYS)),
        call_id=str(ev.get("id") or ev.get("call_id") or ev.get("uuid") or uuid.uuid4()),
        role=role,
        session_id=str(sess) if sess is not None else None,
        task_id=str(task) if task is not None else None,
        input_tokens=in_toks,
        output_tokens=out_toks,
        cache_creation_tokens=cache_cr,
        cache_read_tokens=cache_rd,
        duration_ms=_first(ev, _DURATION_KEYS),
        cost_usd=_first(ev, _COST_KEYS),
        prompt_preview=prompt,
        response_preview=response,
        tool_calls=tool_calls,
        context_window_size=_first(ev, _CTX_KEYS),
        model_id=_first(ev, _MODEL_KEYS),
        source_format="generic",
        raw=ev,
    )
