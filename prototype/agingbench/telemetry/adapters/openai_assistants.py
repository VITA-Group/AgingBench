"""
adapters/openai_assistants.py — OpenAI Assistants API → TelemetryRecord.

The Assistants API exposes four object types over its REST surface:

  - thread.message    → one user or assistant turn
  - thread.run        → one execution of an assistant on a thread
                        (carries `usage` with prompt/completion tokens
                        and `model`)
  - thread.run.step   → one step inside a run (`type` = "message_creation"
                        or "tool_calls"; the latter exposes the tool
                        invocations the assistant chose to make)
  - thread            → thread metadata; skipped (no content)

Threads ARE sessions — `thread_id` becomes `session_id`. Created_at is
an epoch-seconds integer per the API contract.

Reference: https://platform.openai.com/docs/api-reference/messages
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

from ..schema import TelemetryRecord, ToolCall
from .generic import _parse_ts


def _content_text(blocks) -> Optional[str]:
    """Assistants messages are arrays of content parts: text / image_file /
    image_url. Concatenate the text parts."""
    if isinstance(blocks, str):
        return blocks[:1000]
    if not isinstance(blocks, list):
        return None
    parts = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            t = b.get("text")
            if isinstance(t, dict):
                v = t.get("value")
                if isinstance(v, str):
                    parts.append(v)
            elif isinstance(t, str):
                parts.append(t)
    return " ".join(parts)[:1000] if parts else None


def _extract_tool_calls(step_details: dict) -> list[ToolCall]:
    out = []
    for tc in step_details.get("tool_calls", []) or []:
        if not isinstance(tc, dict):
            continue
        kind = tc.get("type", "function")
        if kind == "function":
            fn = tc.get("function") or {}
            name = fn.get("name") or "unknown"
            raw_args = fn.get("arguments")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                if not isinstance(args, dict):
                    args = {"_raw": args}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            out.append(ToolCall(name=str(name), args=args,
                                result_summary=str(fn.get("output", ""))[:200] or None))
        elif kind == "code_interpreter":
            ci = tc.get("code_interpreter") or {}
            out.append(ToolCall(
                name="code_interpreter",
                args={"input": str(ci.get("input", ""))[:500]},
                result_summary=str(ci.get("outputs", ""))[:200] or None,
            ))
        elif kind == "file_search":
            out.append(ToolCall(name="file_search", args={}, result_summary=None))
        else:
            out.append(ToolCall(name=str(kind), args={}))
    return out


def normalize(ev: dict) -> Optional[TelemetryRecord]:
    if not isinstance(ev, dict):
        return None
    obj = ev.get("object", "")
    thread_id = ev.get("thread_id")
    ts = _parse_ts(ev.get("created_at") or ev.get("started_at") or ev.get("timestamp"))

    if obj == "thread.message":
        role_raw = (ev.get("role") or "user").lower()
        role = "user" if role_raw == "user" else "agent"
        text = _content_text(ev.get("content"))
        if not text and not ev.get("attachments"):
            return None
        return TelemetryRecord(
            timestamp=ts,
            call_id=str(ev.get("id") or uuid.uuid4()),
            role=role,
            session_id=str(thread_id) if thread_id else None,
            task_id=str(ev.get("run_id")) if ev.get("run_id") else None,
            prompt_preview=text if role == "user" else None,
            response_preview=text if role == "agent" else None,
            model_id=ev.get("assistant_id"),
            source_format="openai_assistants",
            raw=ev,
        )

    if obj == "thread.run":
        usage = ev.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
        # Latency = completed_at - started_at if both present (epoch seconds → ms)
        duration_ms = None
        if ev.get("started_at") and ev.get("completed_at"):
            try:
                duration_ms = (float(ev["completed_at"]) - float(ev["started_at"])) * 1000.0
            except (TypeError, ValueError):
                pass
        return TelemetryRecord(
            timestamp=ts,
            call_id=str(ev.get("id") or uuid.uuid4()),
            role="agent",
            session_id=str(thread_id) if thread_id else None,
            task_id=str(ev.get("id")) if ev.get("id") else None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            duration_ms=duration_ms,
            model_id=ev.get("model"),
            source_format="openai_assistants",
            raw=ev,
        )

    if obj == "thread.run.step":
        step_type = ev.get("type", "")
        usage = ev.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
        details = ev.get("step_details") or {}
        tool_calls = _extract_tool_calls(details) if step_type == "tool_calls" else []
        if not tool_calls and in_tok == 0 and out_tok == 0:
            return None
        return TelemetryRecord(
            timestamp=ts,
            call_id=str(ev.get("id") or uuid.uuid4()),
            role="agent",
            session_id=str(thread_id) if thread_id else None,
            task_id=str(ev.get("run_id")) if ev.get("run_id") else None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            tool_calls=tool_calls,
            source_format="openai_assistants",
            raw=ev,
        )

    return None
