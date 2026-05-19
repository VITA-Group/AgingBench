"""
adapters/claude_code.py — Claude Code session-JSONL → TelemetryRecord.

Claude Code writes structured session files to
~/.claude/projects/<project-hash>/<session-id>.jsonl by default.
Each line is one event: user turn, assistant turn, tool_use, tool_result.

This adapter is the plug-and-play entry point: pointing AgingBench at
~/.claude/projects/ produces a Tier-2 AgingCard with no instrumentation
required.
"""
from __future__ import annotations

import uuid
from typing import Optional

from ..schema import TelemetryRecord, ToolCall
from .generic import _parse_ts, _summarise_text


_ROLE_MAP = {
    "user":       "user",
    "assistant":  "agent",
    "system":     "system",
    "tool_use":   "tool",
    "tool_result": "tool",
}


def normalize(ev: dict) -> Optional[TelemetryRecord]:
    """Map one Claude Code session-JSONL event to a TelemetryRecord."""
    if not isinstance(ev, dict):
        return None

    msg_type = ev.get("type")
    if msg_type not in _ROLE_MAP:
        # Could be a session-summary or metadata line — skip.
        return None

    msg = ev.get("message") or {}
    usage = msg.get("usage") or {}
    content = msg.get("content")

    # Extract content text + tool calls.
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif block_type == "tool_use":
                tool_calls.append(ToolCall(
                    name=str(block.get("name", "unknown")),
                    args=block.get("input", {}) if isinstance(block.get("input"), dict) else {},
                    result_summary=None,
                ))
            elif block_type == "tool_result":
                # Results are linked to a prior tool_use call; surface them
                # as a degenerate ToolCall with name='_result' for the
                # downstream tool-distribution stats. (No-op if downstream
                # filters synthetic names.)
                pass
    elif isinstance(content, str):
        text_parts.append(content)

    text = " ".join(text_parts)[:1000] if text_parts else None
    role = _ROLE_MAP[msg_type]

    return TelemetryRecord(
        timestamp=_parse_ts(ev.get("timestamp")),
        call_id=str(ev.get("uuid") or uuid.uuid4()),
        role=role,
        session_id=ev.get("sessionId") or ev.get("session_id"),
        task_id=ev.get("parentUuid") or ev.get("parent_id"),
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        duration_ms=ev.get("duration_ms"),
        cost_usd=None,  # Claude Code doesn't report cost; downstream computes from tokens
        prompt_preview=text if role == "user" else None,
        response_preview=text if role == "agent" else None,
        tool_calls=tool_calls,
        context_window_size=None,  # not directly exposed; can infer from prompt_tokens
        model_id=msg.get("model"),
        source_format="claude_code",
        raw=ev,
    )
