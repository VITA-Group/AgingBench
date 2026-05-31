"""Shared tool-registry helpers for Tier-1 ReAct runners (S1, S2, S3, S6).

Two memory-access tools are exposed as alternative affordances:

  * ``build_default_tool_registry`` registers ``read_memory`` — full
    memory dump, no parameters.

  * ``build_lookup_tool_registry`` registers ``lookup_memory(phrase)`` —
    Ctrl+F-style context window around the first match. Mirrors the
    original ReAct paper's ``lookup[string]`` action (Yao et al. 2022).
    Window includes neighbour content so interference and revision
    tests are not trivialised by single-line filtering.

S5 (ReactFileAdapter) and S7/S8 (production CLI adapters) own their
memory and do not use these helpers.
"""
from __future__ import annotations

from typing import Callable

from .tools import ToolRegistry, ToolSpec


_LOOKUP_CONTEXT_BYTES = 200


def build_default_tool_registry(memory_reader: Callable[[], str]) -> ToolRegistry:
    """Registry containing ``read_memory`` — returns the full memory text."""
    registry = ToolRegistry()

    def _read_memory_fn(args: dict):
        mem = memory_reader() or ""
        if not mem:
            return {"result": "(no memory available)"}
        return {"result": mem}

    registry.register(ToolSpec(
        name="read_memory",
        version="1.0.0",
        description=(
            "Read your memory — returns the current memory contents. "
            "No parameters."
        ),
        parameters={"type": "object", "properties": {}},
        fn=_read_memory_fn,
    ))

    return registry


def build_lookup_tool_registry(memory_reader: Callable[[], str]) -> ToolRegistry:
    """Registry containing ``lookup_memory(phrase)`` — Ctrl+F-style retrieval.

    Returns ~200 chars on either side of the first case-insensitive match.
    "Phrase not found" when absent — no fallback to a memory dump (which
    would defeat the compression test).
    """
    registry = ToolRegistry()

    def _lookup_memory_fn(args: dict):
        phrase = (args.get("phrase") or "").strip()
        if not phrase:
            return {"result": (
                "Specify a phrase to look up. "
                "Usage: lookup_memory({\"phrase\": \"...\"})."
            )}
        mem = memory_reader() or ""
        if not mem:
            return {"result": "(no memory available)"}
        idx = mem.lower().find(phrase.lower())
        if idx < 0:
            return {"result": f"Phrase {phrase!r} not found in memory."}
        start = max(0, idx - _LOOKUP_CONTEXT_BYTES)
        end = min(len(mem), idx + len(phrase) + _LOOKUP_CONTEXT_BYTES)
        snippet = mem[start:end]
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(mem) else ""
        return {"result": f"{prefix}{snippet}{suffix}"}

    registry.register(ToolSpec(
        name="lookup_memory",
        version="1.0.0",
        description=(
            "Look up a phrase in your memory (Ctrl+F-style). Returns the "
            "surrounding context (~200 chars before and after the first "
            "match), or 'not found' if the phrase is absent."
        ),
        parameters={
            "type": "object",
            "properties": {
                "phrase": {
                    "type": "string",
                    "description": "The phrase or substring to look up.",
                },
            },
            "required": ["phrase"],
        },
        fn=_lookup_memory_fn,
    ))

    return registry
