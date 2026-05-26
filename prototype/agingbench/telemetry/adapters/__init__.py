"""
agingbench/telemetry/adapters/ — Per-format trace normalisers.

Each adapter exposes `normalize(raw_event: dict) -> Optional[TelemetryRecord]`.
The registry lets `trace_to_card(..., trace_format=NAME)` dispatch.

Adding a new adapter: write `<name>.py` exposing `normalize`, then
register it below.
"""
from __future__ import annotations

from typing import Callable, Optional

from . import generic, claude_code, langfuse_v1, otlp_v1, openai_assistants, openhands
from ..schema import TelemetryRecord


# format name → normaliser
# NOTE: `langsmith` routes through the generic adapter (no dedicated
# normaliser). Kept in the registry for backward compat, but NOT
# advertised as a first-class format in the README until we ship a
# dedicated fixture + adapter-level test for it.
ADAPTERS: dict[str, Callable[[dict], Optional[TelemetryRecord]]] = {
    "generic":            generic.normalize,
    "claude_code":        claude_code.normalize,
    "langfuse":           langfuse_v1.normalize,
    "otlp":               otlp_v1.normalize,
    "langsmith":          generic.normalize,   # generic field-aliasing covers LangSmith run JSON
    "openai_assistants":  openai_assistants.normalize,
    "openhands":          openhands.normalize,
}


def list_supported_formats() -> list[str]:
    return sorted(ADAPTERS.keys())


def get_adapter(name: str) -> Callable[[dict], Optional[TelemetryRecord]]:
    if name not in ADAPTERS:
        raise ValueError(
            f"Unknown trace_format {name!r}. Supported: {list_supported_formats()}"
        )
    return ADAPTERS[name]
