"""
agingbench/runner/trace.py — Structured event logger (PDF §6.1.2, §6.2).

§6.1.2 Standard traces: emit events using OpenInference semantic attribute
conventions so that any backend can ingest them. Field names follow the
OpenInference specification (https://github.com/Arize-ai/openinference):

  gen_ai.system            — LLM provider identifier
  gen_ai.request.model     — model name as requested
  gen_ai.usage.input_tokens  — prompt token count
  gen_ai.usage.output_tokens — completion token count
  input.value / output.value — span input/output content

§6.2 Tracing: the full OpenTelemetry SDK is not a prototype dependency, but
the JSONL schema aligns to OTel span conventions (span_id, trace_id,
parent_span_id, timestamps) so traces can be imported into any OTel-compatible
backend without field renaming.

Trace event types
-----------------
  run_start    : begin of one SUT run
  llm_call     : one LLM inference call (compress step)
  probe_batch  : eval probes scored against current memory snapshot
  cycle_end    : summary of one compression cycle
  run_end      : final aging curve + statistics
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional


def _trace_preview_chars() -> int:
    """Max chars persisted for input.value/output.value previews.

    Default 0 (no truncation — full prompts/responses are written). Set the
    env var AGINGBENCH_TRACE_PREVIEW_CHARS to a positive integer N to cap each
    preview at N characters (useful when trace.jsonl size becomes a concern).
    A value <= 0 disables truncation.
    """
    try:
        n = int(os.environ.get("AGINGBENCH_TRACE_PREVIEW_CHARS", "0"))
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


class TraceLogger:
    """
    Append-only JSONL logger with OpenInference-compatible field names.

    Each call to log() writes one JSON line. The `span_id` field makes every
    event uniquely addressable. `parent_span_id` links child events (e.g. an
    llm_call inside a cycle) to their parent (cycle_start).
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", buffering=1)  # line-buffered
        self._run_span_id = _span_id()               # root span for this run

    def log(self, event: str, parent_span_id: Optional[str] = None, **fields: Any) -> str:
        """
        Write one event. Returns the new span_id so callers can use it as
        parent_span_id for child events.
        """
        sid = _span_id()
        record = {
            "event": event,
            "span_id": sid,
            "parent_span_id": parent_span_id or self._run_span_id,
            # OTel-compatible timestamps
            "timestamp": time.time(),
            **fields,
        }
        self._f.write(json.dumps(record) + "\n")
        return sid

    def log_llm_call(
        self,
        *,
        # parent_span_id is now optional: BaseLLM._log_llm_call calls this
        # helper from inside chat_with_usage() where the LLM does not know
        # which scenario span it's in. Orphan llm_call events still get the
        # cost aggregator to count them correctly.
        parent_span_id: Optional[str] = None,
        model: str = "",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        input_preview: str = "",
        output_preview: str = "",
        thought: str = "",
        cycle: int = -1,
        duration_ms: Optional[float] = None,
        cost_usd: Optional[float] = None,
    ) -> str:
        """
        Structured span for one LLM inference call.
        Field names follow OpenInference gen_ai.* conventions.

        ``thought`` is accepted for API symmetry with callers that extract it,
        but it is NOT persisted to the trace file. Reasoning traces can be
        tens of kB per call and the trace file is meant for final-answer +
        aging-signal analysis, not model introspection. If reasoning content
        is needed for debugging, attach a separate sink.

        ``duration_ms`` and ``cost_usd`` are optional. When passed they are
        recorded as `gen_ai.usage.duration_ms` and `gen_ai.usage.cost_usd`
        respectively. AgingCard's cost_and_efficiency block aggregates these
        across all llm_call events to populate `latency_ms_p50/p95` and
        `total_cost_usd`. Callers that don't know their per-call timing or
        pricing should omit these fields rather than pass zero — the
        aggregator skips missing values rather than treating them as zero.
        """
        del thought  # intentionally dropped
        attrs = {
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "input.value": (
                input_preview[:_trace_preview_chars()]
                if _trace_preview_chars() > 0 else input_preview
            ),
            "output.value": (
                output_preview[:_trace_preview_chars()]
                if _trace_preview_chars() > 0 else output_preview
            ),
        }
        if duration_ms is not None:
            attrs["gen_ai.usage.duration_ms"] = float(duration_ms)
        if cost_usd is not None:
            attrs["gen_ai.usage.cost_usd"] = float(cost_usd)
        return self.log(
            "llm_call",
            parent_span_id=parent_span_id,
            cycle=cycle,
            **attrs,
        )

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
