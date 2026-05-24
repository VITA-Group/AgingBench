"""
agingbench.telemetry — Trace-to-AgingCard mapping (behavioral-DAG pipeline).

Production deployment traces (Langfuse, LangSmith, OpenTelemetry,
custom JSONL logs, Claude Code session files) carry the per-call data
needed to compute aging metrics over time. Inference is behavioral-DAG-
based: tool calls, results, and outcomes form the structural signals;
regex over user text is the final fallback, not the default. This
namespace exposes:

  trace_to_card(...)        — backward-compatible stub: cost block + warnings
  trace_to_card_v11(...)    — full pipeline: adapter → scrub → session
                              → memory-reconstruct → 4-mechanism inference
                              + cross-session consistency (P5) →
                              dominant-mechanism arbitration →
                              AgingCard with trace_audit block + Lifespan-
                              Card surface (signature, repair, trace_regime)
  prepare_trace(source, output) — concatenate fragmented Claude Code
                              per-conversation .jsonl files into a single
                              timestamp-sorted trace

Plus the public surface for synthetic-probe-augmented mode:

  load_profile(name)         — load a shipped deployment-profile YAML
  list_profiles()            — discover available profiles
  list_supported_formats()   — discover supported trace formats
  list_injectable_scenarios() — scenarios runnable as synthetic probes
  load_probe_result(path)    — ingest a scenario AgingCard as probe data
  merge_probe_into_card(card, probe) — combine probe + telemetry results

See agingbench/telemetry/README.md for the design doc and usage examples.
"""
from .trace_to_card import (
    trace_to_card,
    TraceToCardResult,
    SUPPORTED_TRACE_FORMATS,
    trace_to_card_v11,
    TraceToCardV11Result,
)
from .schema import (
    TelemetryRecord,
    OutcomeEvent,
    MemoryEvent,
    ToolCall,
    CoverageReport,
    TraceAuditBlock,
)
from .adapters import list_supported_formats
from .profiles import load_profile, list_profiles, Profile
from .synthetic_probe import (
    list_injectable_scenarios,
    load_probe_result,
    merge_probe_into_card,
    ProbeSchedule,
    ProbeResult,
)
from .outcome_extractors import (
    list_extractors,
    run_extractor,
    extract_from_claude_session_flags,
    extract_from_record_patterns,
    extract_from_git_log,
)
from .prepare_trace import prepare_trace


__all__ = [
    # Legacy (v1.0)
    "trace_to_card", "TraceToCardResult", "SUPPORTED_TRACE_FORMATS",
    # v1.1 pipeline
    "trace_to_card_v11", "TraceToCardV11Result",
    # Schemas
    "TelemetryRecord", "OutcomeEvent", "MemoryEvent", "ToolCall",
    "CoverageReport", "TraceAuditBlock",
    # Adapter discovery
    "list_supported_formats",
    # Profiles
    "load_profile", "list_profiles", "Profile",
    # Synthetic probes
    "list_injectable_scenarios", "load_probe_result", "merge_probe_into_card",
    "ProbeSchedule", "ProbeResult",
    # Outcome extractors
    "list_extractors", "run_extractor",
    "extract_from_claude_session_flags",
    "extract_from_record_patterns",
    "extract_from_git_log",
    # Trace preprocessing (Claude Code fragmented-file concatenation)
    "prepare_trace",
]
