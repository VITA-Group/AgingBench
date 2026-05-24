"""
agingbench/telemetry/schema.py — Canonical internal data shapes.

All trace-format adapters normalize their input into TelemetryRecord.
All inference modules consume TelemetryRecord lists. The downstream
TraceAuditBlock is what gets attached to the AgingCard alongside the
existing scenario-derived `mechanism_metrics` block.

Keeping these as plain dataclasses (no pydantic) for zero extra deps
and trivial JSON serialisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


@dataclass
class ToolCall:
    """One tool invocation inside an LLM call."""
    name: str
    args: dict = field(default_factory=dict)
    result_summary: Optional[str] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None


@dataclass
class TelemetryRecord:
    """Canonical internal representation of one trace event.

    Each adapter's `normalize()` returns one of these per ingested span.
    Tier 0 fields (timestamp, tokens) are required-ish; everything else
    is best-effort and may be None depending on what the source format
    exposed.
    """
    timestamp:             datetime
    call_id:               str
    role:                  str = "agent"           # 'user'|'agent'|'system'|'tool'
    session_id:            Optional[str] = None
    task_id:               Optional[str] = None
    user_id_hash:          Optional[str] = None

    # Token usage
    input_tokens:          int = 0
    output_tokens:         int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens:     int = 0

    # Performance
    duration_ms:           Optional[float] = None
    cost_usd:              Optional[float] = None

    # Content (post privacy-scrub)
    prompt_preview:        Optional[str] = None
    response_preview:      Optional[str] = None

    # Tool surface
    tool_calls:            list[ToolCall] = field(default_factory=list)

    # Memory hints (provider-specific; may be None)
    context_window_size:   Optional[int] = None
    model_id:              Optional[str] = None

    # Linked outcome (set later by outcome_attachment.py)
    outcome:               Optional["OutcomeEvent"] = None

    # Provenance
    source_format:         str = "generic"
    raw:                   dict = field(default_factory=dict)


@dataclass
class OutcomeEvent:
    """A correctness signal for a (session, task) pair.

    Either ingested from a separate outcomes JSONL, derived natively by
    a profile-specific extractor (e.g., `git revert` → fail), or
    produced by the synthetic-probe runner.
    """
    session_id:    str
    task_id:       str
    outcome:       str               # 'success'|'fail'|'user_rejected'|'abandoned'|'revision_fail'
    timestamp:     Optional[datetime] = None
    user_signal:   Optional[str] = None   # 'thumbs_up'|'thumbs_down'|'neutral'
    gold_label:    Optional[dict] = None
    correction_of: Optional[str] = None
    source:        str = "user"           # 'user'|'platform_native'|'synthetic_probe'


@dataclass
class MemoryEvent:
    """A detected (or inferred) memory-state transition."""
    session_idx: int
    timestamp:   datetime
    type:        str               # 'model_swap'|'context_reset'|'system_change'|'compression_spike'|'explicit_command'
    confidence:  float = 1.0       # 0..1
    detail:      dict = field(default_factory=dict)


@dataclass
class CoverageReport:
    """Per-mechanism stress coverage for honest reporting.

    A mechanism whose test never fired in the workload should NOT be
    cited as 'the agent passed' — `verdict` makes that explicit.
    """
    n_observations: int
    coverage_fraction: Optional[float] = None
    verdict: str = "no_test_fired"   # 'strong'|'adequate'|'weak'|'underpowered'|'no_test_fired'

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TraceAuditBlock:
    """The telemetry-derived sub-block attached to an AgingCard.

    Parallel to the scenario-derived `mechanism_metrics` block. Always
    carries `derived_from: 'telemetry'` so consumers can distinguish the
    measurement source.
    """
    derived_from:           str = "telemetry"
    deployment_type:        str = "generic"
    n_sessions_detected:    int = 0
    n_outcome_events:       int = 0
    session_detection_mode: str = "idle_gap"   # 'explicit_id'|'idle_gap'|'user_id_split'
    outcome_rules_hash:     Optional[str] = None

    # Per-mechanism sub-dicts — populated by inference modules.
    compression:    dict = field(default_factory=dict)
    interference:   dict = field(default_factory=dict)
    revision:       dict = field(default_factory=dict)
    maintenance:    dict = field(default_factory=dict)
    headline:       dict = field(default_factory=dict)
    # Cross-session task-consistency probe (P5). Carries
    # behavior_drift_at_repeat (outcome-free headline source), the
    # repeat-task aggregate, and a per-session drop trajectory.
    consistency:    dict = field(default_factory=dict)
    # Trace-regime disclosure: how complete is the input trace?
    # Keys: tool_using:bool, multi_session:bool, outcomes:str
    # ('linked'|'absent'), n_sessions:int, adapter:str.
    trace_regime:   dict = field(default_factory=dict)
    # Card render fields (atlas's Agent Lifespan Card surface).
    # Populated by trace_to_card from the dominant-mechanism selector
    # + card_lookups.py. Empty when no dominant mechanism passes the gate.
    signature:      Optional[str] = None
    repair:         Optional[str] = None


def telemetry_record_to_dict(r: TelemetryRecord) -> dict:
    """JSON-safe serialisation (datetime → ISO string)."""
    d = asdict(r)
    d["timestamp"] = r.timestamp.isoformat() if r.timestamp else None
    if r.outcome and r.outcome.timestamp:
        d["outcome"]["timestamp"] = r.outcome.timestamp.isoformat()
    return d
