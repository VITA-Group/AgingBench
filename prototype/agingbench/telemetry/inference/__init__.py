"""
agingbench/telemetry/inference/ — Per-mechanism inference modules.

Each module exposes one top-level function that takes
`list[list[TelemetryRecord]]` (sessions) plus profile/config kwargs and
returns a JSON-serialisable dict that becomes a sub-block of the
TraceAuditBlock.

All four follow the same shape: `(sessions, **kwargs) -> dict`.
"""
from .compression import infer_compression
from .interference import infer_interference
from .revision import infer_revision
from .maintenance import infer_maintenance


__all__ = ["infer_compression", "infer_interference", "infer_revision", "infer_maintenance"]
