"""
agingbench/core/maintenance.py — Maintenance event injection for aging measurement.

Maintenance events are operational shocks (recompaction, partial reset, reindex,
budget reduction) injected at controlled session boundaries. They use duck-typed
dispatch — only memory policies that implement the matching `maintenance_*` method
are affected. The MemoryPolicy ABC is NOT modified.

Configuration lives in SUT YAML under a `maintenance:` key:

    maintenance:
      shocks:
        - type: partial_reset
          session: 5
          ratio: 0.3
        - type: reindex
          session: 8
        - type: reduce_budget
          session: 10
          new_budget: 150
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MaintenanceEvent:
    """A single maintenance shock to apply at a specific session."""
    event_type: str   # "memory_compaction", "partial_reset", "reindex", "reduce_budget"
    session: int      # session index at which to apply
    params: dict = field(default_factory=dict)  # type-specific params (ratio, new_budget, etc.)

    def apply(self, memory_policy, llm=None) -> str:
        """Apply this shock to a memory policy via duck-typed dispatch.

        Calls `memory_policy.maintenance_{event_type}(**params)`. If the
        method accepts an `llm` keyword (introspection via signature), the
        runner's llm is forwarded so the shock can call the model (needed
        for summarize_store's recompact which re-runs the compaction prompt).

        Returns a string that is either the `event_type` on success or
        ``"UNSUPPORTED:{event_type}"`` when the policy does not implement
        the matching method. Callers (runner) must record the return value
        so downstream metrics can distinguish real shocks from silent no-ops.
        Previously the no-op path only emitted `warnings.warn`, which was
        invisible at analysis time and produced fake \u0394shock signals.
        """
        method_name = f"maintenance_{self.event_type}"
        method = getattr(memory_policy, method_name, None)
        if method is None:
            msg = (
                f"Memory policy {type(memory_policy).__name__} does not support "
                f"maintenance event '{self.event_type}' (no method '{method_name}'). "
                f"Shock is a no-op; \u0394shock data from this run will be invalid."
            )
            warnings.warn(msg)
            # Also print to stderr so it's visible in run logs, not just warning filters.
            import sys
            print(f"[maintenance] WARNING: {msg}", file=sys.stderr, flush=True)
            return f"UNSUPPORTED:{self.event_type}"
        # Forward llm when the method signature accepts it (e.g. summarize_store recompact).
        import inspect
        sig = inspect.signature(method)
        kwargs = dict(self.params)
        if "llm" in sig.parameters:
            kwargs["llm"] = llm
        method(**kwargs)
        return self.event_type


def load_maintenance_config(sut_cfg: dict) -> list[MaintenanceEvent]:
    """Parse maintenance events from SUT YAML config.

    Expected structure:
        maintenance:
          shocks:
            - type: partial_reset
              session: 5
              ratio: 0.3

    Returns empty list if not configured (backward compatible).
    """
    maintenance = sut_cfg.get("maintenance", {})
    shocks = maintenance.get("shocks", [])
    events = []
    for shock in shocks:
        event_type = shock.get("type", "")
        session = shock.get("session", -1)
        if not event_type or session < 0:
            continue
        params = {k: v for k, v in shock.items() if k not in ("type", "session")}
        events.append(MaintenanceEvent(event_type=event_type, session=session, params=params))
    return events
