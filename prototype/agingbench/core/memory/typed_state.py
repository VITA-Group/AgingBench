"""
agingbench/core/memory/typed_state.py — Typed-state overlay for accumulator tracking.

Wraps an existing memory policy with a small JSON state object for accumulators
(running totals like dining budget = initial + sum(deltas)). On write(), parses
sentinel-marked deltas of the form [ACCUM:name:delta] (and [ACCUM_INIT:name:initial]
for initialization) and applies them to internal state. On read(), prepends a
JSON state header to the inner policy's output so the model has the up-to-date
accumulator value alongside the compressed narrative memory.

Used by:
  * the always-on typed-state intervention (always_enabled=True)
  * the runtime controller, which activates the overlay mid-run via set_enabled(True)

When enabled=False, the overlay is a transparent pass-through to the inner policy.
Sentinels are stripped from the text passed to the inner policy in either case so
the compaction prompt does not see them.
"""

import json
import re
from typing import Optional
from .base import MemoryPolicy

# Sentinel formats:
#   [ACCUM_INIT:name:initial_value]  -> sets initial value, also resets remaining
#   [ACCUM:name:delta]               -> applies delta to remaining
_INIT_RE = re.compile(r"\[ACCUM_INIT:(?P<name>[a-z_]+):(?P<initial>-?\d+(?:\.\d+)?)\]")
_DELTA_RE = re.compile(r"\[ACCUM:(?P<name>[a-z_]+):(?P<delta>-?\d+(?:\.\d+)?)\]")
_ANY_SENTINEL_RE = re.compile(r"\[ACCUM(?:_INIT)?:[a-z_]+:-?\d+(?:\.\d+)?\]")


class TypedStateOverlay(MemoryPolicy):
    """Wraps an inner memory policy with a typed JSON state for accumulators.

    Public attributes:
        inner: wrapped MemoryPolicy
        enabled: whether the overlay is active. When False, behaves as a
                 transparent pass-through (sentinels are still stripped).
        state: dict {name: {"initial": float, "remaining": float}}
        write_log: list of (session_inferred, name, delta_or_init, value) tuples
                   for trace-level inspection.
    """

    def __init__(self, inner: MemoryPolicy, enabled: bool = True):
        self.inner = inner
        self.enabled = enabled
        self.state: dict[str, dict[str, float]] = {}
        self.write_log: list[dict] = []

    def set_enabled(self, enabled: bool) -> None:
        """Toggle overlay activation. Used by the runtime controller mid-run."""
        self.enabled = enabled

    def read(self, query: Optional[str] = None) -> str:
        inner_text = self.inner.read(query) or ""
        if not self.enabled or not self.state:
            return inner_text
        header_lines = ["[Typed state — current accumulator values]"]
        for name in sorted(self.state.keys()):
            vals = self.state[name]
            initial = vals.get("initial", 0.0)
            remaining = vals.get("remaining", initial)
            header_lines.append(
                f"  - {name}: remaining = {remaining:.0f} "
                f"(initial: {initial:.0f}, applied deltas sum: {remaining - initial:+.0f})"
            )
        header = "\n".join(header_lines) + "\n\n"
        return header + inner_text

    def write(self, new_content: str, llm=None) -> None:
        # Parse INIT sentinels first (they reset state for that accumulator).
        for m in _INIT_RE.finditer(new_content):
            name = m.group("name")
            initial = float(m.group("initial"))
            self.state[name] = {"initial": initial, "remaining": initial}
            self.write_log.append({
                "kind": "init", "name": name, "value": initial,
                "remaining_after": initial,
            })
        # Then apply DELTA sentinels.
        for m in _DELTA_RE.finditer(new_content):
            name = m.group("name")
            delta = float(m.group("delta"))
            if name not in self.state:
                # Delta without prior INIT: register with initial=0
                self.state[name] = {"initial": 0.0, "remaining": 0.0}
            self.state[name]["remaining"] += delta
            self.write_log.append({
                "kind": "delta", "name": name, "value": delta,
                "remaining_after": self.state[name]["remaining"],
            })

        # Strip sentinels before passing to inner policy: compaction prompt
        # should see clean prose so it doesn't try to summarize the markup.
        cleaned = _ANY_SENTINEL_RE.sub("", new_content)
        # Squeeze repeated whitespace introduced by sentinel removal.
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        self.inner.write(cleaned, llm=llm)

    def reset(self) -> None:
        self.inner.reset()
        self.state = {}
        self.write_log = []

    def snapshot(self) -> str:
        inner_snap = self.inner.snapshot() if hasattr(self.inner, "snapshot") else self.inner.read()
        if self.state:
            return (
                "=== TYPED STATE ===\n"
                + json.dumps(self.state, indent=2)
                + "\n\n=== INNER POLICY ===\n"
                + (inner_snap or "")
            )
        return inner_snap or ""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return self.inner.retrieve(query, top_k)

    def entry_count(self) -> int:
        return self.inner.entry_count()

    @property
    def last_input_tokens(self) -> int:
        return getattr(self.inner, "last_input_tokens", 0)

    @property
    def last_output_tokens(self) -> int:
        return getattr(self.inner, "last_output_tokens", 0)

    @property
    def n_writes(self) -> int:
        return getattr(self.inner, "n_writes", 0)

    # ---- Maintenance hooks (delegate to inner) ----
    def maintenance_flush_history(self) -> None:
        if hasattr(self.inner, "maintenance_flush_history"):
            self.inner.maintenance_flush_history()
        # NOTE: typed-state survives memory flush by design — it is structured
        # state, not narrative. This matches advisor's "typed state for revision-
        # heavy variables" framing. If the runtime controller wants to flush
        # typed state too, call reset_state() explicitly.

    def maintenance_recompact(self, llm=None) -> None:
        if hasattr(self.inner, "maintenance_recompact"):
            self.inner.maintenance_recompact(llm=llm)

    def maintenance_partial_reset(self, ratio: float = 0.5) -> None:
        if hasattr(self.inner, "maintenance_partial_reset"):
            self.inner.maintenance_partial_reset(ratio)

    def reset_state(self) -> None:
        """Clear typed state without resetting inner policy (rare; used by tests)."""
        self.state = {}
        self.write_log = []


def build_typed_state_overlay(
    inner_cfg: dict,
    enabled: bool = True,
    project_root=None,
) -> TypedStateOverlay:
    """Factory for use with memory_policy.type=='custom':

    memory_policy:
      type: custom
      class: agingbench.core.memory.typed_state:build_typed_state_overlay
      inner_cfg:
        type: summarize_store
        compaction_prompt: experiments/prompts/compact_lossy.txt
        word_budget: 300
      enabled: true
    """
    from .base import build_memory_policy
    inner = build_memory_policy(inner_cfg, project_root=project_root)
    return TypedStateOverlay(inner=inner, enabled=enabled)
