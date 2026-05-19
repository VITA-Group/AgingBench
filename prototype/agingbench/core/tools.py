"""
agingbench/baselines/tools.py — Portable tool contracts (PDF §6.1.3, §6.2).

§6.1.3 Portable tools: prefer MCP-style definitions so tool contracts are
explicit and versionable. Each ToolSpec carries a JSON Schema for its
parameters, a semantic version string, and a description. This makes tool
contract changes (the P5 schema-change scenario) directly observable in the
registry rather than implicit in Python dicts.

§6.2 Tool connectivity: MCP servers are the long-term target. For the
prototype, ToolSpec acts as the contract layer — the fn() callable can later
be replaced by an MCP client call without changing agent code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, Optional


class ToolFailure(Exception):
    """Raised when a tool call fails due to injected failure or real error."""
    pass


@dataclass
class FailureInjector:
    """
    Configurable failure injection for tool reliability testing (Family C).

    Parameters
    ----------
    failure_rate : float in [0, 1] — probability of failure per call.
    error_message : str — message returned on failure.
    start_after : int — only inject failures after this many calls (warmup).
    max_failures : int — stop injecting after this many failures (-1 = unlimited).
    """
    failure_rate: float = 0.0
    error_message: str = "Tool temporarily unavailable. Please try again later."
    start_after: int = 0
    max_failures: int = -1
    _call_count: int = field(default=0, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)

    def should_fail(self) -> bool:
        import random
        self._call_count += 1
        if self._call_count <= self.start_after:
            return False
        if 0 <= self.max_failures <= self._failure_count:
            return False
        if random.random() < self.failure_rate:
            self._failure_count += 1
            return True
        return False


@dataclass
class ToolSpec:
    """
    A versioned, schema-backed tool definition.

    Parameters
    ----------
    name        : Tool identifier used in ReAct Action lines.
    version     : Semantic version string. Changing this is the "life event"
                  signal in P5 (tool schema change mid-lifetime).
    description : Human-readable description injected into the agent prompt.
    parameters  : JSON Schema object describing accepted input fields.
                  Used for validation and as the MCP tool manifest.
    fn          : Callable that executes the tool given a dict of arguments.
                  Will be replaced by an MCP client call in a future iteration.
    failure_injector : Optional FailureInjector for reliability testing.
    """
    name: str
    version: str
    description: str
    parameters: dict = field(default_factory=dict)  # JSON Schema object
    fn: Callable[[dict], Any] = field(default=lambda _: None)
    failure_injector: Optional[FailureInjector] = field(default=None)

    def to_prompt_line(self) -> str:
        """One-line tool summary for agent system prompt."""
        return f"- {self.name} (v{self.version}): {self.description}"

    def to_manifest(self) -> dict:
        """MCP-compatible tool manifest dict."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "inputSchema": self.parameters,
        }

    def call(self, arguments: dict) -> Any:
        """Execute the tool, possibly injecting failures for reliability testing."""
        if self.failure_injector and self.failure_injector.should_fail():
            raise ToolFailure(
                f"[{self.name}] {self.failure_injector.error_message}"
            )
        return self.fn(arguments)


class ToolRegistry:
    """
    Ordered collection of ToolSpecs for one scenario.

    The registry tracks the active version of each tool, making schema changes
    (P5 life event) a first-class operation: update_tool() bumps the version
    and records the change event for the trace.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._change_log: list[dict] = []
        self._call_log: list[dict] = []  # tracks all call outcomes for reliability metrics

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def update_tool(self, name: str, new_spec: ToolSpec, reason: str = "") -> None:
        """
        Replace a tool with a new version (the P5 life event).
        Records the change for trace logging.
        """
        old = self._tools.get(name)
        self._tools[name] = new_spec
        self._change_log.append({
            "tool": name,
            "old_version": old.version if old else None,
            "new_version": new_spec.version,
            "reason": reason,
        })

    def prompt_descriptions(self) -> str:
        """All tools formatted for the agent system prompt."""
        return "\n".join(t.to_prompt_line() for t in self._tools.values()) or "(no tools)"

    def manifests(self) -> list[dict]:
        """All tool manifests (MCP format)."""
        return [t.to_manifest() for t in self._tools.values()]

    def call_tool(self, name: str, arguments: dict, session: int = 0) -> dict:
        """
        Call a tool and log the outcome for reliability metrics.

        Returns {"success": bool, "result": Any, "error": str | None}.
        """
        import time as _time
        spec = self._tools.get(name)
        if spec is None:
            entry = {"tool": name, "session": session, "success": False,
                     "error": f"Tool '{name}' not found", "timestamp": _time.time()}
            self._call_log.append(entry)
            return {"success": False, "result": None, "error": entry["error"]}
        try:
            result = spec.call(arguments)
            entry = {"tool": name, "session": session, "success": True,
                     "error": None, "timestamp": _time.time()}
            self._call_log.append(entry)
            return {"success": True, "result": result, "error": None}
        except ToolFailure as e:
            entry = {"tool": name, "session": session, "success": False,
                     "error": str(e), "timestamp": _time.time()}
            self._call_log.append(entry)
            return {"success": False, "result": None, "error": str(e)}

    def reliability_metrics(self) -> dict:
        """
        Compute tool reliability metrics from the call log.

        Returns dict with:
          - success_rate: fraction of calls that succeeded
          - failure_rate: fraction that failed (injected + real)
          - calls_per_tool: {tool_name: n_calls}
          - failures_per_tool: {tool_name: n_failures}
          - adaptation_rate: after a schema change, fraction of subsequent
            calls that use the new schema correctly (1.0 if no changes)
          - mean_recovery_calls: avg calls after failure before next success
        """
        if not self._call_log:
            return {"success_rate": 1.0, "failure_rate": 0.0,
                    "calls_per_tool": {}, "failures_per_tool": {},
                    "adaptation_rate": 1.0, "mean_recovery_calls": 0.0}

        total = len(self._call_log)
        successes = sum(1 for e in self._call_log if e["success"])
        calls_per_tool: dict[str, int] = {}
        failures_per_tool: dict[str, int] = {}
        for e in self._call_log:
            calls_per_tool[e["tool"]] = calls_per_tool.get(e["tool"], 0) + 1
            if not e["success"]:
                failures_per_tool[e["tool"]] = failures_per_tool.get(e["tool"], 0) + 1

        # Recovery: how many calls after a failure before next success?
        recovery_gaps = []
        gap = 0
        in_failure = False
        for e in self._call_log:
            if not e["success"]:
                in_failure = True
                gap = 0
            elif in_failure:
                gap += 1
                recovery_gaps.append(gap)
                in_failure = False

        # Adaptation: after schema changes, are subsequent calls successful?
        adaptation_calls = 0
        adaptation_successes = 0
        for change in self._change_log:
            tool_name = change["tool"]
            # Count calls to this tool after the change
            found_change = False
            for e in self._call_log:
                if not found_change and e["tool"] == tool_name:
                    continue  # before change
                if e["tool"] == tool_name:
                    found_change = True
                    adaptation_calls += 1
                    if e["success"]:
                        adaptation_successes += 1

        return {
            "success_rate": round(successes / total, 4),
            "failure_rate": round(1 - successes / total, 4),
            "calls_per_tool": calls_per_tool,
            "failures_per_tool": failures_per_tool,
            "adaptation_rate": round(
                adaptation_successes / adaptation_calls, 4
            ) if adaptation_calls > 0 else 1.0,
            "mean_recovery_calls": round(
                sum(recovery_gaps) / len(recovery_gaps), 2
            ) if recovery_gaps else 0.0,
        }

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
