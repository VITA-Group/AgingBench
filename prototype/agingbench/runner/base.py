"""
agingbench/runner/base.py — BaseRunner abstract base class.

All scenario runners should subclass BaseRunner so that the CLI and
test harness can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .trace import TraceLogger
from ..core.memory.base import MemoryPolicy
from ..metrics.aging import AgingCurve


class RunResult:
    """
    Standardized return value from BaseRunner.run().

    Every runner must populate at least ``primary_curve`` and
    ``session_results``.  Additional curves and raw data are optional.
    """

    def __init__(
        self,
        primary_curve: AgingCurve,
        session_results: list[dict],
        secondary_curves: dict[str, AgingCurve] | None = None,
        raw: dict[str, list] | None = None,
    ):
        self.primary_curve = primary_curve
        self.session_results = session_results
        self.secondary_curves = secondary_curves or {}
        self.raw = raw or {}

    # Convenience accessors for backward compatibility
    def __getitem__(self, key: str) -> Any:
        if key == "session_results":
            return self.session_results
        if key in self.secondary_curves:
            return self.secondary_curves[key]
        if key in self.raw:
            return self.raw[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class BaseRunner(ABC):
    """
    Abstract base class for scenario runners.

    Subclasses must implement:
        SCENARIO_ID : str          — unique scenario identifier
        run(n_sessions, seed)      — execute the session loop, return RunResult

    The constructor signature is deliberately not enforced — each scenario
    needs different dependencies (LLM vs. ClaudeCodeAdapter, tools, etc.).
    But the run() contract is uniform so the CLI can dispatch generically.
    """

    SCENARIO_ID: str = ""

    @abstractmethod
    def run(self, n_sessions: int = 10, seed: int = 42) -> RunResult:
        """
        Execute the scenario loop.

        Parameters
        ----------
        n_sessions : number of sessions (or cycles) to run.
        seed       : random seed for reproducibility.

        Returns
        -------
        RunResult with at least primary_curve and session_results populated.
        """
