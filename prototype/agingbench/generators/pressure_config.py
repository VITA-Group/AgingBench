"""
agingbench/generators/pressure_config.py — Configurable aging pressure.

Controls how much dependency, versioning, interference, and volume pressure
the generator applies. Presets map to target agent capability tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PressureConfig:
    """
    Controls aging pressure in programmatic scenario generation.

    Parameters
    ----------
    tokens_per_session : int
        Target volume of environment data per session (in tokens).
        Higher = more context pressure for large-context agents.
        500 for small models (8K context), 5000 for frontier (200K).

    dependency_density : float in [0, 1]
        Fraction of tasks (after warmup) that have cross-session dependencies.
        0.0 = current behavior (all standalone), 1.0 = every task references prior.

    update_rate : float in [0, 1]
        Fraction of facts that get updated (superseded) at least once.
        Creates version chains that test whether agent tracks the latest value.

    max_chain_depth : int
        Maximum number of hops in a dependency chain.
        depth=1: single prior fact. depth=3: requires integrating 3 sessions.

    n_confusable_pairs : int
        Number of cross-domain confusable entity groups to create.
        Each group shares a term (e.g., "budget") across different domains
        with different values, creating retrieval interference.

    confusable_start_session : int
        Session index after which interference entities are introduced.
        Earlier sessions establish clean facts before adding confusion.

    warmup_sessions : int
        Number of initial sessions with standalone tasks (no dependencies).
        Allows the agent to build a fact base before testing recall.
    """
    tokens_per_session: int = 2000
    dependency_density: float = 0.5
    update_rate: float = 0.15
    max_chain_depth: int = 3
    n_confusable_pairs: int = 3
    confusable_start_session: int = 5
    warmup_sessions: int = 3
    forget_rate: float = 0.0  # fraction of facts to invalidate per session (0.0 = disabled)
    # High-similarity confusables: near-twin entities (same base, minimal
    # qualifier diff) with CLOSE values (±~5%), instead of the default
    # surface-word-only pairs with order-of-magnitude-apart values. The default
    # pairs can't induce mis-binding (trivially distinguishable); this mode
    # produces genuinely fragile bindings to test the confusion channel.
    confusable_high_similarity: bool = False
    # Similar-NAME confusables (the figure's "two Johns" case): near-identical
    # entity names (John Smith / John Smyth) with DISTINCT attribute values.
    # The ambiguity is in the retrieval KEY (the name), not the value — the
    # agent may grab the wrong record. Tests identity-confusion specifically.
    confusable_similar_names: bool = False
    # Lags (in sessions, after injection) at which to re-probe each confusable
    # pair. Probing the same pair at increasing lags as the append_only store
    # grows gives a context-DENSITY gradient for interference. None → single
    # probe at +2 (backward compatible).
    confusable_probe_lags: Optional[list] = None

    def __post_init__(self) -> None:
        """Validate cross-knob compatibility.

        The interference / version / chain probe path runs through
        ``DependencyMixin.build_dependency_task``, which fires only when
        ``dependency_density > 0``. Setting ``dependency_density = 0`` while
        leaving ``n_confusable_pairs``, ``update_rate``, or ``max_chain_depth``
        nonzero produces a silent metric-vacuity trap: the corresponding facts
        get registered (and interference text injected into transcripts), but
        no probes are emitted to test them, so ``interference_resistance`` /
        ``version_accuracy`` / ``chain_recall`` return their vacuous defaults.
        Issue a warning so users notice when isolating one mechanism axis.
        """
        if self.dependency_density == 0.0:
            silent_axes: list[str] = []
            if self.n_confusable_pairs > 0:
                silent_axes.append(
                    f"n_confusable_pairs={self.n_confusable_pairs}"
                )
            if self.update_rate > 0.0:
                silent_axes.append(f"update_rate={self.update_rate}")
            if self.max_chain_depth > 1:
                silent_axes.append(f"max_chain_depth={self.max_chain_depth}")
            if silent_axes:
                import warnings
                warnings.warn(
                    "PressureConfig: dependency_density=0 disables the "
                    "build_dependency_task probe path, so the following "
                    "knobs will register facts but emit no probes "
                    f"(metrics will be vacuous): {', '.join(silent_axes)}. "
                    "Set dependency_density > 0 (e.g., 0.5) to score "
                    "interference_resistance, version_accuracy, and "
                    "chain_recall.",
                    UserWarning,
                    stacklevel=2,
                )

    @classmethod
    def none(cls) -> PressureConfig:
        """No dependency pressure — reproduces current generator behavior."""
        return cls(
            tokens_per_session=1000,
            dependency_density=0.0,
            update_rate=0.0,
            max_chain_depth=1,
            n_confusable_pairs=0,
            warmup_sessions=999,  # effectively infinite
            forget_rate=0.0,
        )

    @classmethod
    def light(cls) -> PressureConfig:
        """For small models (Llama-3-8B, 8K context)."""
        return cls(
            tokens_per_session=500,
            dependency_density=0.3,
            update_rate=0.1,
            max_chain_depth=2,
            n_confusable_pairs=1,
            confusable_start_session=5,
            warmup_sessions=3,
            forget_rate=0.05,
        )

    @classmethod
    def medium(cls) -> PressureConfig:
        """For mid-size models (Gemma-27B, 32K-128K context)."""
        return cls(
            tokens_per_session=2000,
            dependency_density=0.5,
            update_rate=0.2,
            max_chain_depth=3,
            n_confusable_pairs=3,
            confusable_start_session=5,
            warmup_sessions=3,
            forget_rate=0.1,
        )

    @classmethod
    def heavy(cls) -> PressureConfig:
        """For frontier models (Claude Sonnet/Opus, GPT-4o, 200K context)."""
        return cls(
            tokens_per_session=5000,
            dependency_density=0.7,
            update_rate=0.3,
            max_chain_depth=4,
            n_confusable_pairs=12,
            confusable_start_session=8,
            warmup_sessions=5,
            forget_rate=0.15,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PressureConfig:
        """Load from a YAML config file."""
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Serialize for inclusion in metrics.json."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
