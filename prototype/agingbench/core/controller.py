"""
agingbench/core/controller.py — Threshold-triggered runtime controller.

A minimal between-session controller that observes per-session aging metrics
(accumulator_error, constraint_precision, lag_recall) and dispatches corrective
actions (promote_to_typed_state, switch_compaction_policy) when thresholds
are crossed.

A concrete realization of runtime aging mitigation via closed-loop policy
switching: the benchmark's mechanism-level metrics serve as observation
inputs and the memory architecture's hooks serve as control outputs.

Triggers are one-shot: once fired, the action persists for the remainder of
the run. This is a deliberate simplification appropriate to the demonstration
scope; richer policies (e.g., learned controllers, multi-action sequences)
are future work.

Used as an additive hook in scenario runners; runners that don't set
self.controller default to no-op (existing reproducibility preserved).
"""

from typing import Callable, Optional


class ThresholdController:
    """Minimal threshold-based controller for runtime aging mitigation.

    Available triggers (each can be enabled independently):
      * promote_to_typed_state: fires when last-session accumulator_error
        exceeds theta_acc.
      * switch_compaction_policy: fires when last-session constraint_precision
        drops below theta_prec.

    Args:
        theta_acc: accumulator-error threshold (default 50.0; on S2 the
            non-controller mean error is ~150-227, so 50 is conservative).
        theta_prec: constraint-precision threshold (default 0.5; on S2 the
            lossy baseline drops below 0.5 by session 5-6 for several models).
        enable_typed_state_trigger: master switch for the first action.
        enable_careful_switch_trigger: master switch for the second action.
        warmup_sessions: number of initial sessions to skip before any
            trigger can fire (default 1; baseline aging needs at least one
            data point to produce a meaningful metric).
    """

    def __init__(
        self,
        theta_acc: float = 50.0,
        theta_prec: float = 0.5,
        enable_typed_state_trigger: bool = True,
        enable_careful_switch_trigger: bool = True,
        warmup_sessions: int = 1,
    ):
        self.theta_acc = theta_acc
        self.theta_prec = theta_prec
        self.enable_typed_state_trigger = enable_typed_state_trigger
        self.enable_careful_switch_trigger = enable_careful_switch_trigger
        self.warmup_sessions = warmup_sessions
        # One-shot trigger state.
        self.triggered_typed_state = False
        self.triggered_careful = False
        # Observation log.
        self.action_log: list[dict] = []
        self.observation_log: list[dict] = []

    def step(
        self,
        session_idx: int,
        metrics: dict,
        on_promote_typed_state: Optional[Callable[[], None]] = None,
        on_switch_careful: Optional[Callable[[], None]] = None,
    ) -> list[str]:
        """Inspect metrics from the session that just completed and fire triggers.

        Args:
            session_idx: index of the session that just completed (0-based).
            metrics: dict with optional keys 'accumulator_error',
                'constraint_precision', 'lag_recall'.
            on_promote_typed_state: callback to enable typed-state overlay.
            on_switch_careful: callback to switch the compaction policy
                from lossy to careful.

        Returns:
            list of action names dispatched in this step (may be empty).
        """
        actions: list[str] = []
        acc_err = metrics.get("accumulator_error")
        precision = metrics.get("constraint_precision")
        lag_recall = metrics.get("lag_recall")

        self.observation_log.append({
            "session": session_idx,
            "accumulator_error": acc_err,
            "constraint_precision": precision,
            "lag_recall": lag_recall,
        })

        if session_idx < self.warmup_sessions:
            return actions

        # Trigger 1: promote-to-typed-state on accumulator-error crossing.
        if (
            self.enable_typed_state_trigger
            and not self.triggered_typed_state
            and acc_err is not None
            and acc_err > self.theta_acc
        ):
            self.triggered_typed_state = True
            actions.append("promote_to_typed_state")
            self.action_log.append({
                "session": session_idx,
                "trigger": "accumulator_error",
                "observed_value": acc_err,
                "threshold": self.theta_acc,
                "action": "promote_to_typed_state",
            })
            if on_promote_typed_state is not None:
                on_promote_typed_state()

        # Trigger 2: switch-to-careful-compaction on precision drop.
        if (
            self.enable_careful_switch_trigger
            and not self.triggered_careful
            and precision is not None
            and precision < self.theta_prec
        ):
            self.triggered_careful = True
            actions.append("switch_compaction_policy")
            self.action_log.append({
                "session": session_idx,
                "trigger": "constraint_precision",
                "observed_value": precision,
                "threshold": self.theta_prec,
                "action": "switch_compaction_policy",
            })
            if on_switch_careful is not None:
                on_switch_careful()

        return actions

    def to_dict(self) -> dict:
        """Serializable summary of controller state for run output."""
        return {
            "config": {
                "theta_acc": self.theta_acc,
                "theta_prec": self.theta_prec,
                "enable_typed_state_trigger": self.enable_typed_state_trigger,
                "enable_careful_switch_trigger": self.enable_careful_switch_trigger,
                "warmup_sessions": self.warmup_sessions,
            },
            "triggered_typed_state": self.triggered_typed_state,
            "triggered_careful": self.triggered_careful,
            "action_log": self.action_log,
            "observation_log": self.observation_log,
        }
