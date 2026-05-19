"""
synthetic_probe.py — Run AgingBench scenarios as injected probes against
the user's deployed agent, then mix the resulting AgingCards into the
same telemetry stream.

This is the bridge between pure-telemetry (workload-conditional coverage)
and pure-scenarios (controlled stress). The user provides:
  - which scenario(s) to inject
  - how to invoke their agent (a callable / subprocess command)
  - cadence (handled by the user's cron/CI; we run on demand)

We provide the probe content (S1-S6 already exist), a thin runner that
drives the scenario against the user's agent via an AgentAdapter, and
AgingCard merging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .schema import OutcomeEvent


# Scenario names that ship as injectable probes. These are the existing
# AgingBench scenarios re-used as probe sources.
INJECTABLE_SCENARIOS = (
    "s1_research_literature",
    "s2_lifestyle_assistant",
    "s3_knowledge_base",
    "s4_software_engineering",
    "s6_naturalistic",
)


@dataclass
class ProbeSchedule:
    """User-supplied schedule for synthetic-probe injection."""
    scenario_id:    str
    sessions:       int = 4
    seed:           int = 42
    sut_yaml:       Optional[Path] = None     # path to a SUT YAML wrapping their agent
    output_dir:     Optional[Path] = None     # where to write the resulting card


@dataclass
class ProbeResult:
    scenario_id:    str
    aging_card:     dict
    outcome_events: list[OutcomeEvent] = field(default_factory=list)


def list_injectable_scenarios() -> list[str]:
    return list(INJECTABLE_SCENARIOS)


def run_synthetic_probe(schedule: ProbeSchedule) -> ProbeResult:
    """Run one AgingBench scenario as a probe against the user's deployed
    agent, producing an AgingCard + a stream of synthetic OutcomeEvents.

    This is a thin wrapper around the existing scenario-run path: same
    runner, same probe set, same scoring. The output is tagged
    `derived_from: 'synthetic_probe'` so downstream tools know it was
    AgingBench-supplied content (not workload-derived).

    Note: this does NOT actually execute the LLM/agent — that requires
    a live SUT YAML and is run via the standard CLI. This function
    sketches the API; callers run the scenario via:

        agingbench run --scenario {schedule.scenario_id} \\
            --sut {schedule.sut_yaml} --sessions {schedule.sessions} \\
            --seeds 1 --card --output {schedule.output_dir}

    and then call `load_probe_result(card_path)` to ingest it.
    """
    if schedule.scenario_id not in INJECTABLE_SCENARIOS:
        raise ValueError(
            f"{schedule.scenario_id!r} is not in the injectable-probe set "
            f"({INJECTABLE_SCENARIOS}). Tier-2 scenarios (S5, S7, S8) cannot "
            f"be injected as probes — they require Docker/CLI environment "
            f"control that telemetry mode doesn't have."
        )
    raise NotImplementedError(
        "run_synthetic_probe is the API placeholder; users invoke the "
        "scenario via `agingbench run --scenario ... --sut ... --card` "
        "and then ingest the resulting card via load_probe_result()."
    )


def load_probe_result(card_path: Path) -> ProbeResult:
    """Load a previously-run scenario AgingCard and convert its per-session
    scores into synthetic OutcomeEvents that can be merged into a
    telemetry trace.
    """
    import json
    with Path(card_path).open() as f:
        card = json.load(f)

    scenario = card.get("scenario", "unknown")
    checkpoints = card.get("checkpoints", [])
    # Derive per-session OutcomeEvents from the aging-curve checkpoints:
    # a checkpoint score above 0.5 = success, below = fail.
    outcomes: list[OutcomeEvent] = []
    for entry in checkpoints:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        sess_idx, score = entry
        outcome_val = "success" if (score is not None and score >= 0.5) else "fail"
        outcomes.append(OutcomeEvent(
            session_id=f"probe::{scenario}::s{int(sess_idx)}",
            task_id=f"probe::{scenario}::s{int(sess_idx)}",
            outcome=outcome_val,
            gold_label={"score": score},
            source="synthetic_probe",
        ))

    return ProbeResult(
        scenario_id=scenario,
        aging_card=card,
        outcome_events=outcomes,
    )


def merge_probe_into_card(card: dict, probe: ProbeResult) -> dict:
    """Attach a probe's headline metrics to a telemetry-derived AgingCard
    under a `synthetic_probes` block, so the final card carries both
    observational and controlled signals side by side.
    """
    block = card.setdefault("synthetic_probes", {})
    pr_card = probe.aging_card
    block[probe.scenario_id] = {
        "headline":           pr_card.get("headline", {}),
        "mechanism_metrics":  pr_card.get("mechanism_metrics", {}),
        "n_sessions":         pr_card.get("n_sessions"),
        "n_outcome_events":   len(probe.outcome_events),
        "derived_from":       "synthetic_probe",
        "probe_source_card":  None,    # filled in by the caller if shipping the source
    }
    return card
