"""
agingbench/runner/self_planning_eval.py — Shared evaluator for G5 self-planning metrics.

Each runner calls evaluate_self_planning_probes() after its main session loop
to compute rationale_recall, retrieval_competition, and update_propagation
metrics from the scenario's self_planning_probes.json.

This module is designed to work with BOTH externally-managed and self-planning
agents — the probes are the same, only the aging mechanism differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..metrics.deep_tier_metrics import (
    compute_rationale_recall,
    compute_retrieval_competition_score,
    compute_proactive_check_rate,
)
from ..metrics.aging import AgingCurve


def load_self_planning_probes(scenario_dir: Path) -> dict:
    """Load self_planning_probes.json from a scenario directory."""
    path = scenario_dir / "self_planning_probes.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def evaluate_rationale_probes(
    probes: list[dict],
    agent_fn,
    current_session: int,
    memory_text: str = "",
) -> list[dict]:
    """
    Run rationale probes that are available at the current session.

    Args:
        probes: list of rationale probe dicts from self_planning_probes.json
        agent_fn: callable(question: str) -> str that queries the agent
        current_session: current session/cycle number
        memory_text: current memory content (for context)

    Returns:
        list of {"probe_id", "score", "response"} dicts
    """
    results = []
    for probe in probes:
        available = probe.get("available_after_session", probe.get("available_after_cycle", 0))
        if current_session < available:
            continue

        question = probe["question"]
        response = agent_fn(question)
        score = compute_rationale_recall(response, probe)

        results.append({
            "probe_id": probe["id"],
            "score": score,
            "response": response[:200],
            "type": "rationale",
        })
    return results


def evaluate_competition_probes(
    probes: list[dict],
    agent_fn,
    current_session: int,
) -> list[dict]:
    """
    Run retrieval competition probes that are available at the current session.
    """
    results = []
    for probe in probes:
        available = probe.get("available_after_session", probe.get("available_after_cycle", 0))
        if current_session < available:
            continue

        question = probe["question"]
        response = agent_fn(question)
        score = compute_retrieval_competition_score(
            response, probe["target"], probe.get("competitors", [])
        )

        results.append({
            "probe_id": probe["id"],
            "score": score,
            "response": response[:200],
            "type": "competition",
        })
    return results


def evaluate_update_propagation(
    updates: list[dict],
    agent_fn,
    current_session: int,
) -> list[dict]:
    """
    Check if updates introduced at earlier sessions are reflected in current behavior.
    """
    results = []
    for update in updates:
        update_session = update.get("update_session", update.get("update_cycle", 0))
        if current_session <= update_session:
            continue  # update hasn't happened yet

        question = update.get("probe_question", "")
        if not question:
            continue

        response = agent_fn(question)
        keywords = update.get("keywords", [])
        response_lower = response.lower()
        propagated = any(kw.lower() in response_lower for kw in keywords)

        # Check for stale value (old value present instead of new)
        old_val = update.get("old_value", "")
        has_stale = old_val.lower() in response_lower if old_val else False

        results.append({
            "probe_id": update["id"],
            "propagated": propagated,
            "stale": has_stale and not propagated,
            "latency": current_session - update_session,
            "response": response[:200],
            "type": "update_propagation",
        })
    return results


def compute_session_g5_metrics(
    probes_data: dict,
    agent_fn,
    current_session: int,
    tool_calls: list[dict] = None,
    n_tasks: int = 1,
) -> dict:
    """
    Compute all G5 metrics for one session.

    Args:
        probes_data: loaded self_planning_probes.json
        agent_fn: callable(question: str) -> str
        current_session: current session number
        tool_calls: list of tool call dicts from this session
        n_tasks: number of tasks in this session

    Returns dict with:
        - proactive_check_rate: float [0, 1]
        - rationale_recall: float [0, 1] (avg across available probes)
        - retrieval_competition: float [0, 1] (avg across available probes)
        - update_propagation_rate: float [0, 1] (fraction of past updates reflected)
        - probe_details: list of individual probe results
    """
    result = {}
    all_probe_results = []

    # G5-M1: Proactive checking rate
    if tool_calls is not None:
        check_tools = set(probes_data.get("proactive_check_tools",
                          ["check_constraints", "search_memory", "read_file"]))
        result["proactive_check_rate"] = compute_proactive_check_rate(
            tool_calls, check_tools, n_tasks
        )
    else:
        result["proactive_check_rate"] = None

    # G5-M3: Rationale recall
    rationale_probes = probes_data.get("rationale_probes", [])
    if rationale_probes:
        rationale_results = evaluate_rationale_probes(
            rationale_probes, agent_fn, current_session
        )
        all_probe_results.extend(rationale_results)
        if rationale_results:
            result["rationale_recall"] = (
                sum(r["score"] for r in rationale_results) / len(rationale_results)
            )
        else:
            result["rationale_recall"] = None
    else:
        result["rationale_recall"] = None

    # G5-M4: Retrieval competition
    competition_probes = probes_data.get("retrieval_competition", [])
    if competition_probes:
        competition_results = evaluate_competition_probes(
            competition_probes, agent_fn, current_session
        )
        all_probe_results.extend(competition_results)
        if competition_results:
            result["retrieval_competition"] = (
                sum(r["score"] for r in competition_results) / len(competition_results)
            )
        else:
            result["retrieval_competition"] = None
    else:
        result["retrieval_competition"] = None

    # G5-M2: Update propagation
    update_probes = probes_data.get("update_propagation", [])
    if update_probes:
        update_results = evaluate_update_propagation(
            update_probes, agent_fn, current_session
        )
        all_probe_results.extend(update_results)
        if update_results:
            result["update_propagation_rate"] = (
                sum(1 for r in update_results if r["propagated"]) / len(update_results)
            )
        else:
            result["update_propagation_rate"] = None
    else:
        result["update_propagation_rate"] = None

    result["probe_details"] = all_probe_results
    return result
