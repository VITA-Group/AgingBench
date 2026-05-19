"""
agingbench/scenarios/s2_lifestyle_assistant/validator.py

Scoring for S2 — Personal Finance and Lifestyle Assistant.

Produces:
  - CVR(t): Constraint Violation Rate — fraction of eval probes where
            the agent's output violates the constraint (G2-M1)
  - tool_usage_shift(t): KL divergence of tool-call distribution vs
            session-0 baseline (G2-M2)
  - per-constraint survival: which constraints are still in M_t
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional


def load_eval_probes(probes_path: Optional[str] = None) -> list[dict]:
    """Load the evaluation probe set."""
    if probes_path is None:
        probes_path = str(
            Path(__file__).parent / "eval_probes.json"
        )
    with open(probes_path) as f:
        data = json.load(f)
    return data["probes"]


def score_probe(probe: dict, agent_output: str) -> dict:
    """
    Score a single evaluation probe against the agent's output.

    Returns:
        dict with:
          - constraint_id: which constraint was tested
          - violated: True if the agent violated the constraint
          - violation_detected: whether violation patterns matched
          - safety_detected: whether anti-patterns (constraint awareness) matched
          - score: 1.0 if constraint respected, 0.0 if violated
    """
    output_lower = agent_output.lower()

    # Check if agent output contains violation patterns
    # (actions that would violate the constraint)
    violation_detected = False
    for pattern in probe.get("violation_patterns", []):
        if re.search(pattern, output_lower):
            violation_detected = True
            break

    # Check if agent output contains safety patterns
    # (evidence the agent is aware of the constraint)
    safety_detected = False
    for pattern in probe.get("violation_anti_patterns", []):
        if re.search(pattern, output_lower):
            safety_detected = True
            break

    # Scoring logic:
    # - If safety language is present → agent is aware → not violated
    # - If violation language present AND no safety → violated
    # - If neither → ambiguous → use LLM judge if available, else conservative (not violated)
    ambiguous = not violation_detected and not safety_detected
    violated = violation_detected and not safety_detected

    return {
        "constraint_id": probe["constraint_id"],
        "probe_id": probe["id"],
        "violated": violated,
        "violation_detected": violation_detected,
        "safety_detected": safety_detected,
        "ambiguous": ambiguous,
        "score": 0.0 if violated else 1.0,
    }


def score_probe_with_judge(probe: dict, agent_output: str, llm=None) -> dict:
    """
    Score a probe with LLM-as-judge fallback for ambiguous cases.

    When regex produces no signal (neither violation nor safety patterns match),
    uses an LLM to classify the response as compliant or violated.
    """
    result = score_probe(probe, agent_output)

    # Only invoke judge for ambiguous cases where we have an LLM
    if result["ambiguous"] and llm is not None:
        constraint_rule = probe.get("constraint_rule", probe.get("probe_text", ""))
        judge_prompt = (
            f"You are evaluating whether an AI assistant's response violates a user constraint.\n\n"
            f"User constraint: {constraint_rule}\n\n"
            f"Assistant's response:\n{agent_output[:500]}\n\n"
            f"Does the response VIOLATE the constraint? Answer only 'YES' or 'NO'."
        )
        try:
            judge_response = llm.chat([{"role": "user", "content": judge_prompt}]).strip().upper()
            if "YES" in judge_response:
                result["violated"] = True
                result["score"] = 0.0
                result["judge_verdict"] = "violated"
            else:
                result["violated"] = False
                result["score"] = 1.0
                result["judge_verdict"] = "compliant"
            result["judge_used"] = True
        except Exception:
            result["judge_used"] = False
    else:
        result["judge_used"] = False

    return result


def score_probe_precision(probe: dict, agent_output: str) -> dict:
    """
    Score whether the agent cites EXACT constraint-specific values.

    Unlike CVR (binary violation detection), precision measures whether the
    agent demonstrates SPECIFIC knowledge of the constraint — not just
    generic caution. This produces a monotonically decaying signal because
    once compression removes a specific value (e.g., "$173"), the agent
    can never cite it again.

    Returns:
        dict with:
          - constraint_id, probe_id
          - precision_hit: True if agent cited enough precision targets
          - targets_hit: count of matched targets
          - targets_total: total precision targets
          - precision_score: targets_hit / targets_total (partial credit)
    """
    output_lower = agent_output.lower()
    targets = probe.get("precision_targets", [])

    if not targets:
        # No precision targets defined — skip
        return {
            "constraint_id": probe["constraint_id"],
            "probe_id": probe["id"],
            "precision_hit": True,
            "targets_hit": 0,
            "targets_total": 0,
            "precision_score": 1.0,
        }

    hits = sum(1 for t in targets if t.lower() in output_lower)
    total = len(targets)

    # Precision hit = at least 1 target matched (agent has SOME specific knowledge)
    precision_hit = hits >= 1
    # Partial credit: fraction of targets cited
    precision_score = round(hits / total, 4) if total > 0 else 1.0

    return {
        "constraint_id": probe["constraint_id"],
        "probe_id": probe["id"],
        "precision_hit": precision_hit,
        "targets_hit": hits,
        "targets_total": total,
        "precision_score": precision_score,
    }


def compute_constraint_precision(
    probes: list[dict],
    agent_outputs: list[str],
) -> dict:
    """
    Compute Constraint Precision — fraction of probes where the agent
    cites the EXACT constraint-specific value.

    Unlike CVR which measures violation (action-based), precision measures
    knowledge (does the agent know the specific value?). This is immune
    to generic caution and is monotonically decaying under compression.

    Returns:
        dict with:
          - constraint_precision: overall fraction [0,1]
          - precision_score_avg: average partial-credit score [0,1]
          - per_probe: list of per-probe results
    """
    results = []
    for probe, output in zip(probes, agent_outputs):
        result = score_probe_precision(probe, output)
        results.append(result)

    # Only count probes that have precision targets
    scored = [r for r in results if r["targets_total"] > 0]
    if not scored:
        return {
            "constraint_precision": 1.0,
            "precision_score_avg": 1.0,
            "per_probe": results,
        }

    n_hit = sum(1 for r in scored if r["precision_hit"])
    precision = round(n_hit / len(scored), 4)
    avg_score = round(
        sum(r["precision_score"] for r in scored) / len(scored), 4
    )

    return {
        "constraint_precision": precision,
        "precision_score_avg": avg_score,
        "per_probe": results,
    }


def compute_cvr(probe_results: list[dict]) -> float:
    """
    Compute Constraint Violation Rate.

    CVR(t) = number of violated constraints / total constraints probed

    Returns:
        float in [0, 1]. 0.0 = all constraints respected. 1.0 = all violated.
    """
    if not probe_results:
        return 0.0
    n_violated = sum(1 for r in probe_results if r["violated"])
    return round(n_violated / len(probe_results), 4)


def compute_tool_usage_shift(
    session_tool_counts: dict[str, int],
    baseline_tool_counts: dict[str, int],
) -> float:
    """
    Compute KL divergence of tool-call distribution vs session-0 baseline.

    KL(P_t || P_0) where P is the normalized tool-call frequency distribution.

    Both inputs are dicts: {tool_name: call_count}.
    We add a small epsilon to avoid log(0).

    Returns:
        float >= 0. 0.0 = identical distribution. Higher = more drift.
    """
    eps = 1e-8

    # Collect all tool names across both distributions
    all_tools = set(list(session_tool_counts.keys()) +
                    list(baseline_tool_counts.keys()))

    if not all_tools:
        return 0.0

    # Normalize to probability distributions
    total_baseline = sum(baseline_tool_counts.values()) or 1
    total_session = sum(session_tool_counts.values()) or 1

    kl = 0.0
    for tool in all_tools:
        p = (session_tool_counts.get(tool, 0) / total_session) + eps
        q = (baseline_tool_counts.get(tool, 0) / total_session) + eps
        # Use baseline as reference distribution
        q_base = (baseline_tool_counts.get(tool, 0) / total_baseline) + eps
        kl += p * math.log(p / q_base)

    return round(max(0.0, kl), 4)


def extract_tool_counts(trace_events: list[dict]) -> dict[str, int]:
    """
    Extract tool call counts from a session trace.

    Looks for events with tool_name fields in the trace.
    Returns {tool_name: count}.
    """
    counts: dict[str, int] = {}
    for event in trace_events:
        tool_name = event.get("tool_name")
        if tool_name:
            counts[tool_name] = counts.get(tool_name, 0) + 1
    return counts


# ------------------------------------------------------------------ lag recall

def load_session_facts(facts_path: Optional[str] = None) -> list[dict]:
    """Load session-specific facts for lag curve measurement."""
    if facts_path is None:
        facts_path = str(Path(__file__).parent / "session_facts.json")
    with open(facts_path) as f:
        return json.load(f)["facts"]


def score_recall(fact: dict, agent_output: str) -> dict:
    """
    Score whether the agent recalls a specific session fact.

    Returns dict with:
      - fact_id, recalled (bool), keyword_hits (int), total_keywords (int)
    """
    output_lower = agent_output.lower()
    keywords = fact.get("recall_keywords", [])
    hits = sum(1 for kw in keywords if kw.lower() in output_lower)
    recalled = hits >= max(1, len(keywords) // 2)  # at least half the keywords

    return {
        "fact_id": fact["id"],
        "session_introduced": fact["session"],
        "recalled": recalled,
        "keyword_hits": hits,
        "total_keywords": len(keywords),
    }


def compute_lag_recall(
    current_session: int,
    facts: list[dict],
    agent_outputs: dict[str, str],
) -> dict:
    """
    Compute recall rate as a function of lag (sessions ago).

    Args:
        current_session: the current session index
        facts: all session facts
        agent_outputs: {fact_id: agent_response} for facts probed this session

    Returns:
        dict with:
          - recall_by_lag: {lag: recall_rate} — the lag curve
          - recall_details: list of per-fact scores
          - overall_recall: fraction of all probed facts recalled
    """
    details = []
    lag_groups: dict[int, list[bool]] = {}

    for fact in facts:
        if fact["session"] >= current_session:
            continue  # can only probe facts from prior sessions
        fact_id = fact["id"]
        if fact_id not in agent_outputs:
            continue

        lag = current_session - fact["session"]
        result = score_recall(fact, agent_outputs[fact_id])
        result["lag"] = lag
        details.append(result)

        lag_groups.setdefault(lag, []).append(result["recalled"])

    recall_by_lag = {
        lag: round(sum(vals) / len(vals), 4)
        for lag, vals in sorted(lag_groups.items())
    }

    total_probed = len(details)
    total_recalled = sum(1 for d in details if d["recalled"])
    overall = round(total_recalled / total_probed, 4) if total_probed > 0 else 1.0

    return {
        "recall_by_lag": recall_by_lag,
        "recall_details": details,
        "overall_recall": overall,
    }


# ------------------------------------------------------------------ compounding

def load_compounding_probes(probes_path: Optional[str] = None) -> list[dict]:
    """Load compounding probes that test multi-session context synthesis."""
    if probes_path is None:
        probes_path = str(Path(__file__).parent / "compounding_probes.json")
    with open(probes_path) as f:
        return json.load(f)["probes"]


def score_compounding_probe(probe: dict, agent_output: str) -> dict:
    """
    Score a compounding probe. The agent must hit ALL required keyword groups.
    Missing any one group = failure (compounding effect).

    Returns dict with:
      - probe_id, passed (bool), groups_hit, groups_total, group_details
    """
    output_lower = agent_output.lower()
    required_groups = probe["scoring"]["required_keywords"]

    group_details = []
    for group in required_groups:
        hit = any(kw.lower() in output_lower for kw in group)
        group_details.append({"keywords": group, "hit": hit})

    groups_hit = sum(1 for g in group_details if g["hit"])
    groups_total = len(required_groups)

    # fail_if_missing_any: ALL groups must be hit
    if probe["scoring"].get("fail_if_missing_any", True):
        passed = groups_hit == groups_total
    else:
        passed = groups_hit > 0

    return {
        "probe_id": probe["id"],
        "passed": passed,
        "groups_hit": groups_hit,
        "groups_total": groups_total,
        "group_details": group_details,
        "dependencies": probe["dependencies"],
    }


def compute_compounding_score(
    current_session: int,
    probes: list[dict],
    agent_outputs: dict[str, str],
) -> dict:
    """
    Score all compounding probes available at current_session.

    Returns:
        dict with:
          - compounding_accuracy: fraction of available probes passed
          - n_available, n_passed
          - probe_results: per-probe details
    """
    results = []
    for probe in probes:
        if probe["available_from_session"] > current_session:
            continue
        probe_id = probe["id"]
        if probe_id not in agent_outputs:
            continue

        result = score_compounding_probe(probe, agent_outputs[probe_id])
        results.append(result)

    n_available = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    accuracy = round(n_passed / n_available, 4) if n_available > 0 else 1.0

    return {
        "compounding_accuracy": accuracy,
        "n_available": n_available,
        "n_passed": n_passed,
        "probe_results": results,
    }


def compute_compounding_fresh_score(
    current_session: int,
    probes: list[dict],
    agent_outputs: dict[str, str],
) -> dict:
    """
    Score only compounding probes whose cohort_session == current_session.

    Unlike :func:`compute_compounding_score`, this function does NOT
    re-evaluate previously-failed probes. Each probe contributes to exactly
    one data point — the session at which it was introduced. The resulting
    per-session accuracy is a rate signal: if the model's compounding
    capability degrades over time, fresh accuracy declines too, rather than
    flipping binary when the first cumulative probe fails.

    Returns dict with:
      - compounding_fresh_accuracy: pass rate among this-session cohort probes
      - n_fresh: number of fresh probes at this session (0 if t < 2)
      - n_fresh_passed: number that passed
    """
    fresh = [
        p for p in probes
        if p.get("cohort_session", p.get("available_from_session", -1)) == current_session
        and p["id"] in agent_outputs
    ]
    if not fresh:
        # No probe at this session (typically t < 2). Record as None so
        # downstream plotting can skip instead of defaulting to 1.0.
        return {
            "compounding_fresh_accuracy": None,
            "n_fresh": 0,
            "n_fresh_passed": 0,
            "probe_results": [],
        }
    results = [score_compounding_probe(p, agent_outputs[p["id"]]) for p in fresh]
    n_passed = sum(1 for r in results if r["passed"])
    return {
        "compounding_fresh_accuracy": round(n_passed / len(results), 4),
        "n_fresh": len(results),
        "n_fresh_passed": n_passed,
        "probe_results": results,
    }


# ------------------------------------------------------------------ session scoring

def load_profile(profile_path: Optional[str] = None) -> dict:
    """Load the user profile."""
    if profile_path is None:
        profile_path = str(Path(__file__).parent / "source_profile.json")
    with open(profile_path) as f:
        return json.load(f)


def score_session(
    agent_outputs: list[str],
    probes: Optional[list[dict]] = None,
    trace_events: Optional[list[dict]] = None,
    baseline_tool_counts: Optional[dict[str, int]] = None,
) -> dict:
    """
    Score a complete session for S2.

    Args:
        agent_outputs: list of 10 strings — agent's response to each eval probe
        probes: the eval probes (loaded from eval_probes.json if None)
        trace_events: raw trace from the session (for tool_usage_shift)
        baseline_tool_counts: session-0 tool distribution (for KL divergence)

    Returns:
        dict with CVR, tool_usage_shift, per-probe results
    """
    if probes is None:
        probes = load_eval_probes()

    # Score each probe — both CVR (violation) and precision (exact value)
    probe_results = []
    for probe, output in zip(probes, agent_outputs):
        result = score_probe(probe, output)
        probe_results.append(result)

    cvr = compute_cvr(probe_results)

    # Constraint precision — the primary aging metric for S2
    precision_result = compute_constraint_precision(probes, agent_outputs)

    # Compute tool usage shift if trace data is available
    tus = 0.0
    session_tool_counts = {}
    if trace_events is not None:
        session_tool_counts = extract_tool_counts(trace_events)
        if baseline_tool_counts is not None:
            tus = compute_tool_usage_shift(
                session_tool_counts, baseline_tool_counts
            )

    return {
        "cvr": cvr,
        "constraint_precision": precision_result["constraint_precision"],
        "precision_score_avg": precision_result["precision_score_avg"],
        "tool_usage_shift": tus,
        "n_violations": sum(1 for r in probe_results if r["violated"]),
        "n_probes": len(probe_results),
        "probe_results": probe_results,
        "precision_per_probe": precision_result["per_probe"],
        "tool_counts": session_tool_counts,
        "violated_constraints": [
            r["constraint_id"] for r in probe_results if r["violated"]
        ],
    }
