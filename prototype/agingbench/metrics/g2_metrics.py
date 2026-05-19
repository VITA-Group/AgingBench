"""
agingbench/metrics/g2_metrics.py — G2: Behavioral Drift + Constraint Following.

Metrics:
  CVR(t)              Constraint Violation Rate — fraction of outputs violating
                      explicit constraints at cycle t.
  tool_usage_shift(t) KL divergence of the tool-call distribution at cycle t
                      vs. the session-0 baseline distribution.
  instruction_decay δ Attribution metric (§6) — task_m(oracle) − task_m(degraded).
"""

from __future__ import annotations

import math
from collections import Counter


# ---------------------------------------------------------------------------
# CVR — Constraint Violation Rate
# ---------------------------------------------------------------------------

def compute_cvr(
    task_outputs: list[dict],
    constraints: list[dict],
) -> float:
    """
    Compute CVR for one session.

    Parameters
    ----------
    task_outputs : list[dict]
        Each dict must have at least:
          - "task_id": str
          - "output": str  (agent's response text)
          - "constraint_ids": list[str]  (which constraints apply)
    constraints : list[dict]
        Each dict must have:
          - "constraint_id": str
          - "check": callable(output_str) -> bool  (True = passes, False = violated)

    Returns
    -------
    float  CVR in [0, 1].  0 = no violations, 1 = all violated.
    """
    if not task_outputs:
        return 0.0

    constraint_map = {c["constraint_id"]: c["check"] for c in constraints}

    total_checks = 0
    violations = 0
    for task in task_outputs:
        for cid in task.get("constraint_ids", []):
            check_fn = constraint_map.get(cid)
            if check_fn is None:
                continue
            total_checks += 1
            if not check_fn(task["output"]):
                violations += 1

    return violations / total_checks if total_checks > 0 else 0.0


# ---------------------------------------------------------------------------
# tool_usage_shift — KL divergence of tool-call distribution
# ---------------------------------------------------------------------------

def _tool_call_distribution(trace_events: list[dict]) -> dict[str, float]:
    """
    Extract a normalized distribution over tool names from a list of trace
    events for one session.

    Each event with event == "tool_call" must have a "tool_name" field.
    Returns a dict mapping tool_name -> probability.
    """
    counts: Counter = Counter()
    for ev in trace_events:
        if ev.get("event") == "tool_call":
            counts[ev["tool_name"]] += 1

    total = sum(counts.values())
    if total == 0:
        return {}
    return {name: count / total for name, count in counts.items()}


def compute_tool_usage_shift(
    baseline_events: list[dict],
    current_events: list[dict],
    smoothing: float = 1e-8,
) -> float:
    """
    KL divergence D_KL(current || baseline) of tool-call distributions.

    Parameters
    ----------
    baseline_events : list[dict]
        Trace events from session 0 (the reference distribution).
    current_events : list[dict]
        Trace events from the current session.
    smoothing : float
        Laplace-style smoothing to avoid log(0). Added to both distributions.

    Returns
    -------
    float  KL divergence in nats.  0 = identical distributions.
    """
    p = _tool_call_distribution(baseline_events)
    q = _tool_call_distribution(current_events)

    if not p and not q:
        return 0.0

    # union of all tool names
    all_tools = set(p.keys()) | set(q.keys())
    if not all_tools:
        return 0.0

    # smoothed distributions
    n = len(all_tools)
    p_smooth = {t: p.get(t, 0.0) + smoothing for t in all_tools}
    q_smooth = {t: q.get(t, 0.0) + smoothing for t in all_tools}

    # renormalize
    p_total = sum(p_smooth.values())
    q_total = sum(q_smooth.values())
    p_norm = {t: v / p_total for t, v in p_smooth.items()}
    q_norm = {t: v / q_total for t, v in q_smooth.items()}

    # KL(q || p): how much current diverges from baseline
    kl = 0.0
    for t in all_tools:
        kl += q_norm[t] * math.log(q_norm[t] / p_norm[t])

    return max(kl, 0.0)  # numerical safety


# ---------------------------------------------------------------------------
# instruction_decay δ — §6 Counterfactual Diagnosis
# NOTE: compute_instruction_decay() has been removed. The per-session
# utilization error (1 − Acc_P3) from diagnostics/partitioner.py now provides
# a more precise, per-session decomposition of reasoning failure.
