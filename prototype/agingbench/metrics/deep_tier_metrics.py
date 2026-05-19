"""
agingbench/metrics/deep_tier_metrics.py — Deep-tier metrics for G1-G4.

These metrics extend G1-G4 with "deep-tier" measurements that capture aging
signals in BOTH externally-managed and self-planning agents. They target
failure modes that surface-tier metrics (keyword_m, CVR, fidelity, LA) miss:

- G1 deep: rationale_recall — can the agent explain WHY, not just WHAT?
- G1 deep: retrieval_competition — correct fact retrieved among similar alternatives?
- G2 deep: proactive_check_rate — does the agent still verify before acting?
- G2 deep: update_propagation_rate — does behavior reflect recent changes?
- G3 deep: update_propagation_latency — sessions until memory reflects a change
- G4 deep: cross_cutting_accuracy — can agent predict multi-module cascading effects?
- G4 deep: rationale_informed_LA — does knowing WHY improve impact prediction?

Each metric works identically for externally-managed and self-planning agents,
enabling direct comparison of aging mechanisms across agent types.
"""

from __future__ import annotations

import math
from typing import Optional


def compute_proactive_check_rate(
    session_tool_calls: list[dict],
    check_tool_names: set[str] = None,
    n_tasks: int = 1,
) -> float:
    """
    G5-M1: Proactive Checking Rate.

    Measures: does the agent still proactively verify constraints/context
    before acting, or does it skip this step as sessions accumulate?

    Unlike tool_usage_shift (KL divergence of distribution), this metric
    directly measures the FRACTION of tasks where the agent calls a
    verification tool BEFORE producing its answer.

    For externally-managed agents: counts check_constraints / search_memory calls.
    For self-planning agents: counts file reads / memory lookups before action.

    Args:
        session_tool_calls: list of {"tool": name, "position": "before_answer"|"after_answer"}
        check_tool_names: set of tool names that count as "checking" (default: common check tools)
        n_tasks: number of tasks in this session

    Returns:
        float in [0, 1]. 1.0 = agent checks before every task. 0.0 = never checks.
    """
    if check_tool_names is None:
        check_tool_names = {
            "check_constraints", "search_memory", "read_file",
            "check_all_constraints", "list_files",
        }

    if not session_tool_calls or n_tasks == 0:
        return 0.0

    # Count tasks where at least one check tool was called
    tasks_with_check = 0
    current_task_has_check = False

    for call in session_tool_calls:
        tool = call.get("tool", call.get("name", ""))
        if tool in check_tool_names:
            current_task_has_check = True

    # Simplified: if any check tool was called in the session, rate = calls/tasks
    check_calls = sum(1 for c in session_tool_calls
                      if c.get("tool", c.get("name", "")) in check_tool_names)
    return min(1.0, check_calls / n_tasks)


def compute_update_propagation_latency(
    update_session: int,
    behavior_changed_session: Optional[int],
    max_sessions: int = 20,
) -> int:
    """
    G5-M2: Update Propagation Latency.

    Measures: when a constraint or decision changes at session t, how many
    sessions until the agent's behavior reflects the change?

    For externally-managed: change at session 6 (budget $200->$250), measure
    when agent first recommends within $200-$250 range without hesitation.
    For self-planning: change at session 6, measure when agent updates its
    constraint document to reflect new value.

    Args:
        update_session: session where the update was introduced
        behavior_changed_session: first session where agent behavior reflects the update.
            None if agent never reflects the update.
        max_sessions: cap for "never reflected" case

    Returns:
        int: number of sessions of latency. 0 = immediate. max_sessions = never.
    """
    if behavior_changed_session is None:
        return max_sessions
    return max(0, behavior_changed_session - update_session)


def compute_update_propagation_rate(
    updates: list[dict],
    agent_outputs_after: list[list[str]],
    max_sessions: int = 20,
) -> float:
    """
    G5-M2b: Update Propagation Rate (aggregated).

    Fraction of updates that are reflected in agent behavior within
    `grace_period` sessions after the update.

    Args:
        updates: list of {"session": int, "old_value": str, "new_value": str, "keywords": [str]}
        agent_outputs_after: agent_outputs_after[i] = list of agent outputs from
            sessions after update i was introduced
        max_sessions: max sessions to check

    Returns:
        float in [0, 1]. 1.0 = all updates propagated. 0.0 = none propagated.
    """
    if not updates:
        return 1.0

    propagated = 0
    for i, update in enumerate(updates):
        new_keywords = update.get("keywords", [update.get("new_value", "")])
        outputs = agent_outputs_after[i] if i < len(agent_outputs_after) else []

        # Check if any output after the update contains the new value
        for output in outputs:
            output_lower = output.lower()
            if any(kw.lower() in output_lower for kw in new_keywords):
                propagated += 1
                break

    return propagated / len(updates)


def compute_rationale_recall(
    agent_output: str,
    rationale_probe: dict,
) -> float:
    """
    G5-M3: Design Rationale Recall.

    Measures: can the agent explain WHY a decision was made, not just WHAT
    was decided? This tests memory of the decision process, which can't be
    recovered by reading current code or documents.

    For externally-managed: rationale is in compressed memory, degrades with compression.
    For self-planning: rationale may be in agent's notes, but deprioritized over time
    as the agent focuses on current tasks.

    Args:
        agent_output: the agent's response to a "why" query
        rationale_probe: {
            "question": "Why was hashlib chosen over bcrypt?",
            "gold_rationale": "stdlib availability, no external dependency",
            "rationale_keywords": ["stdlib", "dependency", "hashlib"],
            "decision_keywords": ["hashlib"]  # these test WHAT, not WHY
        }

    Returns:
        float in [0, 1].
        1.0 = agent explains the rationale (WHY keywords present)
        0.5 = agent knows WHAT was decided but not WHY
        0.0 = agent doesn't recall the decision at all
    """
    output_lower = agent_output.lower()

    # Check if agent recalls the rationale (WHY)
    rationale_kws = rationale_probe.get("rationale_keywords", [])
    decision_kws = rationale_probe.get("decision_keywords", [])

    rationale_hits = sum(1 for kw in rationale_kws if kw.lower() in output_lower)
    decision_hits = sum(1 for kw in decision_kws if kw.lower() in output_lower)

    rationale_recall = rationale_hits / len(rationale_kws) if rationale_kws else 0.0
    decision_recall = decision_hits / len(decision_kws) if decision_kws else 0.0

    # Also try semantic similarity for rationale if keyword match is weak
    if rationale_recall < 0.5:
        try:
            from .semantic_scorer import semantic_score
            gold = rationale_probe.get("gold_rationale", "")
            if gold:
                sim = semantic_score(agent_output, gold)
                rationale_recall = max(rationale_recall, sim)
        except Exception:
            pass

    if rationale_recall >= 0.5:
        return 1.0  # knows WHY
    elif decision_recall >= 0.5:
        return 0.5  # knows WHAT but not WHY
    else:
        return 0.0  # doesn't recall at all


def compute_retrieval_competition_score(
    agent_output: str,
    target_fact: dict,
    competitor_facts: list[dict],
) -> float:
    """
    G5-M4: Retrieval Competition Score.

    Measures: when multiple similar facts exist in memory (e.g., budgets
    from different projects, latencies from different benchmarks), does the
    agent retrieve the CORRECT one?

    This captures the D1/D2 failure mode that is the primary aging mechanism
    for self-planning agents with growing memory.

    Args:
        agent_output: agent's response to a query about target_fact
        target_fact: {"fact": str, "keywords": [str]} — the correct answer
        competitor_facts: list of {"fact": str, "keywords": [str]} — similar but wrong

    Returns:
        float: 1.0 = correct fact retrieved, 0.5 = ambiguous, 0.0 = wrong fact retrieved
    """
    output_lower = agent_output.lower()

    target_hits = sum(1 for kw in target_fact["keywords"] if kw.lower() in output_lower)
    target_score = target_hits / len(target_fact["keywords"]) if target_fact["keywords"] else 0.0

    # Check if any competitor's keywords appear (indicating confusion)
    max_competitor_score = 0.0
    for comp in competitor_facts:
        comp_hits = sum(1 for kw in comp["keywords"] if kw.lower() in output_lower)
        comp_score = comp_hits / len(comp["keywords"]) if comp["keywords"] else 0.0
        max_competitor_score = max(max_competitor_score, comp_score)

    if target_score > 0.5 and target_score > max_competitor_score:
        return 1.0  # correct retrieval
    elif target_score > 0 and max_competitor_score > 0:
        return 0.5  # ambiguous (both target and competitor present)
    elif max_competitor_score > 0.5:
        return 0.0  # wrong fact retrieved
    elif target_score > 0:
        return target_score  # partial correct
    else:
        return 0.0  # nothing retrieved


def score_self_planning_session(
    tool_calls: list[dict],
    n_tasks: int,
    updates_applied: list[dict] = None,
    agent_outputs: list[str] = None,
    rationale_probes: list[dict] = None,
    competition_probes: list[dict] = None,
) -> dict:
    """
    Compute all G5 metrics for one session.

    Returns dict with proactive_check_rate, rationale_recall, competition_score.
    Update propagation is computed across sessions, not per-session.
    """
    result = {}

    # G5-M1: Proactive checking rate
    result["proactive_check_rate"] = compute_proactive_check_rate(
        tool_calls, n_tasks=n_tasks
    )

    # G5-M3: Rationale recall (if probes provided)
    if rationale_probes and agent_outputs:
        scores = []
        for probe in rationale_probes:
            # Use the last agent output as the response to the probe
            output = agent_outputs[-1] if agent_outputs else ""
            scores.append(compute_rationale_recall(output, probe))
        result["rationale_recall"] = sum(scores) / len(scores) if scores else 1.0

    # G5-M4: Retrieval competition (if probes provided)
    if competition_probes and agent_outputs:
        scores = []
        for probe in competition_probes:
            output = agent_outputs[-1] if agent_outputs else ""
            scores.append(compute_retrieval_competition_score(
                output, probe["target"], probe.get("competitors", [])
            ))
        result["retrieval_competition"] = sum(scores) / len(scores) if scores else 1.0

    return result


# ── Self-Planning (S5 — workspace-file) Metrics ───────────────────────────────
#
# These metrics capture aging mechanisms unique to agents that manage their own
# workspace files. The dominant failure mode is overwrite loss (Family B), not
# compression loss (Family C).


def compute_overwrite_loss_rate(
    facts_registry: list[dict],
    workspace_text: str,
    current_block: int,
) -> float:
    """
    Fraction of previously-stored facts no longer retrievable in workspace.

    A fact is "lost" if it was introduced at block < current_block AND none
    of its keywords appear in the workspace files. This captures the silent
    overwrite failure: agent stores fact at block 0, overwrites at block 3,
    fact is permanently gone.

    Structurally monotonic: once a fact is overwritten, loss_rate can only
    increase (or stay the same if no new overwrites occur).

    Args:
        facts_registry: list of {"id": str, "keywords": [str], "introduced_at_block": int}
        workspace_text: concatenated text of all workspace files
        current_block: current evaluation block

    Returns:
        float in [0, 1]. 0.0 = all prior facts present. 1.0 = all lost.
    """
    ws_lower = workspace_text.lower()
    prior_facts = [f for f in facts_registry if f.get("introduced_at_block", 0) < current_block]

    if not prior_facts:
        return 0.0

    lost = 0
    for fact in prior_facts:
        keywords = fact.get("keywords", [])
        if not keywords:
            continue
        # Fact is "present" if at least one keyword survives
        if not any(kw.lower() in ws_lower for kw in keywords):
            lost += 1

    return lost / len(prior_facts)


def compute_workspace_fidelity(
    facts_registry: list[dict],
    workspace_text: str,
    current_block: int,
    group_by_cohort: bool = False,
) -> dict | float:
    """
    Fidelity of workspace files against all facts ever introduced.

    Like G3-M1 fidelity but applied to self-managed workspace instead of
    compressed memory. Checks keyword survival per fact.

    If group_by_cohort=True, returns per-origin-block fidelity (for lag curves).
    Otherwise returns aggregate fidelity.

    Args:
        facts_registry: list of {"id": str, "keywords": [str], "introduced_at_block": int}
        workspace_text: concatenated text of all workspace files
        current_block: current evaluation block
        group_by_cohort: if True, return {block: fidelity} dict

    Returns:
        float (aggregate) or dict (per-cohort) in [0, 1].
    """
    ws_lower = workspace_text.lower()
    prior_facts = [f for f in facts_registry if f.get("introduced_at_block", 0) <= current_block]

    if not prior_facts:
        return {} if group_by_cohort else 1.0

    if group_by_cohort:
        cohorts: dict[int, list[float]] = {}
        for fact in prior_facts:
            block = fact.get("introduced_at_block", 0)
            keywords = fact.get("keywords", [])
            if not keywords:
                continue
            hit_rate = sum(1 for kw in keywords if kw.lower() in ws_lower) / len(keywords)
            cohorts.setdefault(block, []).append(hit_rate)
        return {b: sum(scores) / len(scores) for b, scores in cohorts.items()}
    else:
        scores = []
        for fact in prior_facts:
            keywords = fact.get("keywords", [])
            if not keywords:
                continue
            hit_rate = sum(1 for kw in keywords if kw.lower() in ws_lower) / len(keywords)
            scores.append(hit_rate)
        return sum(scores) / len(scores) if scores else 1.0


def compute_cohort_keyword_m(
    facts_registry: list[dict],
    workspace_text: str,
    current_block: int,
) -> dict[int, float]:
    """
    Per-cohort keyword survival in workspace (adapted keyword_m for self-planning).

    Standard keyword_m goes UP for self-planning agents because the workspace
    accumulates facts. Per-cohort measurement fixes this: for facts introduced
    at block b, what fraction of their keywords survive at block t?

    Facts from older cohorts have been through more potential overwrites,
    so cohort_keyword_m(b=0, t=5) < cohort_keyword_m(b=4, t=5).

    Args:
        facts_registry: list of {"id": str, "keywords": [str], "introduced_at_block": int}
        workspace_text: concatenated text of all workspace files
        current_block: current block being evaluated

    Returns:
        dict mapping cohort_block -> keyword survival rate [0, 1]
    """
    ws_lower = workspace_text.lower()
    cohorts: dict[int, list[float]] = {}

    for fact in facts_registry:
        block = fact.get("introduced_at_block", 0)
        if block > current_block:
            continue
        keywords = fact.get("keywords", [])
        if not keywords:
            continue
        survived = sum(1 for kw in keywords if kw.lower() in ws_lower)
        cohorts.setdefault(block, []).append(survived / len(keywords))

    return {b: sum(scores) / len(scores) for b, scores in cohorts.items()}


# ── Accumulation-Complexity Metrics ──────────────────────────────────────────
#
# These metrics detect aging from GROWING state, not information loss.
# They are complementary to the loss-based metrics above.


def compute_interference_rate(
    probe_responses: list[dict],
) -> float:
    """
    Fraction of probes where the agent retrieves a WRONG similar fact
    instead of the correct one.

    Interference = the agent confuses similar entries (e.g., Q1 budget vs
    Q2 budget). Structurally monotonic: more similar entries accumulate
    over time, increasing competition.

    Each probe_response has:
        - "target_keywords": keywords of the CORRECT fact
        - "competitor_keywords": keywords of SIMILAR but wrong facts
        - "response_text": the agent's actual response

    Returns float in [0, 1]. 0.0 = never confused. 1.0 = always retrieves wrong entry.
    """
    if not probe_responses:
        return 0.0

    interference_count = 0
    for pr in probe_responses:
        resp = pr.get("response_text", "").lower()
        target_kws = pr.get("target_keywords", [])
        competitor_kws = pr.get("competitor_keywords", [])

        target_hits = sum(1 for kw in target_kws if kw.lower() in resp)
        competitor_hits = sum(1 for kw in competitor_kws if kw.lower() in resp)

        # Interference: competitor keywords present AND target keywords absent/weaker
        if competitor_hits > 0 and competitor_hits >= target_hits:
            interference_count += 1

    return interference_count / len(probe_responses)


def compute_conflation_rate(
    probe_responses: list[dict],
) -> float:
    """
    Fraction of probes where the agent MERGES attributes from distinct facts.

    Conflation = agent says "Dr. Rivera confirmed penicillin allergy" when
    Dr. Delacroix confirmed it and Dr. Rivera is the dentist. The agent
    mixed attributes from two separate entries.

    Each probe_response has:
        - "entity_keyword": the specific entity being asked about (e.g., "Delacroix")
        - "correct_attribute": the correct attribute for that entity (e.g., "penicillin")
        - "wrong_attributes": attributes from OTHER entities that could be conflated
        - "response_text": the agent's actual response

    Returns float in [0, 1]. 0.0 = never conflates. 1.0 = always mixes entities.
    """
    if not probe_responses:
        return 0.0

    conflation_count = 0
    for pr in probe_responses:
        resp = pr.get("response_text", "").lower()
        entity = pr.get("entity_keyword", "").lower()
        correct_attr = pr.get("correct_attribute", "").lower()
        wrong_attrs = [a.lower() for a in pr.get("wrong_attributes", [])]

        # Check if entity is mentioned
        entity_mentioned = entity in resp if entity else False

        # Conflation: entity is mentioned WITH a wrong attribute
        if entity_mentioned:
            has_wrong = any(wa in resp for wa in wrong_attrs if wa)
            has_correct = correct_attr in resp if correct_attr else False

            if has_wrong and not has_correct:
                conflation_count += 1

    return conflation_count / len(probe_responses)


def compute_specificity_score(
    response_text: str,
    gold_keywords: list[str],
) -> float:
    """
    How SPECIFIC is the agent's response compared to the gold answer?

    Measures the fraction of specific values (numbers, names, dates) from
    the gold answer that appear in the agent's response. A vague answer
    ("your budget is substantial") scores low; a precise answer ("your
    Q1 budget is $340,000") scores high.

    Args:
        response_text: the agent's response
        gold_keywords: list of specific values that should appear

    Returns:
        float in [0, 1]. 1.0 = all specific values cited. 0.0 = completely vague.
    """
    if not gold_keywords:
        return 1.0

    resp_lower = response_text.lower()
    hits = sum(1 for kw in gold_keywords if kw.lower() in resp_lower)
    return hits / len(gold_keywords)


def compute_version_aware_ufr(
    probe_responses: list[dict],
    workspace_text: str,
) -> float:
    """
    Version-aware Utilization Failure Rate.

    Counts probes where the fact's CURRENT keywords are in the workspace
    but NOT in the agent's response. Excludes false positives from outdated
    keywords (old values that were updated).

    Returns float in [0, 1]. Higher = more utilization failures.
    """
    if not probe_responses:
        return 0.0

    ws_lower = workspace_text.lower()
    uf_count = 0
    measurable = 0

    for pr in probe_responses:
        current_kws = pr.get("current_keywords", pr.get("keywords", []))
        resp = pr.get("response_text", "").lower()

        in_workspace = any(kw.lower() in ws_lower for kw in current_kws)
        in_response = any(kw.lower() in resp for kw in current_kws)

        if in_workspace:
            measurable += 1
            if not in_response:
                uf_count += 1

    return uf_count / measurable if measurable > 0 else 0.0


def compute_reasoning_depth(
    response_text: str,
    required_facts: list[dict],
) -> float:
    """
    How many required facts does the agent correctly COMBINE in its response?

    For multi-step reasoning tasks ("Given Q3 budget, vendor costs, and hiring,
    how much is left?"), the agent must combine N facts. Reasoning depth =
    fraction of required facts that appear in the response.

    As context grows, agents tend to shortcut reasoning — answering with
    fewer facts than required.

    Args:
        response_text: the agent's response
        required_facts: list of {"keywords": [str]} — facts that should ALL
            be referenced in a complete answer

    Returns:
        float in [0, 1]. 1.0 = all required facts combined. 0.0 = none referenced.
    """
    if not required_facts:
        return 1.0

    resp_lower = response_text.lower()
    facts_used = 0
    for fact in required_facts:
        keywords = fact.get("keywords", [])
        if keywords and any(kw.lower() in resp_lower for kw in keywords):
            facts_used += 1

    return facts_used / len(required_facts)


def compute_source_attribution_accuracy(
    response_text: str,
    gold_source: dict,
) -> float:
    """
    Can the agent correctly say WHERE/WHEN/WHO for a fact?

    Tests whether the agent attributes facts to the correct source
    (person, date, meeting). As memory grows, attribution degrades
    because more sources contain overlapping information.

    Args:
        response_text: the agent's response
        gold_source: {
            "who_keywords": ["Sarah Chen", "PM"],     # who made the decision
            "when_keywords": ["March 15", "2026-03"],  # when it was made
            "where_keywords": ["kickoff meeting"],     # context of the decision
        }

    Returns:
        float in [0, 1]:
        1.0 = all attribution dimensions correct
        0.67 = 2 of 3 correct
        0.33 = 1 of 3 correct
        0.0 = no attribution
    """
    resp_lower = response_text.lower()
    dimensions = 0
    correct = 0

    for dim in ["who_keywords", "when_keywords", "where_keywords"]:
        kws = gold_source.get(dim, [])
        if kws:
            dimensions += 1
            if any(kw.lower() in resp_lower for kw in kws):
                correct += 1

    return correct / dimensions if dimensions > 0 else 1.0
