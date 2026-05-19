"""
agingbench/metrics/dependency_scorer.py — Post-hoc dependency-aware aging metrics.

Computes additional metrics by joining runner output (session_results) with
the dependency graph extracted from the generator. Zero runner modifications
needed — this is pure post-processing.

Metrics:
  - chain_recall_by_version_depth: recall as a function of the longest version
        chain among a probe's facts (alias: chain_recall_by_depth, kept for
        backward compatibility with consumers of older dependency_metrics.json
        files; both keys are emitted with identical values).
  - chain_recall_by_session_span: recall as a function of the session range
        of a probe's facts (max(s) - min(s)). Captures temporal-span difficulty
        independent of revision frequency. New in 2026-05-03.
  - version_accuracy: fraction of version-test queries citing the latest value.
  - interference_resistance: fraction of interference probes with correct entity.
  - per_hop_failure_analysis: which hop in a chain fails first.
"""

from __future__ import annotations

from typing import Optional


def score_dependency_chain(
    session_results: list[dict],
    dependency_graph: dict,
) -> dict:
    """
    Compute dependency-aware aging metrics.

    Parameters
    ----------
    session_results : list[dict]
        Per-session output from any runner. Must contain at least:
        - "session": int
        - "task_score" or "task_accuracy" or "recall_rate": float
        - "task_keywords_found" or "keywords_found": list[str] (optional)

    dependency_graph : dict
        Output of FactGraph.export(). Contains:
        - "tasks": {task_id: {session, depends_on_facts, chain_depth, ...}}
        - "facts": {fact_id: {versions: [...], ...}}
        - "interference_map": [{shared_term, fact_ids, values}]

    Returns
    -------
    dict with:
        chain_recall_by_depth, version_accuracy, interference_resistance,
        per_hop_analysis, summary
    """
    tasks = dependency_graph.get("tasks", {})
    facts = dependency_graph.get("facts", {})
    interference = dependency_graph.get("interference_map", [])

    # Build session → result lookup
    session_lookup = {}
    for r in session_results:
        s = r.get("session", r.get("session_id", -1))
        session_lookup[s] = r

    # chain_recall_by_version_depth is the canonical name (clarifies the
    # bucketing axis is version chain length). chain_recall_by_depth is an
    # alias kept for backward compatibility.
    crv = chain_recall_by_version_depth(session_lookup, tasks)
    result = {
        "chain_recall_by_version_depth": crv,
        "chain_recall_by_depth": crv,  # alias; same values
        "chain_recall_by_session_span": chain_recall_by_session_span(
            session_lookup, tasks
        ),
        "version_accuracy": version_accuracy(session_lookup, tasks, facts),
        "interference_resistance": interference_resistance(
            session_lookup, tasks, interference
        ),
        "per_hop_analysis": per_hop_failure_analysis(session_lookup, tasks, facts),
        "forget_accuracy": forget_accuracy(session_results, dependency_graph),
        "summary": dependency_graph.get("summary", {}),
    }
    # Per-session trajectories support Table 2 mechanism claims that the
    # aggregate scalars cannot visualize (revision, interference over time).
    result["version_accuracy_per_session"] = version_accuracy_per_session(
        session_lookup, tasks, facts
    )
    result["interference_resistance_per_session"] = (
        interference_resistance_per_session(session_lookup, tasks, interference)
    )
    result["forget_accuracy_per_session"] = forget_accuracy_per_session(
        session_results, dependency_graph
    )
    # Include accumulator metrics if accumulators exist
    if dependency_graph.get("accumulators"):
        result["accumulator_metrics"] = score_accumulator(
            session_results, dependency_graph
        )
    return result


def chain_recall_by_version_depth(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
) -> dict[int, float]:
    """
    Recall as a function of version-chain depth: the longest version chain
    among the facts a probe depends on. Bucketed by ``task['chain_depth']``
    which the FactGraph computes as ``max_i len(version_chain(f_i))``.

    Expected pattern: depth=1 (no superseded facts) → higher recall;
    depth=3 (deeply revised facts) → lower recall.

    NOTE: this metric only varies when ``update_rate > 0`` produces non-trivial
    version chains; with ``update_rate=0`` all probes have depth=1 and the
    metric returns a single bucket. Use ``chain_recall_by_session_span`` for a
    bucketing axis that responds to compression/dependency-density pressure
    instead of revision pressure.
    """
    depth_scores: dict[int, list[float]] = {}

    for task_id, task in tasks.items():
        if task.get("dependency_type", "standalone") == "standalone":
            continue

        session = task["session"]
        result = session_lookup.get(session)
        if result is None:
            continue

        score = _extract_score(result)
        depth = task.get("chain_depth", 1)

        depth_scores.setdefault(depth, []).append(score)

    return {
        depth: round(sum(scores) / len(scores), 4)
        for depth, scores in sorted(depth_scores.items())
        if scores
    }


# Backward-compatible alias for the renamed function (legacy callers can still
# import chain_recall_by_depth). Output is identical.
def chain_recall_by_depth(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
) -> dict[int, float]:
    """Alias for ``chain_recall_by_version_depth``; preserved for callers of
    the original metric name. Returns the same values."""
    return chain_recall_by_version_depth(session_lookup, tasks)


def chain_recall_by_session_span(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
) -> dict[int, float]:
    """
    Recall as a function of session-span: the range of source sessions
    ``max(s) - min(s)`` over the facts a probe depends on. A span of 0 means
    all source facts come from the same session; a span of 5 means the probe
    requires combining facts that were introduced 5 sessions apart.

    Unlike ``chain_recall_by_version_depth``, this responds to compression
    aging directly (longer-span dependencies must survive more compaction
    passes), and it varies even when ``update_rate=0``. Useful as a pressure-
    control metric for ``dependency_density`` or ``max_chain_depth`` sweeps
    where version chains are not the primary axis under test.
    """
    span_scores: dict[int, list[float]] = {}

    for task_id, task in tasks.items():
        if task.get("dependency_type", "standalone") == "standalone":
            continue

        deps_sessions = task.get("depends_on_sessions") or []
        if not deps_sessions:
            continue
        span = int(max(deps_sessions) - min(deps_sessions))

        session = task["session"]
        result = session_lookup.get(session)
        if result is None:
            continue

        score = _extract_score(result)
        span_scores.setdefault(span, []).append(score)

    return {
        span: round(sum(scores) / len(scores), 4)
        for span, scores in sorted(span_scores.items())
        if scores
    }


def version_accuracy(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
    facts: dict[str, dict],
) -> float:
    """
    Fraction of version-test queries where the agent cited the LATEST version.

    A version-test query is a "trend" dependency where a fact has been updated.
    The agent should cite the current value, not the original.
    """
    n_version_tests = 0
    n_correct = 0

    for task_id, task in tasks.items():
        if task.get("dependency_type") != "trend":
            continue

        session = task["session"]
        result = session_lookup.get(session)
        if result is None:
            continue

        # Check if any required version > 1
        versions_req = task.get("fact_versions_required", {})
        has_version_test = any(v > 1 for v in versions_req.values())
        if not has_version_test:
            continue

        n_version_tests += 1
        score = _extract_score(result)
        if score > 0.5:  # at least partial credit
            n_correct += 1

    if n_version_tests == 0:
        return 1.0  # no version tests, vacuously correct

    return round(n_correct / n_version_tests, 4)


def version_accuracy_per_session(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
    facts: dict[str, dict],
) -> dict[int, float]:
    """Per-session version_accuracy trajectory.

    For each session that contains at least one version-test query
    (trend dependency with a required version > 1), compute the fraction
    of such queries the agent answered with the latest value.
    Sessions with no version tests are omitted from the trajectory.
    """
    by_session_tot: dict[int, int] = {}
    by_session_cor: dict[int, int] = {}

    for _, task in tasks.items():
        if task.get("dependency_type") != "trend":
            continue
        versions_req = task.get("fact_versions_required", {})
        if not any(v > 1 for v in versions_req.values()):
            continue
        session = task["session"]
        result = session_lookup.get(session)
        if result is None:
            continue
        by_session_tot[session] = by_session_tot.get(session, 0) + 1
        if _extract_score(result) > 0.5:
            by_session_cor[session] = by_session_cor.get(session, 0) + 1

    return {
        s: round(by_session_cor.get(s, 0) / by_session_tot[s], 4)
        for s in sorted(by_session_tot)
    }


def interference_resistance(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
    interference_map: list[dict],
) -> float:
    """
    Fraction of tasks involving confusable entities where the agent
    retrieved the correct value (not the confusable one).

    Higher = agent resists interference better.
    """
    if not interference_map:
        return 1.0  # no interference, vacuously resistant

    # Build set of facts involved in interference
    interference_fact_ids = set()
    for pair in interference_map:
        for fid in pair.get("fact_ids", []):
            interference_fact_ids.add(fid)

    n_interference_tasks = 0
    n_correct = 0

    for task_id, task in tasks.items():
        deps = set(task.get("depends_on_facts", []))
        if not deps.intersection(interference_fact_ids):
            continue

        session = task["session"]
        result = session_lookup.get(session)
        if result is None:
            continue

        n_interference_tasks += 1
        score = _extract_score(result)
        if score > 0.5:
            n_correct += 1

    if n_interference_tasks == 0:
        return 1.0

    return round(n_correct / n_interference_tasks, 4)


def interference_resistance_per_session(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
    interference_map: list[dict],
) -> dict[int, float]:
    """Per-session interference_resistance trajectory.

    Sessions with no interference-involved tasks are omitted.
    """
    if not interference_map:
        return {}

    interference_fact_ids = set()
    for pair in interference_map:
        for fid in pair.get("fact_ids", []):
            interference_fact_ids.add(fid)

    by_session_tot: dict[int, int] = {}
    by_session_cor: dict[int, int] = {}

    for _, task in tasks.items():
        deps = set(task.get("depends_on_facts", []))
        if not deps.intersection(interference_fact_ids):
            continue
        session = task["session"]
        result = session_lookup.get(session)
        if result is None:
            continue
        by_session_tot[session] = by_session_tot.get(session, 0) + 1
        if _extract_score(result) > 0.5:
            by_session_cor[session] = by_session_cor.get(session, 0) + 1

    return {
        s: round(by_session_cor.get(s, 0) / by_session_tot[s], 4)
        for s in sorted(by_session_tot)
    }


def per_hop_failure_analysis(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
    facts: dict[str, dict],
) -> dict:
    """
    For multi-hop dependency tasks: analyze which hop fails first.

    Returns:
        hop_recall: {hop_index: recall_rate}  (0=first dependency, 1=second, ...)
        common_failure_pattern: description of the most common failure
    """
    hop_scores: dict[int, list[float]] = {}

    for task_id, task in tasks.items():
        deps = task.get("depends_on_facts", [])
        if len(deps) < 2:
            continue

        session = task["session"]
        result = session_lookup.get(session)
        if result is None:
            continue

        # Check which dependency keywords appear in the output
        found_keywords = set()
        for kw_field in ["task_keywords_found", "keywords_found"]:
            if kw_field in result:
                found_keywords.update(
                    kw.lower() for kw in result[kw_field]
                )

        if not found_keywords:
            # Can't do per-hop analysis without keyword-level detail
            continue

        for hop_idx, fact_id in enumerate(deps):
            # Find the fact's keywords
            fact_kws = _get_fact_keywords(fact_id, facts)
            if not fact_kws:
                continue

            # Check if any of this fact's keywords were found
            hop_hit = any(kw.lower() in found_keywords for kw in fact_kws)
            hop_scores.setdefault(hop_idx, []).append(1.0 if hop_hit else 0.0)

    hop_recall = {
        hop: round(sum(scores) / len(scores), 4)
        for hop, scores in sorted(hop_scores.items())
        if scores
    }

    # Determine common failure pattern
    pattern = "insufficient_data"
    if hop_recall:
        sorted_hops = sorted(hop_recall.items())
        if len(sorted_hops) >= 2:
            if sorted_hops[0][1] > sorted_hops[-1][1]:
                pattern = "later_hops_fail_first"
            elif sorted_hops[0][1] < sorted_hops[-1][1]:
                pattern = "earlier_hops_fail_first"
            else:
                pattern = "uniform_failure"

    return {
        "hop_recall": hop_recall,
        "common_failure_pattern": pattern,
    }


def score_execution_drift(
    session_results: list[dict],
    output_dependency_pairs: list[dict],
) -> dict:
    """
    Measure plan-execution drift for S7 output dependency tasks.

    The producer task asks the agent to compute a deterministic value and save it
    to a workspace file.  The consumer task (distance blocks later) asks the agent
    to retrieve that value from the file.

    Drift = producer_accuracy - consumer_accuracy, grouped by block distance.
    A positive drift means the agent computed the value correctly but failed to
    retrieve it later — indicating the workspace write either didn't happen or
    was overwritten/lost.

    Parameters
    ----------
    session_results : list[dict]
        Per-block results from S7Runner.  Each entry must include
        "task_scores": {task_id: score}.
    output_dependency_pairs : list[dict]
        From S7Generator.generate()["output_dependency_pairs"].

    Returns
    -------
    dict with:
        drift_by_distance: {distance: {producer_accuracy, consumer_accuracy,
                                       drift, n_pairs}}
        overall_execution_drift: float
        n_pairs_total: int
    """
    if not output_dependency_pairs:
        return {
            "drift_by_distance": {},
            "overall_execution_drift": 0.0,
            "n_pairs_total": 0,
        }

    # Build flat task_id → score map from all session results
    task_scores: dict[str, float] = {}
    for sr in session_results:
        for tid, score in sr.get("task_scores", {}).items():
            task_scores[tid] = float(score)

    # Group pairs by distance
    by_dist: dict[int, dict] = {}
    for pair in output_dependency_pairs:
        pid = pair["producer_task_id"]
        cid = pair["consumer_task_id"]
        dist = pair["distance"]
        ps = task_scores.get(pid)
        cs = task_scores.get(cid)
        if ps is None or cs is None:
            continue
        bucket = by_dist.setdefault(dist, {"producers": [], "consumers": []})
        bucket["producers"].append(ps)
        bucket["consumers"].append(cs)

    drift_by_distance: dict[str, dict] = {}
    for dist, data in sorted(by_dist.items()):
        prod_acc = sum(data["producers"]) / len(data["producers"])
        cons_acc = sum(data["consumers"]) / len(data["consumers"])
        drift_by_distance[str(dist)] = {
            "producer_accuracy": round(prod_acc, 4),
            "consumer_accuracy": round(cons_acc, 4),
            "drift": round(max(0.0, prod_acc - cons_acc), 4),
            "n_pairs": len(data["producers"]),
        }

    total_drift = sum(d["drift"] for d in drift_by_distance.values())
    n_buckets = len(drift_by_distance)

    return {
        "drift_by_distance": drift_by_distance,
        "overall_execution_drift": round(total_drift / n_buckets, 4) if n_buckets else 0.0,
        "n_pairs_total": sum(d["n_pairs"] for d in drift_by_distance.values()),
    }


def forget_accuracy(
    session_results: list[dict],
    dependency_graph: dict,
) -> float:
    """
    Fraction of post-invalidation sessions where the agent correctly does NOT
    cite invalidated facts.

    For each invalidated fact, scans all session results AFTER the invalidation
    session.  If the invalidated keywords appear in the agent's output, that is
    a failure (the agent cited retracted information).

    Returns 1.0 if no invalidated facts exist (vacuous truth).
    """
    facts = dependency_graph.get("facts", {})

    # Collect all invalidated facts: {fact_id: (invalidated_at_session, keywords)}
    invalidated: dict[str, tuple[int, list[str]]] = {}
    for root_id, fact_data in facts.items():
        for version in fact_data.get("versions", []):
            inv_at = version.get("invalidated_at")
            if inv_at is not None:
                kws = version.get("keywords", [])
                if kws:
                    invalidated[version["fact_id"]] = (inv_at, kws)

    if not invalidated:
        return 1.0

    # Build session → result lookup
    session_lookup = {}
    for r in session_results:
        s = r.get("session", r.get("session_id", -1))
        session_lookup[s] = r

    n_checks = 0
    n_correct = 0  # correct = agent did NOT cite the invalidated keywords
    n_skipped = 0  # no output text available — cannot judge

    # Fields that scenarios actually populate, in order of preference.
    # task_outputs_text and agent_outputs are added by S2/S3/S4 runners.
    # S6/S7 use keyword lists (task_keywords_found, probe_details) — we
    # check those too. The previous implementation only looked at
    # response_text/task_output/agent_output, which no scenario populated,
    # so it defaulted every check to "correct" and forget_accuracy was
    # silently saturated at 1.0.
    text_fields = (
        "task_outputs_text", "agent_outputs", "response_text",
        "task_output", "agent_output",
    )
    keyword_list_fields = (
        "task_keywords_found", "keywords_found",
    )

    def _gather_session_text(result: dict) -> str:
        """Concatenate every text-bearing field we can find in a result."""
        parts: list[str] = []
        for key in text_fields:
            v = result.get(key)
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        for vv in item.values():
                            if isinstance(vv, str):
                                parts.append(vv)
        for key in keyword_list_fields:
            v = result.get(key)
            if isinstance(v, list):
                parts.extend(str(x) for x in v)
        # probe_details / precision_per_probe carry per-probe text
        for probe_field in ("probe_details", "precision_per_probe", "compounding_per_probe"):
            v = result.get(probe_field)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for vv in item.values():
                            if isinstance(vv, str):
                                parts.append(vv)
                            elif isinstance(vv, list):
                                parts.extend(str(x) for x in vv if isinstance(x, (str, int, float)))
        return " ".join(parts).lower()

    for fact_id, (inv_session, keywords) in invalidated.items():
        for s, result in session_lookup.items():
            if s <= inv_session:
                continue
            n_checks += 1
            haystack = _gather_session_text(result)
            if not haystack:
                # No output text discoverable → can't judge → SKIP rather
                # than counting as correct (which silently saturated the
                # metric at 1.0).
                n_skipped += 1
                n_checks -= 1
                continue
            cited = any(kw.lower() in haystack for kw in keywords if isinstance(kw, str))
            if not cited:
                n_correct += 1

    if n_checks == 0:
        # All checks skipped (no scenario stores text yet). Return None
        # so the caller can distinguish "not measurable" from "perfect".
        return None if n_skipped > 0 else 1.0

    return round(n_correct / n_checks, 4)


def forget_accuracy_per_session(
    session_results: list[dict],
    dependency_graph: dict,
) -> dict[int, float]:
    """Per-session forget_accuracy trajectory.

    For each session after some fact became invalidated, what fraction of
    invalidated facts did the agent correctly NOT cite? Sessions before any
    invalidation event are omitted.
    """
    facts = dependency_graph.get("facts", {})

    invalidated: dict[str, tuple[int, list[str]]] = {}
    for _, fact_data in facts.items():
        for version in fact_data.get("versions", []):
            inv_at = version.get("invalidated_at")
            if inv_at is not None:
                kws = version.get("keywords", [])
                if kws:
                    invalidated[version["fact_id"]] = (inv_at, kws)

    if not invalidated:
        return {}

    session_lookup = {}
    for r in session_results:
        s = r.get("session", r.get("session_id", -1))
        session_lookup[s] = r

    # Reuse the in-scope _gather_session_text from forget_accuracy()? It is
    # defined as a local closure there. Inline a small copy to keep this
    # function standalone.
    text_fields = (
        "task_outputs_text", "agent_outputs", "response_text",
        "task_output", "agent_output",
    )
    keyword_list_fields = ("task_keywords_found", "keywords_found")

    def _gather(result: dict) -> str:
        parts: list[str] = []
        for key in text_fields:
            v = result.get(key)
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        for vv in item.values():
                            if isinstance(vv, str):
                                parts.append(vv)
        for key in keyword_list_fields:
            v = result.get(key)
            if isinstance(v, list):
                parts.extend(str(x) for x in v)
        return " ".join(parts).lower()

    trajectory: dict[int, float] = {}
    for s in sorted(session_lookup):
        result = session_lookup[s]
        haystack = _gather(result)
        if not haystack:
            continue
        n_applicable = 0
        n_correct = 0
        for _, (inv_session, keywords) in invalidated.items():
            if s <= inv_session:
                continue
            n_applicable += 1
            cited = any(kw.lower() in haystack for kw in keywords if isinstance(kw, str))
            if not cited:
                n_correct += 1
        if n_applicable > 0:
            trajectory[s] = round(n_correct / n_applicable, 4)
    return trajectory


def score_accumulator(
    session_results: list[dict],
    dependency_graph: dict,
) -> dict:
    """
    Score accuracy of accumulator (Ledger-QA-style derived value) tracking.

    Compares the agent's numeric answers for accumulator probes against the
    ground-truth values computed from the delta history.

    Parameters
    ----------
    session_results : list[dict]
        Per-session output. Each may contain "accumulator_probes" results.
    dependency_graph : dict
        Must contain "accumulators" key from FactGraph.export().

    Returns
    -------
    dict with:
        accumulator_errors: {session: absolute_error}
        error_source_sessions: [session indices where missed deltas were detected]
        compounding_detected: bool (True if error grows monotonically)
        mean_error: float
    """
    accumulators = dependency_graph.get("accumulators", {})
    if not accumulators:
        return {
            "accumulator_errors": {},
            "error_source_sessions": [],
            "compounding_detected": False,
            "mean_error": 0.0,
        }

    import re

    errors: dict[int, float] = {}
    for sr in session_results:
        probes = sr.get("accumulator_probes", [])
        for probe in probes:
            session = probe.get("session", sr.get("session", -1))
            gold = probe.get("gold_value")
            response = probe.get("response_text", "")
            if gold is None or not response:
                continue

            # Extract first number from response
            nums = re.findall(r"[\-]?\d[\d,]*\.?\d*", response.replace(",", ""))
            agent_value = float(nums[0]) if nums else None
            if agent_value is not None:
                errors[session] = abs(agent_value - gold)
            else:
                errors[session] = abs(gold)  # total miss

    # Detect compounding: error grows across sessions
    sorted_errors = sorted(errors.items())
    compounding = False
    if len(sorted_errors) >= 3:
        vals = [e for _, e in sorted_errors]
        compounding = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    # Identify error source sessions: where error first becomes non-zero
    error_sources = []
    for s, err in sorted_errors:
        if err > 0 and (not error_sources or err > errors.get(sorted_errors[0][0], 0)):
            error_sources.append(s)

    mean_err = sum(errors.values()) / len(errors) if errors else 0.0

    return {
        "accumulator_errors": {str(s): round(e, 2) for s, e in sorted(errors.items())},
        "error_source_sessions": error_sources[:5],
        "compounding_detected": compounding,
        "mean_error": round(mean_err, 2),
    }

# NOTE: attribution_report() has been removed. Error partitioning is now
# handled by agingbench.diagnostics.partitioner.partition_errors() using the
# P1/P2/P3 framework (§5.2).


# ------------------------------------------------------------------ helpers

def _extract_score(result: dict) -> float:
    """Extract the primary score from a session result dict."""
    for key in ["task_score", "task_accuracy", "recall_rate", "recall_accuracy"]:
        if key in result:
            val = result[key]
            return float(val) if val is not None else 0.0
    return 0.0


def _get_fact_keywords(fact_id: str, facts: dict) -> list[str]:
    """Get keywords for a fact ID, searching through version chains."""
    # Direct lookup in facts export (keyed by root ID)
    for root_id, fact_data in facts.items():
        for version in fact_data.get("versions", []):
            if version.get("fact_id") == fact_id:
                return version.get("keywords", [])
    return []
