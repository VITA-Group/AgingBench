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

import re as _re
from typing import Optional


def _kw_in_text(kw: str, text: str) -> bool:
    """Keyword presence that won't fire on a longer number — e.g. the stale
    keyword '43' must NOT match inside the current value '143,751'. Numeric
    keywords match only with non-digit flanks; for word keywords the guards are
    harmless. Fixes forget_accuracy substring-collision false positives that
    capped a clean agent below 1.0."""
    return _re.search(r"(?<!\d)" + _re.escape(kw) + r"(?!\d)", text) is not None


def _session_haystack(result: dict):
    """Lowercased agent output for a session (same field preference as
    forget_accuracy). Returns None if no output text is available."""
    for f in ("task_outputs_text", "agent_outputs", "response_text",
              "task_output", "agent_output"):
        v = result.get(f)
        if isinstance(v, str) and v.strip():
            return v.lower()
    for f in ("task_keywords_found", "keywords_found"):
        v = result.get(f)
        if isinstance(v, (list, tuple)) and v:
            return " ".join(str(x).lower() for x in v)
    return None


def _fact_kw_map(facts: dict) -> dict:
    """{fact_id: (version_int, [keywords])} over all version chains."""
    m: dict = {}
    for _root, fd in (facts or {}).items():
        for v in fd.get("versions", []):
            m[v.get("fact_id")] = (v.get("version", 1), v.get("keywords", []) or [])
    return m


def _trend_gold_and_stale(task: dict, facts: dict):
    """For a trend task (depends_on_facts = [old, current]), return
    (gold_keywords=current version, stale_keywords=older values dropped in the
    current version). Used to gold-score whether the agent cited the current
    value and avoided the stale one."""
    m = _fact_kw_map(facts)
    entries = [m[fid] for fid in task.get("depends_on_facts", []) if fid in m]
    if not entries:
        return [], []
    maxv = max(v for v, _ in entries)
    gold, older = [], []
    for v, kws in entries:
        (gold if v == maxv else older).extend(kws)
    goldset = set(gold)
    stale = [k for k in older if k not in goldset]
    return gold, stale


def _dep_gold_recall(task: dict, result: dict, facts: dict):
    """Gold recall for a dependency task: fraction of the task's depended facts
    whose value the agent cited in its session output. Returns None when no gold
    or no output text is available, so the caller can fall back to the headline
    proxy. Replaces reusing the session-wide _extract_score for chain recall."""
    hay = _session_haystack(result)
    if hay is None:
        return None
    kwmap = _fact_kw_map(facts or {})
    golds = [kwmap[f][1] for f in task.get("depends_on_facts", [])
             if f in kwmap and kwmap[f][1]]
    if not golds:
        return None
    hits = sum(1 for kws in golds
               if any(_kw_in_text(k.lower(), hay) for k in kws if k))
    return hits / len(golds)


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
        - a primary correctness score under one of the keys recognized by
          ``_extract_score`` (task_score / task_accuracy / recall_rate /
          recall_accuracy / dep_recall / query_accuracy / keyword_m /
          constraint_precision): float
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
    crv = chain_recall_by_version_depth(session_lookup, tasks, facts)
    result = {
        "chain_recall_by_version_depth": crv,
        "chain_recall_by_depth": crv,  # alias; same values
        "chain_recall_by_session_span": chain_recall_by_session_span(
            session_lookup, tasks, facts
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
    # Forced-answer interference binding (correct vs confused vs miss). Unlike
    # interference_resistance (which reuses the session headline score), this
    # measures actual confusable mis-binding from dedicated probes.
    result["interference_binding"] = score_interference_binding(session_results)
    return result


def chain_recall_by_version_depth(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
    facts: Optional[dict] = None,
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

        g = _dep_gold_recall(task, result, facts) if facts is not None else None
        score = g if g is not None else _extract_score(result)
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
    facts: Optional[dict] = None,
) -> dict[int, float]:
    """Alias for ``chain_recall_by_version_depth``; preserved for callers of
    the original metric name. Returns the same values."""
    return chain_recall_by_version_depth(session_lookup, tasks, facts)


def chain_recall_by_session_span(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
    facts: Optional[dict] = None,
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

        g = _dep_gold_recall(task, result, facts) if facts is not None else None
        score = g if g is not None else _extract_score(result)
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

    Preferred scoring path (faithful):
      When session_lookup[s]["trend_probe_results"] exists, each entry is the
      per-probe verdict from the validator — already accounting for forbidden
      pre-revision keywords. We aggregate those binary scores directly.

    Fallback scoring path (proxy, pre-2026-05 behavior):
      When trend_probe_results is absent, we use _extract_score(result) as a
      session-wide correctness proxy. This was the original implementation;
      it conflates revision accuracy with general keyword recall and is kept
      only so older runners (and runs without per-probe data) keep producing
      a number. Faithful path is preferred whenever available.
    """
    n_version_tests = 0
    n_correct = 0
    used_faithful = False

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

        # Preferred: per-probe faithful score
        trend_results = result.get("trend_probe_results") or []
        if trend_results:
            used_faithful = True
            for tr in trend_results:
                n_version_tests += 1
                if tr.get("score", 0.0) > 0.5:
                    n_correct += 1
            continue

        # Gold-based scan (no per-probe capture available): reconstruct the
        # current (gold) and stale (forbidden) values from the version chain
        # and check the agent's session output — cited the CURRENT value and
        # NOT the stale one. This replaces the old session-headline proxy
        # (_extract_score), which conflated revision with general recall.
        # Caveat: scans concatenated session text, so a gold value appearing
        # elsewhere can false-pass; still far better than an unrelated headline.
        gold_kws, stale_kws = _trend_gold_and_stale(task, facts)
        hay = _session_haystack(result)
        if gold_kws and hay is not None:
            n_version_tests += 1
            cited_current = any(_kw_in_text(k.lower(), hay) for k in gold_kws if k)
            cited_stale = any(_kw_in_text(k.lower(), hay) for k in stale_kws if k)
            if cited_current and not cited_stale:
                n_correct += 1
        else:
            # Last resort (no gold or no output text): legacy headline proxy.
            n_version_tests += 1
            if _extract_score(result) > 0.5:
                n_correct += 1

    if n_version_tests == 0:
        # No qualifying version tests → NO COVERAGE. Return None rather than a
        # vacuous 1.0, which falsely reads as "perfect revision tracking" in
        # tables/plots (e.g. S1/S3 emit zero trend tasks). None = not measured.
        return None

    return round(n_correct / n_version_tests, 4)


def version_accuracy_per_session(
    session_lookup: dict[int, dict],
    tasks: dict[str, dict],
    facts: dict[str, dict],
) -> dict[int, float]:
    """Per-session version_accuracy trajectory.

    Mirrors the priority order of ``version_accuracy``:
      1. session has trend_probe_results → aggregate those per-probe verdicts
      2. otherwise → fall back to the session-wide _extract_score proxy
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

        trend_results = result.get("trend_probe_results") or []
        if trend_results:
            for tr in trend_results:
                by_session_tot[session] = by_session_tot.get(session, 0) + 1
                if tr.get("score", 0.0) > 0.5:
                    by_session_cor[session] = by_session_cor.get(session, 0) + 1
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
        return None  # no interference pairs → NO COVERAGE (not vacuous 1.0)

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
        return None  # no interference-linked tasks fired → NO COVERAGE

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


# Phrases that, when adjacent to a retracted keyword, indicate the agent is
# correctly flagging the value as no-longer-current (rather than asserting it
# as truth). Detected within a small window around each keyword occurrence;
# if every occurrence is in retraction context, the citation does not count
# as a forget-failure.
_RETRACTION_MARKERS = (
    "previously", "previous", "used to be", "used to",
    "no longer", "no more", "outdated", "obsolete",
    "former ", "formerly", "old value", "old number",
    "revised", "revised to", "updated to", "changed to",
    "changed from", "was set to", "originally",
    "deprecated", "before the update", "superseded",
    "was ", "now ",  # e.g. "the figure WAS $23,800" / "is NOW $26,100"
)


def _is_retracted_in_context(haystack_lower: str, kw_lower: str,
                              window: int = 60) -> bool:
    """Return True if EVERY occurrence of `kw_lower` in `haystack_lower`
    has a retraction marker within `window` characters on either side.

    Caller treats a True return as "the agent properly framed the old value
    as retracted, so this mention shouldn't be counted as a forget-failure".
    Returns False if any occurrence appears in plain (non-retracted)
    context — that single occurrence is enough to fail the test.
    """
    pos = 0
    occurrences: list[int] = []
    while True:
        idx = haystack_lower.find(kw_lower, pos)
        if idx < 0:
            break
        occurrences.append(idx)
        pos = idx + 1
    if not occurrences:
        return False  # no mention at all, caller handles separately
    for idx in occurrences:
        start = max(0, idx - window)
        end = min(len(haystack_lower), idx + len(kw_lower) + window)
        ctx = haystack_lower[start:end]
        if not any(marker in ctx for marker in _RETRACTION_MARKERS):
            return False  # this occurrence is a bare citation
    return True


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
            # A keyword counts as "cited" only if at least one occurrence is
            # in a non-retraction context. An agent that says
            # "the previous figure of $23,800 has been revised to $26,100"
            # mentions the old keyword but is correctly framing it as
            # retracted; that should NOT count against forget_accuracy.
            cited = False
            for kw in keywords:
                if not isinstance(kw, str):
                    continue
                kw_l = kw.lower()
                if not _kw_in_text(kw_l, haystack):
                    continue
                if _is_retracted_in_context(haystack, kw_l):
                    continue  # all mentions framed as retracted → safe
                cited = True
                break
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
            # A keyword counts as "cited" only if at least one occurrence is
            # in a non-retraction context. An agent that says
            # "the previous figure of $23,800 has been revised to $26,100"
            # mentions the old keyword but is correctly framing it as
            # retracted; that should NOT count against forget_accuracy.
            cited = False
            for kw in keywords:
                if not isinstance(kw, str):
                    continue
                kw_l = kw.lower()
                if not _kw_in_text(kw_l, haystack):
                    continue
                if _is_retracted_in_context(haystack, kw_l):
                    continue  # all mentions framed as retracted → safe
                cited = True
                break
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

            # Extract the agent's numeric answer. Prefer the LAST dollar-
            # prefixed figure — free-form answers like "you spent $43, so
            # $186 remains" put the answer last; taking the FIRST number
            # mis-read preamble/line-item numbers as the answer (spurious error).
            # Fall back to the last bare number, then None.
            dollar = re.findall(r"\$\s*(-?\d[\d,]*\.?\d*)", response)
            if dollar:
                agent_value = float(dollar[-1].replace(",", ""))
            else:
                nums = re.findall(r"-?\d[\d,]*\.?\d*", response.replace(",", ""))
                agent_value = float(nums[-1]) if nums else None
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

def _binding_classify(gold, distractor, response_text: str) -> str:
    """Classify an interference probe response as correct/confused/both/miss.

    Two pathways depending on the gold/distractor type:
      - Numeric path: gold and distractor reduce to ≥2-digit numbers. Uses
        digit-substring with a subset guard so that e.g. gold='42' is not
        spuriously considered present in a response that only mentions
        distractor='42500'.
      - Textual path: gold/distractor are non-numeric (e.g. 'John in marketing'
        vs 'John in finance'). Uses a token-set discriminator: the answer is
        classified by which entity's *distinguishing* tokens (those unique to
        gold vs distractor, excluding shared tokens) appear in the response.
    """
    import re as _re
    resp_text = (response_text or "")

    def _digits(v):
        return _re.sub(r"[^\d]", "", str(v)) if v is not None else ""

    def _word_tokens(v):
        return set(t for t in _re.findall(r"[a-z0-9]+", str(v).lower()) if len(t) >= 2)

    gd, dd = _digits(gold), _digits(distractor)
    resp_d = _digits(resp_text)

    # Numeric path: BOTH gold and distractor have ≥2 digits → digit-substring
    # with subset guard. The subset guard prevents misclassification when one
    # value's digits are a substring of the other's (e.g. "42" ⊂ "42500"):
    # we only credit a value if its digits appear AND are not entirely
    # explainable as a substring of the other value's digits appearing in
    # the response.
    if len(gd) >= 2 and len(dd) >= 2:
        gd_in = gd in resp_d
        dd_in = dd in resp_d
        # Subset adjustment: if dd contains gd (e.g. gd='42', dd='42500'),
        # the gd substring match is only real if it occurs OUTSIDE the dd span.
        if gd_in and gd in dd and dd_in:
            # gd's only appearance might be inside dd. Check by removing dd
            # occurrences and re-testing for gd.
            stripped = resp_d.replace(dd, "")
            gd_in = gd in stripped
        if dd_in and dd in gd and gd_in:
            stripped = resp_d.replace(gd, "")
            dd_in = dd in stripped
        if gd_in and dd_in:
            return "both"
        if gd_in:
            return "correct"
        if dd_in:
            return "confused"
        return "miss"

    # Textual path: at least one side is not a usable number. Use distinguishing
    # tokens to classify. Shared tokens (e.g. "John" appears in both
    # "John-marketing" and "John-finance") are excluded because they don't help.
    g_tokens = _word_tokens(gold)
    d_tokens = _word_tokens(distractor)
    g_only = g_tokens - d_tokens
    d_only = d_tokens - g_tokens
    resp_tokens = _word_tokens(resp_text)
    g_match = bool(g_only & resp_tokens) if g_only else False
    d_match = bool(d_only & resp_tokens) if d_only else False
    # Edge case: if g_only and d_only are both empty (gold == distractor or
    # both reduce to no usable tokens), we cannot classify — return "miss".
    if not g_only and not d_only:
        return "miss"
    if g_match and d_match:
        return "both"
    if g_match:
        return "correct"
    if d_match:
        return "confused"
    return "miss"


def score_interference_binding(session_results: list[dict]) -> dict:
    """Forced-choice interference binding from dedicated probes.

    Each ``interference_probe`` asks for one confusable entity's value (gold),
    with the partner entity's value as the distractor. The agent's answer is
    classified:
      - correct  : contains the gold value (right entity)  — resisted interference
      - confused : contains the distractor only            — TRUE interference
      - both     : contains both values                    — ambiguous dump
      - miss     : contains neither                         — omission / recall failure

    Supports both numeric and textual gold/distractor pairs (delegates to
    ``_binding_classify``). The textual path uses distinguishing-token
    discrimination so name-based confusables (e.g. "John in marketing" vs
    "John in finance") are scored correctly — previously the digit-only path
    silently classified all non-numeric pairs as "miss".

    This separates genuine confusable mis-binding (``confusion_rate``) from
    omission (``miss_rate``) — a distinction ``interference_resistance`` cannot
    make, since it reuses the session's generic headline score.
    """
    correct = confused = both = miss = 0
    per_session: dict[int, list[str]] = {}
    detail: list[dict] = []
    for sr in session_results:
        for p in sr.get("interference_probes", []) or []:
            sess = p.get("session", sr.get("session", -1))
            cls = _binding_classify(
                p.get("gold_value"),
                p.get("distractor_value"),
                p.get("response_text", ""),
            )
            if cls == "correct":
                correct += 1
            elif cls == "confused":
                confused += 1
            elif cls == "both":
                both += 1
            else:
                miss += 1
            per_session.setdefault(sess, []).append(cls)
            detail.append({"session": sess, "task_id": p.get("task_id"),
                           "gold": p.get("gold_value"),
                           "distractor": p.get("distractor_value"), "class": cls})
    total = correct + confused + both + miss
    return {
        "binding_accuracy": round(correct / total, 4) if total else None,
        "confusion_rate": round(confused / total, 4) if total else None,
        "miss_rate": round(miss / total, 4) if total else None,
        "both_rate": round(both / total, 4) if total else None,
        "n_probes": total,
        "per_session": {s: cs for s, cs in sorted(per_session.items())},
        "detail": detail,
    }


# NOTE: attribution_report() has been removed. Error partitioning is now
# handled by agingbench.diagnostics.partitioner.partition_errors() using the
# P1/P2/P3 framework (§5.2).


# ------------------------------------------------------------------ helpers

def _extract_score(result: dict) -> float:
    """Extract the primary per-session correctness score from a result dict.

    The DAG mechanism metrics (version_accuracy, interference_resistance,
    chain_recall_*) use this single [0,1] value as a correctness proxy for the
    session's task. Each scenario runner stores that score under a
    scenario-specific key, so all of them must be recognized here — otherwise
    the metric silently reads 0.0 for every session and collapses to a
    degenerate value (this was the case for S1/S2/S3/S4 before 2026-05).

    Keys are checked in priority order; the first present one wins. The four
    generic keys come first for back-compat with S5/S6 and synthetic test
    fixtures; scenario-specific headline keys follow:
      - dep_recall            → S4 software-engineering (dependency recall)
      - query_accuracy        → S3 knowledge-base
      - keyword_m             → S1 research-literature (keyword recall)
      - constraint_precision  → S2 lifestyle-assistant
    """
    for key in ("task_score", "task_accuracy", "recall_rate", "recall_accuracy",
                "dep_recall", "query_accuracy", "keyword_m", "constraint_precision"):
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
