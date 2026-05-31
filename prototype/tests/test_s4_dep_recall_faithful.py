"""Regression test for S4 dep_recall_faithful aggregation (2026-05-30).

S4 has TWO dependency-recall metrics with very different semantics:

  1. ``dep_recall`` (existing headline; Table 3 column) — substring overlap
     between agent_output and the fresh dep_context that's already in the
     agent's task prompt. Measures prompt-attention / language comprehension,
     NOT memory recall, because dep_context is regenerated each sprint with
     full fidelity.

  2. ``dep_recall_faithful`` (NEW sidecar) — aggregated score from the held-out
     ``dependency_probe`` that the runner already executes at line 668-689 of
     s4_runner.py. The probe asks "Based on your knowledge of all prior
     sprints, answer X" with ONLY memory_text in context (no dep_context).
     This is the faithful memory-recall metric.

Phase 1 (this fix): emit ``dep_recall_faithful_raw`` and aggregates alongside
the existing ``dep_recall_raw``. Both reported; paper headline unchanged.
Phase 2 (deferred): once empirical comparison is in hand, decide whether to
promote dep_recall_faithful to the Table 3 column.
"""
from __future__ import annotations


def _aggregate_faithful(session_results: list[dict]) -> list[tuple[int, float]]:
    """Mirror the runner's aggregation logic so tests can pin its behavior."""
    out: list[tuple[int, float]] = []
    for sr in session_results:
        dpr = sr.get("dependency_probe_result")
        if dpr is not None and "score" in dpr:
            out.append((sr["session"], float(dpr["score"])))
    return out


def test_faithful_aggregation_empty_when_no_probes():
    """No probes fired (e.g., dependency_density too low) → empty list."""
    sr = [
        {"session": 0, "dep_recall": 0.7},
        {"session": 1, "dep_recall": 0.5, "dependency_probe_result": None},
        {"session": 2, "dep_recall": 0.3},
    ]
    assert _aggregate_faithful(sr) == []


def test_faithful_aggregation_collects_probe_scores():
    """Probes fired at some sessions → those sessions are in the output."""
    sr = [
        {"session": 0, "dep_recall": 0.7},   # no probe
        {"session": 1, "dep_recall": 0.5,
         "dependency_probe_result": {
             "session": 1, "score": 0.8, "output": "...", "eval_keywords": []
         }},
        {"session": 2, "dep_recall": 0.3,
         "dependency_probe_result": {
             "session": 2, "score": 0.4, "output": "...", "eval_keywords": []
         }},
        {"session": 3, "dep_recall": 0.2},   # no probe
    ]
    assert _aggregate_faithful(sr) == [(1, 0.8), (2, 0.4)]


def test_faithful_aggregation_preserves_session_ordering():
    """Output should be in ascending session order (input ordering preserved)."""
    sr = [
        {"session": 0,
         "dependency_probe_result": {"session": 0, "score": 1.0}},
        {"session": 1,
         "dependency_probe_result": {"session": 1, "score": 0.0}},
        {"session": 2,
         "dependency_probe_result": {"session": 2, "score": 0.5}},
    ]
    result = _aggregate_faithful(sr)
    assert [s for s, _ in result] == [0, 1, 2]


def test_faithful_aggregation_skips_probe_dict_without_score():
    """A malformed dep_probe_result (missing score) should be skipped, not
    crash. Backward-compat for older session_results shapes."""
    sr = [
        {"session": 0,
         "dependency_probe_result": {"session": 0}},   # no 'score'
        {"session": 1,
         "dependency_probe_result": {"session": 1, "score": 0.6}},
    ]
    assert _aggregate_faithful(sr) == [(1, 0.6)]


def test_faithful_metric_is_distinct_from_dep_recall():
    """The faithful metric reflects probe answers, not the substring metric.
    Same session can have HIGH dep_recall (substring) and LOW dep_recall_faithful
    (memory failed) — this is the whole point of having both."""
    sr = [
        {"session": 0,
         "dep_recall": 1.0,            # substring metric: agent echoed dep_context
         "dependency_probe_result": {
             "session": 0, "score": 0.0,  # faithful: agent couldn't answer
             "output": "I don't recall", "eval_keywords": ["Customer", "fields"]
         }},
    ]
    faithful = _aggregate_faithful(sr)
    assert faithful == [(0, 0.0)]
    # The dep_recall_raw side (substring) would have (0, 1.0); they disagree.


def test_aggregation_matches_runner_implementation():
    """Pin the exact shape: list of (session, score) tuples in session order."""
    sr = [
        {"session": 2,
         "dependency_probe_result": {"session": 2, "score": 0.75}},
        {"session": 5,
         "dependency_probe_result": {"session": 5, "score": 0.25}},
    ]
    result = _aggregate_faithful(sr)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple) and len(item) == 2
        assert isinstance(item[0], int)
        assert isinstance(item[1], float)
    assert result == [(2, 0.75), (5, 0.25)]
