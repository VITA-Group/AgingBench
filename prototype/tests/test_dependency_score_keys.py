"""Regression guard for the DAG mechanism metrics' score-key plumbing.

Background: ``dependency_scorer._extract_score`` maps a session-result dict to
the single [0,1] correctness proxy used by ``version_accuracy``,
``interference_resistance`` and ``chain_recall_*``. Each scenario runner stores
that score under a *scenario-specific* key. A refactor once left
``_extract_score`` recognizing only S5/S6's keys, so S1-S4 silently scored 0.0
for every session and their DAG mechanism metrics collapsed to degenerate
values regardless of model performance. These tests pin every runner's real
key so that regression cannot recur — they assert that a *perfect* run scores a
perfect (non-degenerate) metric, and that a *failing* run still scores 0.

The session-result shapes below mirror the dicts actually appended by the
runners (see runner/sN_runner.py):
  S1 -> "keyword_m"            S4 -> "dep_recall"
  S2 -> "constraint_precision" S5 -> "task_accuracy" / "recall_accuracy"
  S3 -> "query_accuracy"       S6 -> "task_score" / "recall_rate"
"""
import pytest

from agingbench.metrics.dependency_scorer import (
    _extract_score,
    version_accuracy,
    interference_resistance,
    score_dependency_chain,
)

# (label, session-result key used by that runner's headline correctness score)
RUNNER_SCORE_KEYS = [
    ("s1_research_literature", "keyword_m"),
    ("s2_lifestyle_assistant", "constraint_precision"),
    ("s3_knowledge_base", "query_accuracy"),
    ("s4_software_engineering", "dep_recall"),
    ("s5_self_planning", "task_accuracy"),
    ("s5_self_planning_alt", "recall_accuracy"),
    ("s6_naturalistic", "task_score"),
    ("s6_naturalistic_alt", "recall_rate"),
]


def _minimal_graph():
    """A tiny DAG with one version-test (trend) task and one interference task."""
    return {
        "tasks": {
            "t_trend": {
                "session": 1,
                "dependency_type": "trend",
                "fact_versions_required": {"f1": 2},  # requires latest (v2)
                "depends_on_facts": ["f1"],
                "chain_depth": 2,
            },
            "t_interf": {
                "session": 2,
                "dependency_type": "compare",
                "depends_on_facts": ["f2"],
                "chain_depth": 1,
            },
        },
        "facts": {
            "f1": {"versions": [{"fact_id": "f1_v1"}, {"fact_id": "f1_v2"}]},
            "f2": {"versions": [{"fact_id": "f2_v1"}]},
        },
        "interference_map": [
            {"shared_term": "budget", "fact_ids": ["f2", "f3"], "values": ["A", "B"]},
        ],
    }


@pytest.mark.parametrize("label,key", RUNNER_SCORE_KEYS)
def test_extract_score_recognizes_every_runner_key(label, key):
    assert _extract_score({key: 1.0}) == 1.0, f"{label}: '{key}' not recognized"
    assert _extract_score({key: 0.0}) == 0.0, f"{label}: '{key}' not recognized"


@pytest.mark.parametrize("label,key", RUNNER_SCORE_KEYS)
def test_version_accuracy_not_degenerate_for_perfect_run(label, key):
    """A perfect answer must yield version_accuracy == 1.0 for every runner shape.

    Before the fix this returned 0.0 for S1-S4 (their keys were unrecognized),
    which is the exact silent-degeneration bug this file guards against.
    """
    g = _minimal_graph()
    lookup = {1: {key: 1.0}}
    assert version_accuracy(lookup, g["tasks"], g["facts"]) == 1.0, (
        f"{label}: perfect run scored degenerate version_accuracy "
        f"(key '{key}' likely unrecognized by _extract_score)"
    )


@pytest.mark.parametrize("label,key", RUNNER_SCORE_KEYS)
def test_interference_resistance_not_degenerate_for_perfect_run(label, key):
    g = _minimal_graph()
    lookup = {2: {key: 1.0}}
    assert interference_resistance(lookup, g["tasks"], g["interference_map"]) == 1.0, (
        f"{label}: perfect run scored degenerate interference_resistance "
        f"(key '{key}' likely unrecognized by _extract_score)"
    )


@pytest.mark.parametrize("label,key", RUNNER_SCORE_KEYS)
def test_failing_run_scores_zero(label, key):
    """Negative control: a wrong answer must score 0, not a vacuous 1.0."""
    g = _minimal_graph()
    assert version_accuracy({1: {key: 0.0}}, g["tasks"], g["facts"]) == 0.0
    assert interference_resistance(
        {2: {key: 0.0}}, g["tasks"], g["interference_map"]
    ) == 0.0


def test_end_to_end_score_dependency_chain_s3_shape():
    """Full score_dependency_chain on S3-shaped (`query_accuracy`) results."""
    g = _minimal_graph()
    perfect = [{"session": 1, "query_accuracy": 1.0},
               {"session": 2, "query_accuracy": 1.0}]
    out = score_dependency_chain(perfect, g)
    assert out["version_accuracy"] == 1.0
    assert out["interference_resistance"] == 1.0

    failing = [{"session": 1, "query_accuracy": 0.0},
               {"session": 2, "query_accuracy": 0.0}]
    out_fail = score_dependency_chain(failing, g)
    assert out_fail["version_accuracy"] == 0.0
    assert out_fail["interference_resistance"] == 0.0
