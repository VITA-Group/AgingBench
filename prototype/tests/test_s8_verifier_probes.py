"""Phase 4 — verifier + 4-mechanism probe tests (credential-free)."""
from __future__ import annotations

import pytest

from agingbench.scenarios.s8_swe_bench.probes import (
    _files_in_diff,
    _instance_short_id,
    _symbols_in_diff,
    compute_compression_probe,
    compute_interference_probe,
    compute_maintenance_probe,
    compute_revision_probe,
    extract_task_critical_facts,
)
from agingbench.scenarios.s8_swe_bench.verifier import (
    _parse_pytest_log,
    apply_diff_in_container,
)


# ---- helpers --------------------------------------------------------------

def test_instance_short_id():
    assert _instance_short_id("sphinx-doc__sphinx-7454") == "7454"
    assert _instance_short_id("django__django-12345") == "12345"


def test_files_in_diff_extracts_paths():
    diff = """diff --git a/sphinx/ext/autodoc.py b/sphinx/ext/autodoc.py
--- a/sphinx/ext/autodoc.py
+++ b/sphinx/ext/autodoc.py
@@ -1,2 +1,2 @@
-old
+new
diff --git a/tests/test_a.py b/tests/test_a.py
--- a/tests/test_a.py
+++ b/tests/test_a.py
"""
    assert _files_in_diff(diff) == {"sphinx/ext/autodoc.py", "tests/test_a.py"}


def test_files_in_diff_empty():
    assert _files_in_diff("") == set()
    assert _files_in_diff(None) == set()


# ---- pytest log parser ----------------------------------------------------

def test_parse_pytest_log_passed_failed():
    log = """tests/test_x.py::test_a PASSED [10%]
tests/test_x.py::test_b FAILED [20%]
PASSED tests/test_x.py::test_c
FAILED tests/test_x.py::test_d
"""
    out = _parse_pytest_log(log)
    assert out["tests/test_x.py::test_a"] == "passed"
    assert out["tests/test_x.py::test_b"] == "failed"
    assert out["tests/test_x.py::test_c"] == "passed"
    assert out["tests/test_x.py::test_d"] == "failed"


# ---- compression probe ---------------------------------------------------

def test_compression_probe_recalls_prior_via_iid_fallback():
    """When no problem_statement supplied, fall back to lexical iid match."""
    notes = "## session 0 — sphinx-doc__sphinx-7454\nfixed autodoc_typehints"
    priors = [{"session": 0, "instance_id": "sphinx-doc__sphinx-7454"},
              {"session": 1, "instance_id": "sphinx-doc__sphinx-7462"}]
    probe = compute_compression_probe(2, notes, priors)
    assert probe.references_to_priors[0] is True   # 7454 mentioned
    assert probe.references_to_priors[1] is False  # 7462 not mentioned
    assert probe.recall_rate == 0.5


def test_compression_probe_token_overlap_continuous():
    """With problem_statement supplied, recall is continuous in [0,1]."""
    priors = [{
        "session": 0,
        "instance_id": "sphinx-doc__sphinx-7454",
        "problem_statement": ("Inconsistent handling of None by "
                              "autodoc_typehints. With description "
                              "mode, intersphinx clickable link..."),
    }]
    # Notes mentions some but not all key tokens
    notes = "Worked on autodoc_typehints None handling; affects description mode"
    probe = compute_compression_probe(1, notes, priors)
    # Continuous recall: matched / total key tokens; non-binary
    assert 0.0 < probe.recall_by_gap[1] < 1.0
    assert probe.recall_rate > 0.0
    assert probe.references_to_priors[0] is True  # nonzero overlap


def test_compression_probe_handles_empty_notes():
    priors = [{"session": 0, "instance_id": "sphinx-doc__sphinx-7454"}]
    probe = compute_compression_probe(1, "", priors)
    assert probe.recall_rate == 0.0
    assert probe.notes_size_bytes == 0


def test_compression_probe_recall_by_gap():
    """Recall should be tracked by gap; values are floats in [0,1]."""
    notes = "I worked on sphinx-doc__sphinx-7454 a while back"
    priors = [{"session": 0, "instance_id": "sphinx-doc__sphinx-7454"}]
    probe = compute_compression_probe(5, notes, priors)
    # Lexical fallback returns 1.0 when iid found
    assert probe.recall_by_gap[5] == 1.0


# ---- task-critical facts (gold-patch derived) --------------------------

_GOLD = """diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -42,7 +42,7 @@ class Documenter:
-    def format_signature(self):
+    def format_signature(self, **kwargs):
         pass

@@ -100,3 +100,8 @@ def stringify(annotation):
+def stringify_none(annotation):
+    if annotation is None:
+        return "None"
+    return annotation
"""


def test_symbols_in_diff_extracts_defs_and_classes():
    syms = _symbols_in_diff(_GOLD)
    assert "format_signature" in syms
    assert "stringify_none" in syms
    assert "Documenter" in syms


def test_extract_task_critical_facts_buckets():
    statement = (
        "autodoc_typehints with description mode renders None inconsistently;"
        " stringify should handle None and the format_signature path is affected."
    )
    facts = extract_task_critical_facts(statement, _GOLD)
    assert "sphinx/ext/autodoc/__init__.py" in facts["files"]
    assert "format_signature" in facts["symbols"]
    assert "stringify_none" in facts["symbols"]
    # concepts = problem-statement tokens that also survive into the patch.
    assert "stringify" in facts["concepts"]
    # raw problem-statement tokens not in the patch should be excluded.
    assert "description" not in facts["concepts"]


def test_compression_probe_task_critical_mode_continuous():
    """When gold_patch is supplied for a prior, recall is scored over the
    task-critical fact bundle (files + symbols + concepts) and stays
    continuous in [0, 1]."""
    priors = [{
        "session": 0,
        "instance_id": "sphinx-doc__sphinx-7454",
        "problem_statement": (
            "autodoc_typehints with description mode renders None"
            " inconsistently; stringify should handle None."
        ),
        "gold_patch": _GOLD,
    }]
    # Notes mention the file tail and one of the symbols but not all facts.
    notes = "session 2: touched __init__.py; rewrote format_signature"
    probe = compute_compression_probe(1, notes, priors)
    detail = probe.per_prior[0]  # type: ignore[attr-defined]
    assert detail["mode"] == "task_critical"
    assert 0.0 < detail["recall"] < 1.0
    # Per-bucket breakdown present.
    assert "by_bucket" in detail
    assert detail["by_bucket"]["files"]["matched"] >= 1
    assert detail["by_bucket"]["symbols"]["matched"] >= 1


def test_compression_probe_task_critical_zero_when_notes_empty():
    priors = [{
        "session": 0,
        "instance_id": "sphinx-doc__sphinx-7454",
        "problem_statement": "stringify None autodoc_typehints",
        "gold_patch": _GOLD,
    }]
    probe = compute_compression_probe(1, "", priors)
    assert probe.recall_rate == 0.0
    assert probe.per_prior[0]["mode"] == "task_critical"  # type: ignore[attr-defined]


# ---- interference probe -------------------------------------------------

def test_interference_probe_detects_overlap():
    """Pair (a, b) where b's diff touches the same file as a's gold."""
    sessions = [
        {"session": 0, "instance_id": "a", "agent_action":
            {"solution_diff_text": "--- a/foo.py\n+++ b/foo.py\n",
             "verification": {"n_pass_to_pass_total": 10, "n_pass_to_pass_passed": 10}}},
        {"session": 1, "instance_id": "b", "agent_action":
            {"solution_diff_text": "--- a/foo.py\n+++ b/foo.py\n",
             "verification": {"n_pass_to_pass_total": 10, "n_pass_to_pass_passed": 5}}},
    ]
    pairs = [["a", "b"]]
    gold = {"a": "--- a/foo.py\n+++ b/foo.py\n", "b": "--- a/bar.py\n+++ b/bar.py\n"}
    probe = compute_interference_probe(sessions, pairs, gold)
    assert probe.n_pairs_evaluated == 1
    assert probe.n_pairs_with_overlap == 1
    assert probe.resistance == 0.0
    # Continuous regression trajectory
    assert probe.regression_rate_trajectory == [(0, 0.0), (1, 0.5)]
    assert probe.regression_rate_mean == 0.25


def test_interference_probe_no_overlap():
    sessions = [
        {"session": 0, "instance_id": "a", "agent_action":
            {"solution_diff_text": "--- a/foo.py\n+++ b/foo.py\n"}},
        {"session": 1, "instance_id": "b", "agent_action":
            {"solution_diff_text": "--- a/baz.py\n+++ b/baz.py\n"}},
    ]
    gold = {"a": "--- a/foo.py\n+++ b/foo.py\n", "b": "--- a/bar.py\n+++ b/bar.py\n"}
    probe = compute_interference_probe(sessions, [["a", "b"]], gold)
    assert probe.resistance == 1.0  # no overlap = perfect resistance


def test_interference_probe_no_pairs_returns_empty_evaluable():
    probe = compute_interference_probe([], [], {})
    assert probe.n_pairs_evaluated == 0
    assert probe.resistance == 1.0
    assert probe.active_recall_mean is None


def test_interference_probe_active_recall_scores_attestation():
    """Active recall: the LATER member's attestation_text mentions the
    EARLIER member's gold-patch facts (files + symbols) -> high recall.
    """
    sessions = [
        {"session": 0, "instance_id": "a", "agent_action":
            {"solution_diff_text": "--- a/foo.py\n+++ b/foo.py\n",
             "verification": {"n_pass_to_pass_total": 10, "n_pass_to_pass_passed": 10}}},
        {"session": 1, "instance_id": "b", "agent_action":
            {"solution_diff_text": "--- a/foo.py\n+++ b/foo.py\n",
             "verification": {"n_pass_to_pass_total": 10, "n_pass_to_pass_passed": 5},
             # Active attestation: mentions the shared file + a fn name
             # from the gold patch:
             "attestation_text": (
                 "## Q1 [interference]\nRecall prior edit to foo.py.\n\n"
                 "In session 0 I edited foo.py — added a helper "
                 "stringify_none() to handle None typehints."
             )}},
    ]
    pairs = [["a", "b"]]
    gold = {
        "a": ("diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n"
              "@@ -1,1 +1,3 @@\n+def stringify_none(x):\n+    return 'None'\n"),
        "b": "--- a/bar.py\n+++ b/bar.py\n",
    }
    probe = compute_interference_probe(sessions, pairs, gold)
    assert len(probe.active_recall_per_pair) == 1
    pp = probe.active_recall_per_pair[0]
    assert pp["later_session"] == 1
    assert pp["earlier_session"] == 0
    assert pp["attestation_present"] is True
    # Both file (foo.py) and symbol (stringify_none) should match.
    assert pp["active_recall"] > 0.5
    assert probe.active_recall_mean == pp["active_recall"]


def test_interference_probe_per_session_trajectory_scores_recall_attestation():
    """Phase 12 dense trajectory: each session t>=1 has a 'recall'
    attestation question naming a prior session P; the probe scores the
    answer section against P's gold-patch facts and emits one trajectory
    point per session."""
    sessions = [
        {"session": 0, "instance_id": "a",
         "agent_action": {"solution_diff_text": "--- a/foo.py\n"}},
        {"session": 1, "instance_id": "b",
         "agent_action": {
             "solution_diff_text": "--- a/bar.py\n",
             "attestation_questions": {
                 "recall": "In session 0 you worked on issue a. recall."
             },
             "attestation_text": (
                 "## Q1 [recall]\nrecall.\nI edited foo.py — added "
                 "stringify_none() to handle the None case.\n"
                 "## Q2 [env]\nversion?\nVersion: 1.0.0"
             ),
         }},
        {"session": 2, "instance_id": "c",
         "agent_action": {
             "solution_diff_text": "--- a/baz.py\n",
             "attestation_questions": {
                 "recall": "In session 1 you worked on issue b. recall."
             },
             "attestation_text": (
                 "## Q1 [recall]\nrecall.\n(I don't remember)\n"
             ),
         }},
    ]
    gold = {
        "a": ("diff --git a/foo.py b/foo.py\n--- a/foo.py\n@@ -1 +1,3 @@\n"
              "+def stringify_none(x):\n+    return 'None'\n"),
        "b": ("diff --git a/bar.py b/bar.py\n--- a/bar.py\n@@ -1 +1,2 @@\n"
              "+def something_else(): pass\n"),
        "c": "",
    }
    probe = compute_interference_probe(sessions, [], gold)
    traj = probe.per_session_recall_trajectory
    assert len(traj) == 2
    # session 1 recalls foo.py + stringify_none -> high recall
    t1 = next(r for t, r in traj if t == 1)
    assert t1 > 0.5
    # session 2 doesn't recall bar.py / something_else -> low recall
    t2 = next(r for t, r in traj if t == 2)
    assert t2 == 0.0


def test_revision_probe_per_session_env_trajectory_continuous_partial_credit():
    """Phase 12 continuous scoring: per-session env answer scores +1/3
    for naming the probed pkg, +1/3 for any version-shape digits,
    +1/3 for a full X.Y.Z semver. So a complete answer = 1.0, a partial
    answer = 0.33/0.67, an empty answer = 0.0."""
    sessions = [
        # Empty placeholder -> 0
        {"session": 0, "instance_id": "a",
         "agent_action": {"solution_diff_text": ""}},
        # Full answer (pkg + X.Y.Z) -> 1.0
        {"session": 1, "instance_id": "b",
         "agent_action": {
             "solution_diff_text": "",
             "attestation_questions": {"env": "Run `pip show foo`."},
             "attestation_text": "## Q1 [env]\npip show foo result:\nVersion: 2.1.0 of foo\n",
         }},
        # Partial answer (X.Y only, no pkg name) -> 1/3 (loose ver only)
        {"session": 2, "instance_id": "c",
         "agent_action": {
             "solution_diff_text": "",
             "attestation_questions": {"env": "Run `pip show bar`."},
             "attestation_text": "## Q1 [env]\nSomething about 2.1 release.\n",
         }},
        # Empty -> 0
        {"session": 3, "instance_id": "d",
         "agent_action": {
             "solution_diff_text": "",
             "attestation_questions": {"env": "Run `pip show baz`."},
             "attestation_text": "## Q1 [env]\n(skipped)\n",
         }},
    ]
    probe = compute_revision_probe(sessions, [])
    traj = dict(probe.per_session_env_trajectory)
    assert traj[1] == 1.0          # full credit
    assert abs(traj[2] - 1/3) < 0.01    # partial: digit pattern only
    assert traj[3] == 0.0          # skipped
    assert abs(probe.per_session_env_mean - (1.0 + 1/3 + 0.0) / 3) < 0.01


def test_interference_probe_active_recall_zero_when_attestation_blank():
    sessions = [
        {"session": 0, "instance_id": "a", "agent_action":
            {"solution_diff_text": "--- a/foo.py\n+++ b/foo.py\n"}},
        {"session": 1, "instance_id": "b", "agent_action":
            {"solution_diff_text": "--- a/foo.py\n+++ b/foo.py\n",
             "attestation_text": ""}},
    ]
    gold = {
        "a": ("diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n"
              "@@ -1,1 +1,3 @@\n+def stringify_none(x):\n+    return 'None'\n"),
        "b": "",
    }
    probe = compute_interference_probe(sessions, [["a", "b"]], gold)
    assert probe.active_recall_per_pair[0]["active_recall"] == 0.0


# ---- revision probe -----------------------------------------------------

def test_revision_probe_acks_bumped_pkg():
    sessions = [
        {"session": 0, "instance_id": "a", "agent_notes_text": "first session note",
         "agent_action": {"solution_diff_text": ""}},
        {"session": 1, "instance_id": "b",
         "agent_notes_text": "session 1: new pytest version available",
         "agent_action": {"solution_diff_text": ""}},
    ]
    lifecycle = [
        {"session": 0, "event_type": "dep_bump",
         "detail": "pip install --upgrade pytest  (update_rate dial)"},
    ]
    probe = compute_revision_probe(sessions, lifecycle)
    assert probe.n_sessions_post_bump == 1
    assert probe.n_sessions_acknowledging == 1
    assert probe.rate == 1.0


def test_revision_probe_no_ack():
    sessions = [
        {"session": 0, "instance_id": "a", "agent_notes_text": "first session"},
        {"session": 1, "instance_id": "b", "agent_notes_text": "no mention",
         "agent_action": {"solution_diff_text": ""}},
    ]
    lifecycle = [
        {"session": 0, "event_type": "dep_bump",
         "detail": "pip install --upgrade Pygments"},
    ]
    probe = compute_revision_probe(sessions, lifecycle)
    assert probe.rate == 0.0


def test_revision_probe_no_bumps():
    probe = compute_revision_probe([], [])
    assert probe.rate == 0.0
    assert probe.n_sessions_post_bump == 0


def test_revision_probe_active_attestation_lifts_score():
    """When attestation_text contains a version string, the probe should
    score the session as acknowledging the bump even if the pkg name
    doesn't appear in notes/diff."""
    sessions = [
        {"session": 0, "instance_id": "a", "agent_action": {"solution_diff_text": ""}},
        {"session": 1, "instance_id": "b",
         "agent_notes_text": "fixed something unrelated",
         "agent_action": {
             "solution_diff_text": "no mention",
             # active attestation answers the version question:
             "attestation_text": (
                 "## Q1 [revision]\nReport version of pluggy.\n\n"
                 "Version: 1.5.0 (from pip show pluggy)\n"
             ),
         }},
    ]
    lifecycle = [{"session": 0, "event_type": "dep_bump",
                  "detail": "pip install --upgrade pluggy"}]
    probe = compute_revision_probe(sessions, lifecycle)
    assert probe.n_sessions_post_bump == 1
    assert probe.rate == 1.0
    r = probe.per_session[0]
    assert r["scoring_mode"] == "active"
    assert r["attested_version"] == "1.5.0"
    assert r["ack"] is True


def test_revision_probe_latent_accumulator_abs_error():
    """Phase 17.3: latent-accumulator probe (S7 m_revision_latent_abs_err
    equivalent). Agent reports an integer count each session; we score
    |reported - ground_truth|. Ground truth at session t = t."""
    sessions = [
        {"session": 0, "instance_id": "a",
         "agent_action": {"solution_diff_text": ""}},
        # session 1: agent reports correctly (1 session completed before this)
        {"session": 1, "instance_id": "b",
         "agent_action": {
             "solution_diff_text": "",
             "attestation_questions": {
                 "accumulator_sessions_completed": "How many sessions completed?",
             },
             "attestation_text": "## Q1 [accumulator_sessions_completed]\n\nI have completed 1 session so far.\n",
         }},
        # session 2: agent reports STALE belief (says 1 instead of 2 — drift!)
        {"session": 2, "instance_id": "c",
         "agent_action": {
             "solution_diff_text": "",
             "attestation_questions": {
                 "accumulator_sessions_completed": "How many sessions completed?",
             },
             "attestation_text": "## Q1 [accumulator_sessions_completed]\n\nI think 1 session completed.\n",
         }},
        # session 3: agent corrects to 3 (revised back)
        {"session": 3, "instance_id": "d",
         "agent_action": {
             "solution_diff_text": "",
             "attestation_questions": {
                 "accumulator_sessions_completed": "How many sessions completed?",
             },
             "attestation_text": "## Q1 [accumulator_sessions_completed]\n\n3 sessions are done.\n",
         }},
    ]
    probe = compute_revision_probe(sessions, [])
    records = probe.latent_per_session
    by_t = {r["session"]: r for r in records}
    assert by_t[1]["extracted"] == 1 and by_t[1]["gold"] == 1 and by_t[1]["abs_err"] == 0
    assert by_t[2]["extracted"] == 1 and by_t[2]["gold"] == 2 and by_t[2]["abs_err"] == 1
    assert by_t[3]["extracted"] == 3 and by_t[3]["gold"] == 3 and by_t[3]["abs_err"] == 0
    # mean abs err = (0 + 1 + 0) / 3
    assert abs(probe.latent_abs_err_mean - 1/3) < 0.01


def test_revision_probe_passive_fallback_when_no_attestation():
    """When attestation is absent and the agent's notes/diff mention the
    bumped pkg, passive scoring still fires."""
    sessions = [
        {"session": 0, "instance_id": "a", "agent_action": {"solution_diff_text": ""}},
        {"session": 1, "instance_id": "b",
         "agent_notes_text": "upgraded pluggy",
         "agent_action": {"solution_diff_text": ""}},  # no attestation
    ]
    lifecycle = [{"session": 0, "event_type": "dep_bump",
                  "detail": "pip install --upgrade pluggy"}]
    probe = compute_revision_probe(sessions, lifecycle)
    assert probe.rate == 1.0
    assert probe.per_session[0]["scoring_mode"] == "passive"
    assert probe.per_session[0]["passive_hit"] is True


# ---- maintenance probe --------------------------------------------------

def test_maintenance_probe_pre_post_delta():
    sessions = [
        {"session": 0, "verification": {"passed": True}},
        {"session": 1, "verification": {"passed": True}},
        {"session": 2, "verification": {"passed": False}},
        {"session": 3, "verification": {"passed": False}},
    ]
    lifecycle = [{"session": 2, "event_type": "workspace_flush"}]
    probe = compute_maintenance_probe(sessions, lifecycle, window=2)
    assert probe.shock_sessions == [2]
    assert probe.pre_shock_pass_rate == 1.0
    assert probe.post_shock_pass_rate == 0.0
    assert probe.delta == -1.0


def test_maintenance_probe_no_shocks():
    probe = compute_maintenance_probe([], [], window=2)
    assert probe.shock_sessions == []
    assert probe.delta is None


def test_maintenance_probe_memory_recall_delta_at_pass_floor():
    """Even when pass_rate stays at 0 pre and post, memory_recall should
    pick up a flush-induced loss of task-critical facts in notes."""
    prior_facts = [{
        "session": 0,
        "instance_id": "sphinx-doc__sphinx-7454",
        "problem_statement": "stringify None autodoc_typehints",
        "gold_patch": _GOLD,
    }]
    sessions = [
        # pre-shock: notes contain the file + symbols (high recall).
        {"session": 1, "verification": {"passed": False},
         "agent_notes_text": "touched sphinx/ext/autodoc/__init__.py; "
                              "edited format_signature; added stringify_none",
         "prior_facts": prior_facts},
        # post-shock: notes are empty (recall collapses).
        {"session": 2, "verification": {"passed": False},
         "agent_notes_text": "",
         "prior_facts": prior_facts},
    ]
    lifecycle = [{"session": 2, "event_type": "workspace_flush"}]
    probe = compute_maintenance_probe(sessions, lifecycle, window=1)
    assert probe.shock_sessions == [2]
    # Pass rate is flat at 0 -> capability signal is dead.
    assert probe.pre_shock_pass_rate == 0.0
    assert probe.post_shock_pass_rate == 0.0
    assert probe.pass_rate_delta == 0.0
    # Memory recall picks up the regression.
    assert probe.pre_shock_memory_recall is not None
    assert probe.pre_shock_memory_recall > 0.0
    assert probe.post_shock_memory_recall == 0.0
    assert probe.delta is not None and probe.delta < 0.0
