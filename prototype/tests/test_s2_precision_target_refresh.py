"""Regression test for the S2 precision_targets refresh fix (2026-05-30).

Bug: ``_generate_eval_probes`` ran before ``_generate_constraint_updates``,
so eval probes had FROZEN ``precision_targets`` even when a "relax" update
later changed the constraint's numeric value (e.g. dining budget $309 →
$400). After the update, an agent that correctly learned the new value
would cite $400 → no substring match against frozen target [$309] → score
0; an agent that ignored the update would cite $309 → match → score 1.
**The metric was rewarding staleness.**

Fix: the generator now annotates affected probes with
``precision_target_change={"session": N, "new_targets": [...]}``, and the
validator's ``_resolve_active_targets`` picks the post-update target at
sessions >= N. Backward-compatible when no change is set.
"""
from __future__ import annotations

from agingbench.scenarios.s2_lifestyle_assistant.validator import (
    score_probe_precision, compute_constraint_precision, _resolve_active_targets,
)


def test_resolve_targets_falls_back_when_no_change_set():
    probe = {"precision_targets": ["309"]}
    assert _resolve_active_targets(probe) == ["309"]
    assert _resolve_active_targets(probe, session_idx=0) == ["309"]
    assert _resolve_active_targets(probe, session_idx=100) == ["309"]


def test_resolve_targets_switches_at_change_session():
    probe = {
        "precision_targets": ["309"],
        "precision_target_change": {"session": 6, "new_targets": ["400"]},
    }
    # Before the change: still use original
    assert _resolve_active_targets(probe, session_idx=0) == ["309"]
    assert _resolve_active_targets(probe, session_idx=5) == ["309"]
    # At and after the change: new targets active
    assert _resolve_active_targets(probe, session_idx=6) == ["400"]
    assert _resolve_active_targets(probe, session_idx=9) == ["400"]


def test_score_probe_precision_credits_correct_value_per_session():
    """The scoring metric should credit '400' after the update, not before."""
    probe = {
        "id": "eval_C1",
        "constraint_id": "C1",
        "precision_targets": ["309"],
        "precision_target_change": {"session": 6, "new_targets": ["400"]},
    }
    response_with_old = "The dining budget is $309 per month."
    response_with_new = "The dining budget is $400 per month."

    # Pre-update sessions: old value scores, new value does not
    assert score_probe_precision(probe, response_with_old, session_idx=3)["precision_score"] == 1.0
    assert score_probe_precision(probe, response_with_new, session_idx=3)["precision_score"] == 0.0

    # Post-update sessions: new value scores, old value does not (the bug)
    assert score_probe_precision(probe, response_with_new, session_idx=7)["precision_score"] == 1.0
    assert score_probe_precision(probe, response_with_old, session_idx=7)["precision_score"] == 0.0


def test_compute_constraint_precision_session_aware():
    """End-to-end: per-session aggregation honors precision_target_change."""
    probes = [
        {
            "id": "eval_C1",
            "constraint_id": "C1",
            "precision_targets": ["309"],
            "precision_target_change": {"session": 6, "new_targets": ["400"]},
        },
        {
            "id": "eval_C2",
            "constraint_id": "C2",
            "precision_targets": ["boycott"],
        },
    ]
    outputs = ["I see your dining budget is $400.", "Walmart is on your boycott list."]
    result = compute_constraint_precision(probes, outputs, session_idx=7)
    # eval_C1: $400 matches new target → 1
    # eval_C2: "boycott" matches → 1
    assert result["constraint_precision"] == 1.0

    # Same outputs at session 3 (pre-update):
    # eval_C1: $400 does NOT match old target $309 → 0
    # eval_C2: still 1
    result_pre = compute_constraint_precision(probes, outputs, session_idx=3)
    assert result_pre["constraint_precision"] == 0.5


def test_generator_emits_precision_target_change_for_relax_updates():
    """Live generator test: after a relax update, the affected probe carries
    a precision_target_change entry pointing at the new value and session."""
    from agingbench.generators.s2_generator import S2Generator
    from agingbench.generators.pressure_config import PressureConfig

    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S2Generator(seed=42, pressure=p).generate(n_sessions=10)

    updates = r["constraint_updates"]["updates"]
    relax_updates = [u for u in updates if u.get("type") == "relax"]
    assert relax_updates, "expected at least one relax update at seed 42, n=10"

    eval_probes = r["eval_probes"]["probes"]
    for upd in relax_updates:
        probe = next((p for p in eval_probes if p["constraint_id"] == upd["constraint_id"]), None)
        assert probe is not None, f"no eval probe for constraint {upd['constraint_id']}"
        change = probe.get("precision_target_change")
        assert change, (
            f"relax update at session {upd['session']} for {upd['constraint_id']} "
            f"should have set precision_target_change on its eval probe; "
            f"probe = {probe}"
        )
        assert change["session"] == upd["session"]
        # new_targets should be NEW value (not in original precision_targets)
        assert change["new_targets"], "new_targets should be non-empty"
        assert change["new_targets"] != probe["precision_targets"], (
            f"new targets should differ from original; both are {probe['precision_targets']}"
        )
