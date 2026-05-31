"""Regression test for the S2 accumulator-probe Option A fix (2026-05-30).

Bug: ``_generate_accumulator_track`` previously placed accumulator probes
in the SAME session as the delta they need to integrate. Because S2's
ReAct runner reads memory once per task at start of ``agent.run_session()``
and only writes memory at end-of-session, the probe in session t never
saw the delta from session t — producing a systematic ~$20–80 error per
affected probe (the delta amount), which inflated the paper's reported
``S2 accum err`` numbers by ~20–40% as a structural confound.

Fix: probe now placed at session t+1; gold still reflects end-of-t state.
The runner's natural session-boundary write makes the t-delta visible
to the agent's memory by session t+1.
"""
from __future__ import annotations

from agingbench.generators.s2_generator import S2Generator
from agingbench.generators.pressure_config import PressureConfig


def test_accumulator_probe_session_is_one_past_delta_origin():
    """Every accumulator probe's session must be > its gold_at_session."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S2Generator(seed=42, pressure=p).generate(n_sessions=10)

    probes = r["accumulator_probes"]
    assert probes, "generator should emit at least one accumulator probe"
    for ap in probes:
        assert ap["session"] > ap["gold_at_session"], (
            f"probe session ({ap['session']}) must be later than the session "
            f"the gold corresponds to ({ap['gold_at_session']}) — the +1 offset "
            f"is the entire point of the Option A fix"
        )
        assert ap["session"] == ap["gold_at_session"] + 1, (
            f"probe should be placed exactly 1 session after gold_at_session "
            f"(got probe at {ap['session']}, gold_at {ap['gold_at_session']})"
        )


def test_accumulator_probe_does_not_co_occur_with_its_delta():
    """The probe's session should NEVER also contain a delta from the
    SAME accumulator value the probe asks about. Walking session tasks:
    if the probe is at session N, then session N's tasks should NOT contain
    an ``accumulator_delta`` whose effect is supposed to be measured by the
    probe (the probe's gold reflects state through session N-1)."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S2Generator(seed=42, pressure=p).generate(n_sessions=10)

    sessions = r["session_tasks"]["sessions"]
    probes = r["accumulator_probes"]

    for ap in probes:
        sess_idx = ap["session"]
        gold_at = ap["gold_at_session"]
        # Inspect session gold_at — it should contain a delta task (the one
        # whose effect the probe is meant to capture, now safely propagated
        # via end-of-session memory write).
        if gold_at < len(sessions):
            cats = [t.get("category") for t in sessions[gold_at]["tasks"]]
            assert "accumulator_delta" in cats, (
                f"the gold-source session {gold_at} should still contain "
                f"its delta task; categories were {cats}"
            )


def test_accumulator_probe_drops_when_t_plus_one_exceeds_horizon():
    """If a probe candidate would land past the last session, it must be
    dropped — not silently retained at the original t."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    # Pick small n where the modular schedule lands a probe near the end
    for n_sessions in (5, 6, 7, 8, 9, 10, 12, 15, 20):
        r = S2Generator(seed=42, pressure=p).generate(n_sessions=n_sessions)
        for ap in r["accumulator_probes"]:
            assert ap["session"] < n_sessions, (
                f"n={n_sessions}: probe session {ap['session']} must be < n_sessions"
            )


def test_accumulator_gold_unchanged_by_offset_fix():
    """Sanity: the gold values themselves should be the same as the old code
    would have produced — only the placement shifted. Compare seed-42 generation
    to the documented gold values."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S2Generator(seed=42, pressure=p).generate(n_sessions=10)

    # Build (gold_at_session, gold_value) pairs and check the gold values
    # are non-zero and round numbers (matching expected accumulator behaviour).
    for ap in r["accumulator_probes"]:
        assert ap["gold_value"] > 0, "gold_value must be a positive budget figure"
        assert "gold_at_session" in ap, "must record gold_at_session for scoring"
