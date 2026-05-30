"""Regression test for the S6 probe-key revision sync.

Bug (pre-fix): `version_random_facts` mutates a fact's numeric value but the
recall-probe keywords generated when the fact was first introduced were never
updated. An agent that correctly tracked the revision and cited the *new*
value failed the keyword match — inverting the revision-mechanism signal.

This test forces every prior fact to be versioned every session, then asserts
that for every revised fact, the originating session's recall probes carry the
new keywords (not the original).
"""
from __future__ import annotations

from agingbench.generators.s6_generator import S6Generator
from agingbench.generators.pressure_config import PressureConfig


def _high_pressure() -> PressureConfig:
    """Pressure config that maximises versioning so the test reliably fires."""
    cfg = PressureConfig.none()
    cfg.update_rate = 1.0
    cfg.warmup_sessions = 0
    return cfg


def test_revision_sync_updates_probe_keywords():
    gen = S6Generator(seed=123, pressure=_high_pressure())
    data = gen.generate(n_sessions=10)

    sessions = data["session_tasks"]["sessions"]
    facts_export = data["dependency_graph"]["facts"]

    revised_seen = 0
    for root_id, fdata in facts_export.items():
        versions = fdata.get("versions") or []
        if len(versions) < 2:
            continue  # not revised — nothing to check

        original = versions[0]
        latest = versions[-1]
        original_kws = list(original.get("keywords") or [])
        latest_kws = list(latest.get("keywords") or [])
        if set(original_kws) == set(latest_kws):
            continue  # version chain present but no keyword-level change

        origin = original.get("session", fdata.get("introduced_session"))
        if origin is None or not (0 <= origin < len(sessions)):
            continue

        # Find probes in the originating session that previously held the
        # original keywords (subset match). After the fix they should carry
        # the latest keywords instead.
        relevant = [
            p for p in sessions[origin].get("recall_probes", []) or []
            if p.get("keywords") and set(p["keywords"]) <= set(original_kws + latest_kws)
        ]
        if not relevant:
            continue

        revised_seen += 1
        for p in relevant:
            pkws = p["keywords"]
            # PRE-FIX: pkws would still be a subset of original_kws.
            # POST-FIX: at least one of pkws should be in latest_kws.
            assert any(k in latest_kws for k in pkws), (
                f"probe {p.get('probe_id')!r} in session {origin} still uses "
                f"original keywords {pkws!r}; expected at least one of the "
                f"revised keywords {latest_kws!r} for root fact {root_id}."
            )
            # And it should NOT exclusively quote stale numeric values.
            stale_only = [k for k in original_kws if k not in latest_kws]
            still_stale = [k for k in pkws if k in stale_only]
            assert not still_stale, (
                f"probe {p.get('probe_id')!r} in session {origin} still "
                f"contains stale tokens {still_stale!r} after revision; "
                f"expected the position-aligned new tokens from {latest_kws!r}."
            )

    assert revised_seen >= 1, (
        "expected at least one revised fact with a matchable origin-session "
        "probe under update_rate=1.0; if 0 the test is vacuous (generator "
        "shape may have changed)."
    )


def test_no_revisions_means_no_changes():
    """With pressure.none, `version_random_facts` returns [] every iteration,
    so the sync helper is never invoked and probe keywords must be identical
    across two independent generator runs with the same seed."""
    gen1 = S6Generator(seed=42, pressure=PressureConfig.none())
    gen2 = S6Generator(seed=42, pressure=PressureConfig.none())
    d1 = gen1.generate(n_sessions=8)
    d2 = gen2.generate(n_sessions=8)
    for s1, s2 in zip(
        d1["session_tasks"]["sessions"], d2["session_tasks"]["sessions"]
    ):
        for p1, p2 in zip(s1.get("recall_probes", []) or [], s2.get("recall_probes", []) or []):
            assert p1["keywords"] == p2["keywords"]
