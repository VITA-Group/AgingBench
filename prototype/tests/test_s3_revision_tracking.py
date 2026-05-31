"""Regression test for S3 revision-tracking fix (2026-05-30).

Bug: ``compute_fidelity`` checked memory_text against each gold decision's
FROZEN ORIGINAL keywords. When ``version_random_facts`` updated a fact's
keywords in the FactGraph (e.g. budget revised from $429,374 to $394,560),
the gold_timeline entry kept the OLD keywords. Result:

  - Agent that correctly adopts the revised value → memory has $394,560 →
    "$429,374" not in memory → fidelity says DECISION LOST (wrong: penalizes
    correctness)
  - Agent that retains stale residue → memory has $429,374 → fidelity says
    DECISION SURVIVED (wrong: rewards staleness)

Net: S3's headline ``summarization_fidelity`` was a partly-inverted revision
signal. Of 47 facts in a seed-42 8-session run, 29 (62%) get versioned —
substantial inversion.

Fix:
  - Generator now captures ``fact_id`` per decision and appends to
    ``keywords_history: [(session, [kws]), ...]`` each time
    ``version_random_facts`` fires
  - ``compute_fidelity(..., at_session=t)`` picks the latest history entry
    with session <= t (the "active" keywords)
  - Runner passes ``at_session=t`` when scoring fidelity
"""
from __future__ import annotations

from agingbench.scenarios.s3_knowledge_base.validator import (
    compute_fidelity, compute_fidelity_detailed, _active_keywords,
)


def test_active_keywords_falls_back_to_original_when_no_history():
    d = {"keywords": ["429,374"], "id": "D01"}
    assert _active_keywords(d) == ["429,374"]
    assert _active_keywords(d, at_session=5) == ["429,374"]


def test_active_keywords_falls_back_when_at_session_is_none():
    d = {
        "keywords": ["429,374"],
        "keywords_history": [(0, ["429,374"]), (3, ["394,560"])],
    }
    assert _active_keywords(d) == ["429,374"]  # None session → fall back


def test_active_keywords_picks_latest_le_session():
    d = {
        "keywords": ["429,374"],  # original
        "keywords_history": [
            (0, ["429,374"]),
            (3, ["394,560"]),
            (7, ["350,000"]),
        ],
    }
    # Before any revision
    assert _active_keywords(d, at_session=2) == ["429,374"]
    # At revision session
    assert _active_keywords(d, at_session=3) == ["394,560"]
    # Between revisions
    assert _active_keywords(d, at_session=5) == ["394,560"]
    # At second revision
    assert _active_keywords(d, at_session=7) == ["350,000"]
    # After second revision
    assert _active_keywords(d, at_session=20) == ["350,000"]


def test_fidelity_rewards_correct_revised_value():
    """Memory has the NEW value (agent learned the update). Pre-fix this
    would have scored 0 (NEW value doesn't match frozen OLD keyword).
    Post-fix it scores 1."""
    decisions = [{
        "id": "D01",
        "keywords": ["429,374"],  # frozen original
        "keywords_history": [(0, ["429,374"]), (3, ["394,560"])],
        "fact": "Contingency reserve",
    }]
    memory_with_new_value = "Contingency reserve set at $394,560 after the Q2 revision."
    score = compute_fidelity(memory_with_new_value, decisions, at_session=5)
    assert score == 1.0, (
        f"Memory with the revised value should fully credit the decision "
        f"(at_session=5 >= revision session 3); got {score}"
    )


def test_fidelity_does_not_reward_stale_residue_after_revision():
    """Memory has the OLD value only (agent failed revision). Pre-fix this
    scored 1 (matched frozen OLD keyword). Post-fix it scores 0 because the
    OLD keyword is no longer the active gold after the revision.

    Uses compute_fidelity (keyword-only path) to isolate the active-keyword
    selection from any semantic scoring behavior.
    """
    decisions = [{
        "id": "D01",
        "keywords": ["429,374"],  # frozen original (deprecated)
        "keywords_history": [(0, ["429,374"]), (3, ["394,560"])],
        "fact": "Contingency reserve",
    }]
    memory_with_stale_only = "Contingency reserve set at $429,374 (no update reflected)."
    score = compute_fidelity(memory_with_stale_only, decisions, at_session=5)
    assert score == 0.0, (
        f"Memory with only the stale (pre-revision) value should NOT score "
        f"survival after the revision; got {score}"
    )


def test_fidelity_uses_pre_revision_keywords_before_revision_session():
    """At sessions before the revision fires, the original keywords are the
    active gold. Memory with the original value scores 1."""
    decisions = [{
        "id": "D01",
        "keywords": ["429,374"],
        "keywords_history": [(0, ["429,374"]), (3, ["394,560"])],
        "fact": "Contingency reserve",
    }]
    memory_with_original = "Contingency reserve set at $429,374."
    # at_session=2 is before the session-3 revision
    score = compute_fidelity(memory_with_original, decisions, at_session=2)
    assert score == 1.0


def test_back_compat_when_decision_lacks_history():
    """Decisions without a keywords_history field (e.g. produced by older
    generator versions) should fall back to the original behavior."""
    decisions = [
        {"id": "D_old1", "keywords": ["foo"], "fact": "foo"},
        {"id": "D_old2", "keywords": ["bar"], "fact": "bar"},
    ]
    memory = "foo qux"
    # No keywords_history → behaves like pre-fix
    assert compute_fidelity(memory, decisions, at_session=5) == 0.5


def test_fidelity_detailed_session_aware_per_decision():
    """compute_fidelity_detailed should also respect at_session for the
    per_decision map and category_fidelity rollup."""
    decisions = [
        {
            "id": "D01",
            "keywords": ["429,374"],
            "keywords_history": [(0, ["429,374"]), (3, ["394,560"])],
            "fact": "Budget revised",
            "category": "budget",
        },
        {
            "id": "D02",
            "keywords": ["OAuth"],
            "fact": "OAuth required",
            "category": "security",
        },
    ]
    memory = "Budget revised to $394,560 with OAuth required everywhere."
    result = compute_fidelity_detailed(memory, decisions, at_session=5)
    assert result["per_decision"]["D01"] > 0.5  # revised value present
    assert result["per_decision"]["D02"] > 0.5  # OAuth present


def test_generator_actually_emits_history_field():
    """End-to-end sanity: the generator output now carries keywords_history
    on every decision and adds entries when versions fire."""
    from agingbench.generators.s3_generator import S3Generator
    from agingbench.generators.pressure_config import PressureConfig
    p = PressureConfig.medium(); p.warmup_sessions = 0; p.update_rate = 0.8
    r = S3Generator(seed=42, pressure=p).generate(n_sessions=8)
    decisions = r["gold_timeline"]["decisions"]
    assert decisions, "should produce decisions"
    for d in decisions:
        assert "keywords_history" in d, f"decision {d['id']} missing keywords_history"
        history = d["keywords_history"]
        assert history, f"decision {d['id']} has empty history"
        # First entry should be the original (session == decision's session)
        assert history[0][0] == d["session"]
        assert history[0][1] == d["keywords"]
    # At seed=42 with update_rate=0.8, expect a meaningful fraction revised
    revised = [d for d in decisions if len(d["keywords_history"]) > 1]
    assert len(revised) >= 5, (
        f"expected several decisions to be revised at heavy update_rate; "
        f"got {len(revised)}/{len(decisions)}"
    )
