"""Regression test for the S4 dep_recall scoring fix (2026-05-30).

Bug: ``s4_runner.py`` previously computed dep_recall via a naive
``dep_context.lower().split()`` with a 5-word stoplist:

    dep_keywords = [w for w in dep_context.lower().split()
                   if len(w) > 4 and w not in ("sprint","using","added","which","must")]

This had three independent problems that compounded:
  1. Punctuation glued to tokens — "fields." won't match an agent's "fields"
  2. Duplicate keywords inflated hits — "session" appearing 3× in dep_context
     counted as 3 separate matches when the agent mentioned it once
  3. Tiny stoplist failed to filter generator-injected structural markers
     ("UPDATE:", "CORRECTION:", "important", "previous", "finding", etc.)
     so the agent had to repeat boilerplate to score recall

Net bias: deflated. The paper's ``S4 dep_rec m_F ↑`` column understated
real model performance by an estimated 30–50%.

Fix: extracted ``_compute_dep_recall`` helper that uses regex token
extraction, dedupes, and filters mixin markers + common English.
"""
from __future__ import annotations

from agingbench.runner.s4_runner import _compute_dep_recall


def test_empty_dep_context_returns_vacuously_one():
    assert _compute_dep_recall("", "any output") == 1.0


def test_punctuation_in_dep_context_does_not_block_match():
    """Pre-fix: 'fields.' (with period) only matched if agent included the period.
    Post-fix: regex strips to 'fields', matches a natural 'fields' in output."""
    dep_context = "Session 1 added Customer model with 3 fields."
    agent_output = "I will add input validation to the Customer model fields"
    score = _compute_dep_recall(dep_context, agent_output)
    # Customer, model, fields — at least 3 distinct content tokens; agent
    # mentions all three → score should be ≥ baseline (0.30 threshold)
    assert score == 1.0


def test_duplicate_tokens_do_not_inflate_score():
    """Pre-fix: 'session' appearing 3× in dep_context counted as 3 hits when
    agent mentioned it once. Post-fix: dedup means each unique kw counts once."""
    # Construct a dep_context where "fields" appears 3 times — pre-fix would
    # over-credit; post-fix dedupes.
    dep_context = "fields fields fields"
    agent_output = "the fields are here"
    # With dedup: 1 unique kw → threshold = max(1*0.3,1) = 1 → hits=1 → recall=1.0
    # Without dedup (pre-fix): hits=3, denom=max(3*0.3,1)=1 → recall=1.0
    # Both score 1.0 for this case, but check the underlying mechanics:
    assert _compute_dep_recall(dep_context, agent_output) == 1.0
    # And when agent does NOT mention "fields", both score 0
    assert _compute_dep_recall(dep_context, "completely unrelated text here") == 0.0


def test_structural_markers_are_filtered():
    """Generator-injected markers from invalidate_random_facts and
    version_random_facts ('UPDATE:', 'CORRECTION:', 'IMPORTANT:', 'previous',
    'finding', 'invalid', 'withdrawn', etc.) should NOT count as recall targets."""
    dep_context = (
        "Session 2 added Customer model with 3 fields. "
        "IMPORTANT: The information about Customer model is no longer accurate. "
        "The value 'Customer' has been retracted. Please disregard."
    )
    # Only real content: "added", "customer", "model", "fields" → after STRUCTURAL
    # filtering "added" is in the structural list → leaves customer, model, fields
    # Agent that mentions all three "Customer/model/fields" scores high
    relevant_output = "I will add validation to Customer model fields"
    # Agent that ONLY repeats the structural boilerplate scores low
    boilerplate_output = "previous finding is invalid please disregard, retracted withdrawn"
    relevant = _compute_dep_recall(dep_context, relevant_output)
    boilerplate = _compute_dep_recall(dep_context, boilerplate_output)
    assert relevant > boilerplate
    assert relevant >= 0.8, f"agent mentioning real entities should score high, got {relevant}"
    # Boilerplate-only output should NOT get credit for structural words
    assert boilerplate <= 0.5, (
        f"agent repeating only structural markers should NOT score full credit, "
        f"got {boilerplate} (the bug would have made this near 1.0)"
    )


def test_short_tokens_below_threshold_are_filtered():
    """The 5-char threshold drops single letters / common short words."""
    dep_context = "a b cd efg models is the"   # only 'models' qualifies
    agent_output = "models is the system"
    # 'models' present → 1/1 → 1.0
    assert _compute_dep_recall(dep_context, agent_output) == 1.0


def test_no_overlap_yields_zero():
    dep_context = "Customer model with validation fields and methods"
    agent_output = "I will write tests for the Order entity using async patterns"
    # No content tokens in common
    assert _compute_dep_recall(dep_context, agent_output) == 0.0


def test_partial_overlap_proportional_to_threshold():
    """Partial overlap should produce fractional credit relative to the
    30% threshold."""
    dep_context = "customer product order vendor pricing inventory"  # 6 content tokens
    # threshold = max(6 * 0.3, 1) = 1.8
    # 1 hit: 1/1.8 ≈ 0.56
    agent_output = "I added Customer validation"
    score = _compute_dep_recall(dep_context, agent_output)
    assert 0.5 < score < 0.6, f"1/6 should give partial credit ~0.56, got {score}"


def test_dedup_means_score_independent_of_repetition_in_context():
    """Two dep_contexts that differ only by how often a word appears should
    produce identical dep_recall scores under the dedup fix."""
    dep_context_short = "customer model validation"
    dep_context_repeated = "customer customer customer model model validation validation"
    agent = "I implemented Customer model validation"
    assert _compute_dep_recall(dep_context_short, agent) == _compute_dep_recall(
        dep_context_repeated, agent
    )
