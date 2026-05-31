"""Regression test for S6 xref-task leakage fix (2026-05-30).

Bug: ``_generate_xref_session`` built the cross-reference task text by
literally embedding the selected facts' key_fact strings (e.g. "Revenue:
$785,163") and then used the same facts' keywords as ``eval_keywords``. The
agent could score a perfect cross-reference "synthesis" by simply echoing
substrings from its own prompt input — no memory recall required.

In a seed-42 12-session run, every one of the 6 eval_keywords was a literal
substring of the task text — turning S6's xref task into a copy-echo test
rather than the intended "multi-domain synthesis from memory" probe that
the paper describes (§B.6, line 1283 "Cross-Reference with Correction").

Fix: redact each fact's value tokens in the task text while preserving the
topic/label structure. The agent now sees "Revenue: $[?]" and "Top product:
[?]" — enough hint to know WHAT to recall, no leakage of the actual values.
``eval_keywords`` remain the original gold values so the scoring path is
unchanged.
"""
from __future__ import annotations

from agingbench.generators.s6_generator import S6Generator
from agingbench.generators.pressure_config import PressureConfig


def _xref_sessions(result: dict) -> list[dict]:
    return [s for s in result["session_tasks"]["sessions"]
            if s.get("is_cross_reference")]


def test_no_eval_keyword_appears_in_xref_task_text():
    """The whole point of the fix: every eval_keyword must NOT be present in
    the task text. Tested across multiple seeds for robustness."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    for seed in (42, 123, 7, 2024):
        r = S6Generator(seed=seed, pressure=p).generate(n_sessions=12)
        for xref in _xref_sessions(r):
            text_lower = xref["task"]["text"].lower()
            for kw in xref["task"]["eval_keywords"]:
                assert kw.lower() not in text_lower, (
                    f"seed={seed}: eval_keyword {kw!r} leaked into xref task "
                    f"text: {xref['task']['text'][:200]!r}"
                )


def test_xref_task_text_includes_recall_placeholders():
    """The redaction should leave a visible placeholder so the agent knows
    WHAT to recall (e.g. 'Revenue: $[?]') rather than just stripping it."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S6Generator(seed=42, pressure=p).generate(n_sessions=12)
    for xref in _xref_sessions(r):
        assert "[?]" in xref["task"]["text"], (
            f"redacted task text should include [?] placeholders so agent "
            f"knows what to recall; got: {xref['task']['text'][:200]!r}"
        )


def test_xref_eval_keywords_preserved_as_gold():
    """eval_keywords still hold the original (un-redacted) gold values so
    the scoring path is unchanged."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S6Generator(seed=42, pressure=p).generate(n_sessions=12)
    for xref in _xref_sessions(r):
        kws = xref["task"]["eval_keywords"]
        assert kws, "xref must still emit eval_keywords"
        # Make sure they look like real fact values, not placeholders
        for kw in kws:
            assert kw != "[?]", f"eval_keyword should be a gold value, not the placeholder; got {kw!r}"
            assert "[?]" not in kw


def test_xref_task_text_still_has_topic_structure():
    """The redacted prompt should still convey topic structure so the agent
    has something to anchor on — e.g. 'Revenue: $[?]' tells the agent the
    category is 'Revenue' even though the value is masked."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S6Generator(seed=42, pressure=p).generate(n_sessions=12)
    for xref in _xref_sessions(r):
        text = xref["task"]["text"]
        # Expect bullet structure preserved
        assert text.count("- ") >= 3, (
            f"xref task should bullet-list the topics; got {text[:200]!r}"
        )


def test_xref_task_no_longer_lists_unredacted_fact_summaries():
    """The pre-fix pattern '- Revenue: $785,163' must not appear. After the
    fix, every bulleted line should contain a [?] placeholder."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S6Generator(seed=42, pressure=p).generate(n_sessions=12)
    for xref in _xref_sessions(r):
        text = xref["task"]["text"]
        for line in text.split("\n"):
            if line.startswith("- "):
                assert "[?]" in line, (
                    f"every bulleted topic line should carry a [?] placeholder "
                    f"to mark the value the agent must recall; got: {line!r}"
                )


def test_reference_answer_still_carries_full_facts():
    """For trace logging / debugging, the reference_answer should still
    contain the full fact text (not redacted) so analysts can see what the
    gold synthesis would look like."""
    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S6Generator(seed=42, pressure=p).generate(n_sessions=12)
    for xref in _xref_sessions(r):
        ra = xref["task"].get("reference_answer", "")
        assert "[?]" not in ra, (
            f"reference_answer should preserve the original fact text "
            f"(not redacted); got {ra!r}"
        )
        # And it should contain at least one of the eval keywords (as a
        # sanity that ref_answer matches the gold).
        assert any(kw in ra for kw in xref["task"]["eval_keywords"]), (
            f"reference_answer should contain at least one eval_keyword; got {ra!r}"
        )
