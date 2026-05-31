"""Regression tests for the S5 keyword-extraction fix (2026-05-30).

Bug 1 (info facts): ``_generate_info_fact`` hard-coded
``keywords = [amount, person.split()[0], category]`` regardless of which
template was selected. Different templates use different ``fill`` keys
(appointment/schedule/location for dentist, allergen/doctor for medical,
account/provider for banking, etc), so the gold keywords were frequently
absent from the rendered prompt. On seed=42, 5 of 6 sample facts had
keywords like ['456', 'Kai', 'clothing'] attached to a prompt about
*dentist* appointments — meaning an agent that perfectly recalled the
fact still scored 0 because the gold tokens were unrelated.

Fix: collect candidate fill values from a broad key list, then filter to
those that actually appear in the rendered prompt.

Bug 2 (update tasks): ``_generate_update`` set
``eval_keywords = new_keywords = [new_amount] + old_keywords[1:]`` —
carrying forward stale contextual metadata (e.g. 'Bergström', 'tech
gadgets') into the *update task's* own scoring. The correction prompt
only mentions the OLD value and the NEW value; an agent acknowledging
the correction has no reason to echo the carried-forward metadata, and
its response is penalised for a prompt that never asked for those tokens.

Fix: tighten the update task's ``eval_keywords`` to only the new value
(which is the single specific token in the correction prompt). The
``new_keywords`` field is preserved as the structural version-chain
record used by the registry's recall-question gold (an inherited
recall question DOES need to score against the full post-update gold).
"""
from __future__ import annotations

from agingbench.generators.s5_generator import S5Generator
from agingbench.generators.pressure_config import PressureConfig


def _gen(seed: int = 42, n_sessions: int = 4):
    p = PressureConfig.medium()
    p.warmup_sessions = 0
    return S5Generator(seed=seed, pressure=p).generate(n_sessions=n_sessions)


def test_info_fact_keywords_appear_in_rendered_prompt():
    """For every original (non-replacement) info fact, every keyword in
    the gold must appear as a substring of the rendered prompt — this is
    the property the pre-fix code violated for 5/6 sample facts."""
    for seed in (42, 123, 7, 2024):
        r = _gen(seed=seed)
        for fact in r["facts_registry"]:
            if fact.get("replaces"):  # update facts checked separately
                continue
            prompt_lower = fact["prompt"].lower()
            missing = [kw for kw in fact["keywords"] if kw.lower() not in prompt_lower]
            assert not missing, (
                f"seed={seed} {fact['id']}: keywords {missing} not in "
                f"rendered prompt {fact['prompt'][:150]!r}"
            )


def test_info_facts_have_at_least_one_keyword():
    """The filter must not strip every candidate — every info fact still
    needs a recoverable gold target. (The fallback path keeps at least
    the amount if every other key gets filtered.)"""
    for seed in (42, 123, 7, 2024):
        r = _gen(seed=seed)
        for fact in r["facts_registry"]:
            if fact.get("replaces"):
                continue
            assert fact["keywords"], (
                f"seed={seed} {fact['id']}: keyword list is empty after filter"
            )


def test_update_task_eval_keywords_are_in_correction_prompt():
    """The update task's ``eval_keywords`` is scored against the agent's
    response to a correction prompt. Only tokens actually present in the
    correction prompt can reasonably appear in a correct acknowledgement
    — pre-fix code carried over original-fact metadata that never appeared
    in the prompt at all."""
    for seed in (42, 123, 7, 2024):
        r = _gen(seed=seed)
        for task in r["task_stream"]["tasks"]:
            if task.get("type") != "update":
                continue
            # The b{n}_update_0 path is the explicit correction prompt
            # we tightened; b{n}_ver_update_0 is the version-chain UPDATE
            # whose prompt restates the full fact so its eval_keywords
            # legitimately include the full new_keywords.
            if not task["id"].endswith("_update_0") or "ver" in task["id"]:
                continue
            prompt_lower = task["prompt"].lower()
            missing = [kw for kw in task["eval_keywords"]
                       if kw.lower() not in prompt_lower]
            assert not missing, (
                f"seed={seed} {task['id']}: eval_keywords {missing} not in "
                f"correction prompt {task['prompt']!r}"
            )


def test_update_task_preserves_new_keywords_for_downstream_gold():
    """``new_keywords`` is the structural record consumed by the
    facts_registry to set the post-update recall gold. It must stay as
    the full [new_amount, ...inherited_metadata] list even though
    ``eval_keywords`` is now tighter."""
    r = _gen()
    for task in r["task_stream"]["tasks"]:
        if task.get("type") != "update":
            continue
        nk = task.get("new_keywords")
        assert nk, f"{task['id']}: new_keywords must be populated"
        # First entry is the new amount; rest is inherited from old_keywords[1:]
        ok = task.get("old_keywords")
        assert ok, f"{task['id']}: old_keywords must be populated"
        assert nk[1:] == ok[1:], (
            f"{task['id']}: new_keywords should carry over old_keywords[1:] "
            f"for inherited-recall gold; got new={nk} old={ok}"
        )


def test_replacement_fact_in_registry_keeps_inherited_recall_gold():
    """When an update replaces fact_N, the new registry entry inherits
    fact_N's ``recall_question``. The new entry's ``keywords`` must
    therefore retain the contextual metadata fields (date / category /
    allergen / etc.) so that the inherited recall question still has the
    right gold answer — these aren't expected to appear in the update
    prompt itself (the update prompt only restates the changed value)."""
    r = _gen()
    n_replacements = 0
    for fact in r["facts_registry"]:
        if not fact.get("replaces"):
            continue
        n_replacements += 1
        assert fact["keywords"], (
            f"{fact['id']}: replacement fact must still carry the inherited gold"
        )
        # And the inherited recall_question should be non-empty
        assert fact.get("recall_question"), (
            f"{fact['id']}: replacement fact must inherit recall_question"
        )
    # Sanity: with seed=42, n_sessions=4 we should see at least one replacement
    assert n_replacements >= 1


def test_keyword_extraction_robust_across_seeds_and_domains():
    """Coverage check across multiple seeds: ≥ 90% of info facts in each
    run should have every keyword present in their prompt. (Two
    pathological templates that compose only literal sentences without
    fills are allowed to fall through to the amount-only fallback.)"""
    for seed in (42, 123, 7, 2024, 9999):
        r = _gen(seed=seed)
        info_facts = [f for f in r["facts_registry"] if not f.get("replaces")]
        good = 0
        for fact in info_facts:
            prompt_lower = fact["prompt"].lower()
            if all(kw.lower() in prompt_lower for kw in fact["keywords"]):
                good += 1
        ratio = good / max(1, len(info_facts))
        assert ratio >= 0.9, (
            f"seed={seed}: only {good}/{len(info_facts)} info facts have "
            f"all keywords in prompt (ratio={ratio:.2f}); pre-fix this was ~0"
        )
