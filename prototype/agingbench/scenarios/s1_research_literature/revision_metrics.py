"""Baseline-corrected revision-aging metrics for S1, ported from S3.

S3's "trident" approach uses the difference between revised and unrevised facts
as a baseline correction for compression aging — the residual is the revision-
specific contribution. The same logic applies to S1 since both scenarios use
DependencyMixin's version_random_facts and produce versioned facts in the
dependency graph.

All three metrics operate on memory text (no agent-response brittleness) and
compare revised vs unrevised partitions to subtract out compression aging.

Metrics:
  revision_fidelity_excess   — fidelity(unrevised) − fidelity(revised) ∈ [-1, 1]
                                positive = revised facts decay faster than baseline
  stale_residue_rate         — fraction of revised facts where old keyword is in M_t
                                but current keyword is absent (pure revision failure)
  coverage_verdict           — sample-size honesty
                                (no_revisions / underpowered / adequate / strong)
"""

from __future__ import annotations

import re
from typing import Optional


_MIN_DISCRIMINATING_LEN = 3  # filter single-digit numerics like '6', '4' from
                              # the discriminating set — they match almost any
                              # decimal/percent text and produce false-positives.


def _present(needle: str, haystack_lower: str) -> bool:
    """Substring presence with digit/word-flank guard.

    For purely-numeric needles, additionally requires length ≥ 3 — single-digit
    keywords like '6' would otherwise match inside '20.6%', '$788,508', '2026',
    etc., trivially inflating stale-residue and fidelity counts.
    """
    if not needle:
        return False
    n = needle.lower()
    if n.isdigit() and len(n) < _MIN_DISCRIMINATING_LEN:
        return False
    return re.search(r"(?<!\d)" + re.escape(n) + r"(?!\d)",
                     haystack_lower) is not None


def _active_version(fact_data: dict, at_cycle: int) -> Optional[dict]:
    """The latest non-invalidated version of a fact as of `at_cycle`. None if
    the fact wasn't introduced yet or every version is invalidated by now."""
    versions = fact_data.get("versions") or []
    candidates = [
        v for v in versions
        if v.get("session", 0) <= at_cycle
        and (v.get("invalidated_at") is None or v["invalidated_at"] > at_cycle)
    ]
    return candidates[-1] if candidates else None


def _active_keywords(fact_data: dict, at_cycle: int) -> list[str]:
    v = _active_version(fact_data, at_cycle)
    return list(v.get("keywords", [])) if v else []


def _discriminating_keywords(fact_data: dict, at_cycle: int) -> list[str]:
    """Keywords specific to the currently-active version — i.e., values that
    distinguish this version from OTHER versions of the same fact.

    For revised facts: returns the active version's keywords with the entity-
    name-like shared keywords (those present in earlier versions too) removed.
    For unrevised facts (single version): returns the full keyword list, since
    "discriminating from what" is vacuous — every keyword identifies the fact.

    This is critical for S1's keyword structure, where entity names like
    'Search Engine' appear in BOTH v1 and v2 and dominate naive substring-
    presence checks. Without this filter, revised facts can score the same
    "fidelity" as unrevised purely on the shared entity name, and the
    revision-specific signal is washed out.
    """
    versions = fact_data.get("versions") or []
    active = _active_version(fact_data, at_cycle)
    if not active:
        return []
    active_kws = active.get("keywords", []) or []
    if len(versions) <= 1:
        return list(active_kws)
    other_kws = set()
    for v in versions:
        if v.get("session", 0) <= at_cycle and v is not active:
            other_kws.update(v.get("keywords", []) or [])
    return [k for k in active_kws if k not in other_kws]


def _was_revised_by(fact_data: dict, at_cycle: int) -> bool:
    """True if at least one revision has landed by `at_cycle` (i.e., fact has
    a version-2 or later whose session ≤ at_cycle)."""
    versions = fact_data.get("versions") or []
    return any(v.get("version", 1) > 1 and v.get("session", 0) <= at_cycle
               for v in versions)


def _partition_by_revision(
    facts_dict: dict, at_cycle: int
) -> tuple[list[dict], list[dict]]:
    """Split facts into (revised_R, unrevised_U) as of `at_cycle`. Facts not
    yet introduced are excluded."""
    R: list[dict] = []
    U: list[dict] = []
    for f in facts_dict.values():
        if f.get("introduced_session", 0) > at_cycle:
            continue
        (R if _was_revised_by(f, at_cycle) else U).append(f)
    return R, U


def _fidelity_on_subset(
    memory_text_lower: str, subset: list[dict], at_cycle: int
) -> tuple[int, int]:
    """(n_survived, n_total) over a subset. A fact survives if any of its
    DISCRIMINATING keywords appears in memory (entity-name-like shared
    keywords are filtered out for revised facts; see ``_discriminating_keywords``)."""
    if not subset:
        return 0, 0
    survived = 0
    for f in subset:
        kws = _discriminating_keywords(f, at_cycle)
        if kws and any(_present(kw, memory_text_lower) for kw in kws):
            survived += 1
    return survived, len(subset)


def compute_revision_fidelity_excess(
    memory_text: str,
    facts_dict: dict,
    at_cycle: int,
    min_unrevised_for_signal: int = 3,
) -> Optional[float]:
    """``fidelity_unrevised − fidelity_revised``.

    Positive = revised facts decay faster than the never-revised baseline,
    meaning revision-specific aging above general compression drift. Returns
    None when the signal is underpowered (no revisions yet, or |unrevised|
    below threshold)."""
    R, U = _partition_by_revision(facts_dict, at_cycle)
    if not R or len(U) < min_unrevised_for_signal:
        return None
    text_lower = (memory_text or "").lower()
    s_R, n_R = _fidelity_on_subset(text_lower, R, at_cycle)
    s_U, n_U = _fidelity_on_subset(text_lower, U, at_cycle)
    if n_R == 0 or n_U == 0:
        return None
    return round((s_U / n_U) - (s_R / n_R), 4)


def compute_revision_fidelity_excess_count(
    memory_text: str,
    facts_dict: dict,
    at_cycle: int,
) -> Optional[int]:
    """Count form. ``actual_R_failures − expected_R_failures_at_baseline_rate``.

    Absolute count of revision-attributable failures over and above what
    compression alone would predict. Doesn't dilute as revised pool grows."""
    R, U = _partition_by_revision(facts_dict, at_cycle)
    if not R or not U:
        return None
    text_lower = (memory_text or "").lower()
    s_R, n_R = _fidelity_on_subset(text_lower, R, at_cycle)
    s_U, n_U = _fidelity_on_subset(text_lower, U, at_cycle)
    if n_U == 0:
        return None
    baseline_failure_rate = (n_U - s_U) / n_U
    actual_failures = n_R - s_R
    expected_failures = baseline_failure_rate * n_R
    return round(actual_failures - expected_failures)


def _stale_residue_count(
    memory_text_lower: str,
    revised_facts: list[dict],
    at_cycle: int,
) -> int:
    """Count revised facts exhibiting the stale-residue pattern: a purely-old
    keyword is in memory AND no currently-active keyword is in memory.

    This pattern cannot be produced by compression alone — compression only
    drops content. Stale-residue requires the compactor to have written the
    revision into memory but failed to remove the superseded value, then
    later compression dropped the new value while leaving the old. It is a
    pure revision-flow failure signature."""
    stale = 0
    for f in revised_facts:
        versions = f.get("versions") or []
        if len(versions) <= 1:
            continue
        original_kws = versions[0].get("keywords") or []
        active_set = set(_active_keywords(f, at_cycle))
        purely_old = [k for k in original_kws if k and k not in active_set]
        # Use NOVEL-only keywords for the new_present check — checking against
        # all active keywords would let shared entity names (e.g. 'Search Engine'
        # present in both v1 and v2) trivially satisfy new_present and prevent
        # stale_residue from ever firing.
        novel_only = _discriminating_keywords(f, at_cycle)
        if not purely_old or not novel_only:
            continue
        old_present = any(_present(k, memory_text_lower) for k in purely_old)
        new_present = any(_present(k, memory_text_lower) for k in novel_only)
        if old_present and not new_present:
            stale += 1
    return stale


def compute_stale_residue_rate(
    memory_text: str,
    facts_dict: dict,
    at_cycle: int,
) -> Optional[float]:
    """Fraction of revised-by-t facts exhibiting stale residue.

    Pure revision-failure signal — compression alone cannot produce this
    pattern. Returns None before any revision has occurred."""
    R, _ = _partition_by_revision(facts_dict, at_cycle)
    if not R:
        return None
    text_lower = (memory_text or "").lower()
    return round(_stale_residue_count(text_lower, R, at_cycle) / len(R), 4)


def compute_stale_residue_count_abs(
    memory_text: str,
    facts_dict: dict,
    at_cycle: int,
) -> int:
    """Absolute count form for aging curves — doesn't dilute as the revised
    pool grows."""
    R, _ = _partition_by_revision(facts_dict, at_cycle)
    if not R:
        return 0
    text_lower = (memory_text or "").lower()
    return _stale_residue_count(text_lower, R, at_cycle)


def score_revision_aging(
    memory_text: str,
    facts_dict: dict,
    at_cycle: int,
) -> dict:
    """Combined revision-aging snapshot at cycle `at_cycle`.

    coverage_verdict:
      ``"no_revisions"`` — no revisions applied yet; rates are None
      ``"underpowered"`` — |R| < 3 or |U| < 5; differential is noisy
      ``"adequate"``     — |R| ≥ 3 and |U| ≥ 5
      ``"strong"``       — |R| ≥ 8 and |U| ≥ 5
    """
    R, U = _partition_by_revision(facts_dict, at_cycle)
    n_R, n_U = len(R), len(U)
    if n_R == 0:
        verdict = "no_revisions"
    elif n_R < 3 or n_U < 5:
        verdict = "underpowered"
    elif n_R < 8:
        verdict = "adequate"
    else:
        verdict = "strong"
    return {
        "revision_fidelity_excess": compute_revision_fidelity_excess(
            memory_text, facts_dict, at_cycle
        ),
        "revision_fidelity_excess_count": compute_revision_fidelity_excess_count(
            memory_text, facts_dict, at_cycle
        ),
        "stale_residue_rate": compute_stale_residue_rate(
            memory_text, facts_dict, at_cycle
        ),
        "stale_residue_count": compute_stale_residue_count_abs(
            memory_text, facts_dict, at_cycle
        ),
        "n_revised": n_R,
        "n_unrevised": n_U,
        "coverage_verdict": verdict,
    }
