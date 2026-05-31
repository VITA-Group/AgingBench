"""
S3 Knowledge Base — Validator

Scores agent responses and memory state against the gold timeline.
Produces G3 metrics: summarization_fidelity, memory_bloat, contradiction_rate.
"""

from __future__ import annotations
import re
from typing import Optional


def _present(needle: str, haystack_lower: str) -> bool:
    """Digit-flank-safe substring presence so a short numeric keyword ("73")
    does not match inside a longer number ("731"). ``haystack_lower`` must be
    lowercased; word keywords keep plain substring semantics."""
    if not needle:
        return False
    return re.search(r"(?<!\d)" + re.escape(needle.lower()) + r"(?!\d)",
                     haystack_lower) is not None


def _active_keywords(decision: dict, at_session: Optional[int] = None) -> list[str]:
    """Pick the keywords valid at session ``at_session``.

    When the decision carries ``keywords_history`` (a list of (session, kws)
    tuples emitted by the generator each time version_random_facts updates the
    fact), choose the latest entry whose session is <= ``at_session``. Falls
    back to ``decision["keywords"]`` (original) when no history is present or
    no session is specified.

    Earlier the gold timeline froze at original keywords, so an agent
    that correctly adopted a revised value (e.g. $429,374 → $394,560) was
    marked as having LOST the decision, while an agent that kept stale
    residue scored a survival. This helper restores the correct sign.
    """
    history = decision.get("keywords_history")
    if not history or at_session is None:
        return decision.get("keywords", [])
    active: list[str] = decision.get("keywords", [])
    for sess, kws in history:
        if sess <= at_session:
            active = kws
        else:
            break
    return active


def score_query(response: str, query: dict) -> float:
    """
    Score a single query response using hybrid keyword + semantic matching.
    Returns float in [0, 1] — keyword match gives 1.0, semantic gives partial credit.
    """
    text = response.lower()
    # Keyword match (fast path)
    for kw in query["keywords"]:
        if kw.lower() in text:
            return 1.0

    # Semantic fallback (handles paraphrasing)
    try:
        from agingbench.metrics.semantic_scorer import semantic_score
        reference = query["question"] + " " + " ".join(query["keywords"])
        sim = semantic_score(response, reference, threshold=0.60)
        if sim >= 0.60:
            return sim
    except Exception:
        pass

    return 0.0


def score_queries(responses: list[str], queries: list[dict]) -> tuple[list[int], float]:
    """Score all queries, return (per-query scores, accuracy)."""
    scores = [score_query(r, q) for r, q in zip(responses, queries)]
    acc = sum(scores) / len(scores) if scores else 0.0
    return scores, acc


def compute_fidelity(
    memory_text: str,
    gold_decisions: list[dict],
    at_session: Optional[int] = None,
) -> float:
    """
    G3-M1: Summarization fidelity.
    Fraction of gold decisions whose CURRENTLY-active keywords survive in M_t.

    When ``at_session`` is provided, decisions with a ``keywords_history`` use
    their post-revision keywords (the value valid at that session). Otherwise
    the original keywords are used (back-compat).
    """
    text = memory_text.lower()
    survived = 0
    for d in gold_decisions:
        active = _active_keywords(d, at_session)
        for kw in active:
            if _present(kw, text):
                survived += 1
                break
    return survived / len(gold_decisions) if gold_decisions else 1.0


def compute_fidelity_detailed(
    memory_text: str,
    gold_decisions: list[dict],
    at_session: Optional[int] = None,
) -> dict:
    """
    Detailed fidelity: per-decision survival and by category.
    Uses hybrid keyword + semantic scoring for partial credit.

    When ``at_session`` is provided, decisions with a ``keywords_history`` use
    their post-revision keywords for both the keyword pass and the
    semantic-survival call.
    """
    try:
        from agingbench.metrics.semantic_scorer import score_fact_survival
        use_semantic = True
    except Exception:
        use_semantic = False

    text = memory_text.lower()
    per_decision = {}
    by_category = {}

    for d in gold_decisions:
        active = _active_keywords(d, at_session)
        if use_semantic:
            score = score_fact_survival(memory_text, d["fact"], active)
        else:
            score = 1.0 if any(_present(kw, text) for kw in active) else 0.0

        per_decision[d["id"]] = score
        cat = d.get("category", "other")
        if cat not in by_category:
            by_category[cat] = {"score_sum": 0.0, "total": 0}
        by_category[cat]["total"] += 1
        by_category[cat]["score_sum"] += score

    fidelity = sum(per_decision.values()) / len(per_decision) if per_decision else 1.0
    category_fidelity = {
        cat: v["score_sum"] / v["total"] for cat, v in by_category.items()
    }
    return {
        "fidelity": fidelity,
        "per_decision": per_decision,
        "category_fidelity": category_fidelity,
    }


def detect_contradictions(memory_text: str, gold_decisions: list[dict]) -> list[dict]:
    """
    G3-M3: Detect contradictions in memory against gold timeline.

    Two detection methods:
    1. Hardcoded staleness checks for known timeline updates
    2. Semantic contradiction detection: finds memory sentences that are
       semantically similar to a gold fact but contain different numerical values

    Returns list of detected contradictions.
    """
    contradictions = []
    text = memory_text.lower()

    # --- Method 1: Hardcoded staleness checks for known updates ---
    timeline_updates = [
        ("june 15", "june 28", "D20", "Phase 1 deadline update session 4"),
        ("june 28", "july 18", "D45", "Phase 1 deadline update session 10"),
        ("45,000", "38,000", "D21", "Contingency reduction session 4"),
        ("40", "32", "D47", "Component target reduction session 10"),
    ]

    for old_val, new_val, did, desc in timeline_updates:
        if old_val in text and new_val not in text:
            contradictions.append({
                "decision_id": did,
                "type": "stale",
                "description": f"Memory has old value '{old_val}' but not updated '{new_val}': {desc}",
            })

    # --- Method 2: Semantic contradiction detection ---
    # For numerical facts, check if memory contains a semantically similar
    # sentence but with a DIFFERENT number than the gold value.
    _detect_semantic_contradictions(memory_text, gold_decisions, contradictions)

    return contradictions


def _extract_numbers(text: str) -> set[str]:
    """Extract all number-like tokens from text."""
    return set(re.findall(r'\$?[\d,]+(?:\.\d+)?%?', text))


def _detect_semantic_contradictions(
    memory_text: str,
    gold_decisions: list[dict],
    contradictions: list[dict],
) -> None:
    """
    Find sentences in memory that are semantically similar to a gold fact
    but contain different numerical values — indicating the LLM hallucinated
    or merged numbers during compression.
    """
    try:
        from agingbench.metrics.semantic_scorer import _get_model, cosine_similarity
        model = _get_model()
        if model is None:
            return
    except Exception:
        return

    # Split memory into sentences
    sentences = [s.strip() for s in memory_text.replace('\n', '. ').split('.')
                 if len(s.strip()) > 15]
    if not sentences:
        return

    # Only check facts with numerical keywords
    numerical_facts = [
        d for d in gold_decisions
        if any(re.search(r'\d', kw) for kw in d["keywords"])
    ]
    if not numerical_facts:
        return

    # Encode all sentences and facts
    fact_texts = [d["fact"] for d in numerical_facts]
    all_texts = sentences + fact_texts
    embeddings = model.encode(all_texts)
    sent_embs = embeddings[:len(sentences)]
    fact_embs = embeddings[len(sentences):]

    for i, fact in enumerate(numerical_facts):
        gold_numbers = _extract_numbers(fact["fact"])
        if not gold_numbers:
            continue

        # Find most similar sentence in memory
        best_sim = 0.0
        best_sent = ""
        for j, sent in enumerate(sentences):
            sim = cosine_similarity(fact_embs[i], sent_embs[j])
            if sim > best_sim:
                best_sim = sim
                best_sent = sent

        # If high similarity but different numbers → contradiction
        if best_sim >= 0.55:
            sent_numbers = _extract_numbers(best_sent)
            if sent_numbers and gold_numbers:
                # Check if ANY gold number is absent but a different number is present
                missing_gold = gold_numbers - sent_numbers
                extra_in_memory = sent_numbers - gold_numbers
                if missing_gold and extra_in_memory:
                    contradictions.append({
                        "decision_id": fact["id"],
                        "type": "semantic_mismatch",
                        "description": (
                            f"Memory sentence similar to '{fact['fact'][:60]}...' "
                            f"but has numbers {extra_in_memory} instead of {missing_gold}"
                        ),
                        "similarity": round(best_sim, 3),
                        "memory_sentence": best_sent[:100],
                    })


def compute_contradiction_rate(memory_text: str, gold_decisions: list[dict]) -> float:
    """
    G3-M3: Fraction of checkable decisions that are contradicted.
    Uses both hardcoded staleness checks (4 known updates) and
    semantic contradiction detection (numerical facts).

    Note: for aging curves, use compute_contradiction_count() instead —
    the rate can decrease as denominator grows, masking accumulation.
    """
    contradictions = detect_contradictions(memory_text, gold_decisions)
    contradicted_ids = set(c["decision_id"] for c in contradictions)
    numerical_facts = [
        d for d in gold_decisions
        if any(re.search(r'\d', kw) for kw in d["keywords"])
    ]
    n_checkable = max(len(numerical_facts), 4)
    return len(contradicted_ids) / n_checkable if n_checkable > 0 else 0.0


def compute_contradiction_count(memory_text: str, gold_decisions: list[dict]) -> int:
    """
    G3-M3 (preferred): Absolute count of contradicted decisions.

    Unlike contradiction_rate, this metric can only increase as more
    decisions accumulate and more stale/mismatched values appear.
    Use for aging curves; the rate version is for per-session context.
    """
    contradictions = detect_contradictions(memory_text, gold_decisions)
    contradicted_ids = set(c["decision_id"] for c in contradictions)
    return len(contradicted_ids)


# ---------------------------------------------------------------------------
# Revision-aging trident (G3-M3+): three concepts × {rate, count}.
#
# General fidelity decay conflates revision failure with compression and
# interference. These signals isolate the revision component:
#
#   (1) fidelity excess  — paired control (revised vs never-revised). Cancels
#       compression/interference because both populations are exposed to the
#       same summarisation pressure; the residual is revision-specific decay.
#
#   (2) stale residue    — old kw present AND new kw absent. Pattern
#       compression cannot produce (compression removes both); interference
#       cannot either (no selective preservation of obsoleted values).
#
# Each is exposed in rate AND count form. Rates are useful for per-session
# human readability but DILUTE as the revision pool grows; counts are stable
# and belong in aging curves. Mirrors the existing contradiction_rate /
# contradiction_count split — same reason.
# ---------------------------------------------------------------------------


def _was_revised_by(decision: dict, at_session: int) -> bool:
    """True iff the decision had at least one revision applied at session
    ≤ ``at_session``. The generator appends a ``(session, new_keywords)``
    tuple to ``keywords_history`` on every revision; the first entry is the
    initial-creation snapshot, entries beyond index 0 are revisions."""
    history = decision.get("keywords_history") or []
    if len(history) <= 1:
        return False
    return history[1][0] <= at_session


def _partition_by_revision(
    gold_decisions: list[dict], at_session: int
) -> tuple[list[dict], list[dict]]:
    """Split into (revised_by_t, never_revised_by_t)."""
    R: list[dict] = []
    U: list[dict] = []
    for d in gold_decisions:
        (R if _was_revised_by(d, at_session) else U).append(d)
    return R, U


def _fidelity_on_subset(
    memory_text_lower: str, subset: list[dict], at_session: int
) -> tuple[int, int]:
    """(n_survived, n_total) over a subset using each decision's active
    keywords at ``at_session``."""
    if not subset:
        return 0, 0
    survived = 0
    for d in subset:
        active = _active_keywords(d, at_session)
        if any(_present(kw, memory_text_lower) for kw in active):
            survived += 1
    return survived, len(subset)


def compute_revision_fidelity_excess(
    memory_text: str,
    gold_decisions: list[dict],
    at_session: int,
    min_unrevised_for_signal: int = 5,
) -> Optional[float]:
    """Rate form. ``fidelity_unrevised - fidelity_revised``.

    Positive and rising = revised facts decay faster than never-revised
    baseline = revision-specific aging above general drift. Returns None
    when undersampled (no revisions yet or |unrevised| below threshold).
    """
    R, U = _partition_by_revision(gold_decisions, at_session)
    if not R or len(U) < min_unrevised_for_signal:
        return None
    text = memory_text.lower()
    surv_R, n_R = _fidelity_on_subset(text, R, at_session)
    surv_U, n_U = _fidelity_on_subset(text, U, at_session)
    return (surv_U / n_U) - (surv_R / n_R)


def compute_revision_fidelity_excess_count(
    memory_text: str,
    gold_decisions: list[dict],
    at_session: int,
) -> Optional[int]:
    """Count form. ``actual_R_failures - expected_R_failures_at_baseline_rate``.

    Excess revision-attributable failures over and above what compression
    alone would predict from the never-revised baseline. Does not dilute
    as the revision pool grows. Returns None when there's no baseline
    (no unrevised decisions yet) or no revisions to score.
    """
    R, U = _partition_by_revision(gold_decisions, at_session)
    if not R or not U:
        return None
    text = memory_text.lower()
    surv_R, n_R = _fidelity_on_subset(text, R, at_session)
    surv_U, n_U = _fidelity_on_subset(text, U, at_session)
    baseline_failure_rate = (n_U - surv_U) / n_U
    actual_failures = n_R - surv_R
    expected_failures = baseline_failure_rate * n_R
    return round(actual_failures - expected_failures)


def _stale_residue_decisions(
    memory_text_lower: str,
    revised_decisions: list[dict],
    at_session: int,
) -> int:
    """Count revised decisions exhibiting the stale-residue pattern:
    a superseded keyword present in memory AND none of the currently-active
    keywords present."""
    stale = 0
    for d in revised_decisions:
        history = d.get("keywords_history") or []
        if len(history) <= 1:
            continue
        original_kws = history[0][1] or []
        active_kws = _active_keywords(d, at_session)
        active_set = {k for k in active_kws}
        purely_old = [k for k in original_kws if k and k not in active_set]
        if not purely_old:
            continue
        old_present = any(_present(k, memory_text_lower) for k in purely_old)
        new_present = any(_present(k, memory_text_lower) for k in active_kws)
        if old_present and not new_present:
            stale += 1
    return stale


def compute_stale_residue_rate(
    memory_text: str,
    gold_decisions: list[dict],
    at_session: int,
) -> Optional[float]:
    """Rate form. Fraction of revised-by-t decisions where a superseded
    value lingers in memory while the current value is absent.

    Pure revision-failure signal — compression cannot produce this pattern.
    Returns None before any revision has occurred."""
    R, _ = _partition_by_revision(gold_decisions, at_session)
    if not R:
        return None
    return _stale_residue_decisions(memory_text.lower(), R, at_session) / len(R)


def compute_stale_residue_count(
    memory_text: str,
    gold_decisions: list[dict],
    at_session: int,
) -> int:
    """Count form. Absolute number of revised decisions exhibiting stale
    residue. Use for aging curves — does not dilute as more revisions land."""
    R, _ = _partition_by_revision(gold_decisions, at_session)
    if not R:
        return 0
    return _stale_residue_decisions(memory_text.lower(), R, at_session)


def score_revision_aging(
    memory_text: str,
    gold_decisions: list[dict],
    at_session: int,
) -> dict:
    """Combined revision-aging snapshot at session ``at_session``.

    Returns both rate and count forms of each trident signal, plus
    partition sizes and a coverage verdict so downstream consumers can
    honestly degrade when the signal is underpowered.

    coverage_verdict:
      ``"no_revisions"``  — no revisions applied yet; rates are None
      ``"underpowered"``  — |R| < 3 or |U| < 5; differential noisy
      ``"adequate"``      — |R| ≥ 3 and |U| ≥ 5
      ``"strong"``        — |R| ≥ 8 and |U| ≥ 5
    """
    R, U = _partition_by_revision(gold_decisions, at_session)
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
            memory_text, gold_decisions, at_session
        ),
        "revision_fidelity_excess_count": compute_revision_fidelity_excess_count(
            memory_text, gold_decisions, at_session
        ),
        "stale_residue_rate": compute_stale_residue_rate(
            memory_text, gold_decisions, at_session
        ),
        "stale_residue_count": compute_stale_residue_count(
            memory_text, gold_decisions, at_session
        ),
        "n_revised": n_R,
        "n_unrevised": n_U,
        "coverage_verdict": verdict,
    }


def score_session(
    memory_text: str,
    query_responses: list[str],
    queries: list[dict],
    gold_decisions_so_far: list[dict],
    at_session: Optional[int] = None,
) -> dict:
    """
    Score one session: query accuracy + fidelity + bloat + contradiction +
    revision-aging trident.

    ``at_session`` is required for the time-versioned signals (fidelity's
    ``_active_keywords`` selector and the revision-aging trident's
    partition). Pass the integer session index. For back-compat with
    callers that don't yet pass it, the time-versioned signals fall back
    to original-keyword scoring and the revision_aging block reports
    ``coverage_verdict="no_revisions"``.
    """
    query_scores, query_acc = score_queries(query_responses, queries)
    fidelity_detail = compute_fidelity_detailed(
        memory_text, gold_decisions_so_far, at_session=at_session
    )
    contradiction_rate = compute_contradiction_rate(memory_text, gold_decisions_so_far)
    contradiction_count = compute_contradiction_count(memory_text, gold_decisions_so_far)
    # Revision-aging trident — None coverage when no at_session passed.
    if at_session is not None:
        revision_aging = score_revision_aging(
            memory_text, gold_decisions_so_far, at_session=at_session
        )
    else:
        revision_aging = {
            "revision_fidelity_excess": None,
            "revision_fidelity_excess_count": None,
            "stale_residue_rate": None,
            "stale_residue_count": 0,
            "n_revised": 0,
            "n_unrevised": len(gold_decisions_so_far),
            "coverage_verdict": "no_revisions",
        }

    # Memory bloat: simple character count (token count deferred to runner)
    bloat = len(memory_text)

    return {
        "query_accuracy": query_acc,
        "query_scores": query_scores,
        "fidelity": fidelity_detail["fidelity"],
        "category_fidelity": fidelity_detail["category_fidelity"],
        "contradiction_rate": contradiction_rate,
        "contradiction_count": contradiction_count,
        "revision_aging": revision_aging,
        "memory_bloat_chars": bloat,
    }
