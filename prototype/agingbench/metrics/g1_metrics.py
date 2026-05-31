"""
agingbench/metrics/g1_metrics.py — G1 Task Performance metrics.

Centralized metric functions for G1. Scenario validators
(s1_research_literature/validator.py) call these.

Metrics:
  G1-M1  keyword_m(t)   — fraction of keyword probes surviving in M_t
                           (task-independent: inspects memory text directly)
  G1-M2  task_m(t)       — fraction of tasks answered correctly using M_t
                           (task-dependent: requires agent inference)
  G1-M3  rationale_recall(t) — fraction of "why" probes answered with
                           specific rationale (deep tier, task-dependent)
"""

from __future__ import annotations

import re
from typing import Optional


# ─── G1-M1: keyword_m ───────────────────────────────────────────────────────

def _keyword_present(keyword: str, lower_text: str) -> bool:
    """Word-boundary-aware presence check for a single keyword.

    Avoids unanchored-substring false positives that would otherwise count
    short numeric keywords like "847" as present inside larger numbers like
    "1847" or "8470", and short year fragments like "14th" as present inside
    unrelated text.

    For multi-token keywords (e.g. "PostgreSQL 15") we anchor only the outer
    boundaries, so they still match when surrounded by punctuation/whitespace
    but not when embedded inside larger tokens.
    """
    kw = keyword.lower().strip()
    if not kw:
        return False
    # Word-boundary pattern: non-alphanumeric (or string boundary) on both
    # sides of the keyword. Allow optional English plural suffix `s`/`es`
    # so "amazon" matches "amazons" / "address" matches "addresses" without
    # admitting embedded matches like "amazonian".
    pattern = (
        r"(?<![A-Za-z0-9])"
        + re.escape(kw)
        + r"(?:es|s)?"
        + r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, lower_text) is not None


def compute_keyword_survival(
    memory_text: str,
    probe: dict,
) -> int:
    """
    Score a single keyword probe against memory text.

    Returns 1 if ANY keyword from probe["keywords"] appears in memory_text
    with word-boundary anchoring (case-insensitive). Word boundaries prevent
    false positives where a short keyword (e.g. "847") would otherwise match
    inside a longer token ("1847", "8470").

    This is task-independent: it inspects M_t directly without running the
    agent.
    """
    lower = memory_text.lower()
    return int(any(_keyword_present(kw, lower) for kw in probe["keywords"]))


def compute_keyword_m(
    memory_text: str,
    probes: list[dict],
) -> tuple[list[int], float]:
    """
    Compute keyword_m — fraction of keyword probes surviving in memory.

    Args:
        memory_text: current memory content M_t
        probes: list of probe dicts, each with "keywords" list

    Returns:
        (per_probe_scores, keyword_m) where keyword_m in [0, 1]
    """
    scores = [compute_keyword_survival(memory_text, p) for p in probes]
    m = sum(scores) / len(scores) if scores else 0.0
    return scores, round(m, 4)


def compute_cohort_keyword_m(
    memory_text: str,
    probes: list[dict],
    cohort_field: str = "cohort",
) -> dict:
    """
    Compute keyword_m grouped by cohort (e.g., paper batch index).

    Returns dict with:
      - overall_m: float
      - per_cohort: {cohort_id: keyword_m}
      - per_probe: list of {probe_id, cohort, survived}
    """
    cohort_groups: dict[int, list[int]] = {}
    details = []

    for p in probes:
        survived = compute_keyword_survival(memory_text, p)
        cohort = p.get(cohort_field, 0)
        cohort_groups.setdefault(cohort, []).append(survived)
        details.append({
            "probe_id": p.get("probe_id", p.get("id", "?")),
            "cohort": cohort,
            "survived": survived,
        })

    per_cohort = {
        c: round(sum(vals) / len(vals), 4)
        for c, vals in sorted(cohort_groups.items())
    }
    all_scores = [d["survived"] for d in details]
    overall = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    return {
        "overall_m": overall,
        "per_cohort": per_cohort,
        "per_probe": details,
    }


# ─── G1-M2: task_m ──────────────────────────────────────────────────────────

def parse_yes_no(text: str) -> Optional[str]:
    """Extract yes/no from the start of a response."""
    t = text.strip().lower()
    if re.match(r"\byes\b", t):
        return "yes"
    if re.match(r"\bno\b", t):
        return "no"
    first = re.split(r"[.!?\n]", t)[0]
    if "yes" in first.split():
        return "yes"
    if "no" in first.split():
        return "no"
    return None


def score_task_response(response: str, task: dict) -> int:
    """
    Score a single task response.

    For yes/no tasks: parsed answer must match task["correct_answer"].
    For open-answer tasks: check if correct_answer keywords appear in response.
    """
    correct = task["correct_answer"].lower()
    if correct in ("yes", "no"):
        parsed = parse_yes_no(response)
        return int(parsed == correct) if parsed is not None else 0
    else:
        return int(correct in response.lower())


def compute_task_m(
    task_scores: list[int],
) -> float:
    """
    Compute task_m — fraction of tasks answered correctly.

    Args:
        task_scores: list of 0/1 per task

    Returns:
        float in [0, 1]
    """
    if not task_scores:
        return 0.0
    return round(sum(task_scores) / len(task_scores), 4)


# ─── G1-M3: rationale_recall (deep tier) ────────────────────────────────────

def score_rationale_probe(
    agent_output: str,
    probe: dict,
) -> dict:
    """
    Score whether the agent can explain WHY a decision was made.

    The probe has:
      - "rationale_keywords": list of keywords that prove specific knowledge
      - "generic_patterns": patterns that indicate vague/guessed answers

    Returns dict with: probe_id, recalled, keywords_hit, is_generic
    """
    output_lower = agent_output.lower()

    rationale_kws = probe.get("rationale_keywords", [])
    hits = sum(1 for kw in rationale_kws if kw.lower() in output_lower)
    recalled = hits >= max(1, len(rationale_kws) // 2)

    generic_patterns = probe.get("generic_patterns", [])
    is_generic = any(re.search(p, output_lower) for p in generic_patterns)

    return {
        "probe_id": probe.get("id", "?"),
        "recalled": recalled and not is_generic,
        "keywords_hit": hits,
        "keywords_total": len(rationale_kws),
        "is_generic": is_generic,
    }


def compute_rationale_recall(
    probes: list[dict],
    agent_outputs: dict[str, str],
) -> dict:
    """
    Compute rationale_recall — fraction of "why" probes answered
    with specific rationale.

    Args:
        probes: list of rationale probe dicts
        agent_outputs: {probe_id: agent_response}

    Returns:
        dict with: rationale_recall (float), per_probe (list)
    """
    results = []
    for probe in probes:
        pid = probe.get("id", "?")
        if pid not in agent_outputs:
            continue
        result = score_rationale_probe(agent_outputs[pid], probe)
        results.append(result)

    if not results:
        return {"rationale_recall": 1.0, "per_probe": []}

    n_recalled = sum(1 for r in results if r["recalled"])
    recall = round(n_recalled / len(results), 4)

    return {
        "rationale_recall": recall,
        "per_probe": results,
    }
