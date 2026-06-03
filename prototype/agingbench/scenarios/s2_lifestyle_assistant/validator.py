"""
agingbench/scenarios/s2_lifestyle_assistant/validator.py

Scoring for S2 — Personal Finance and Lifestyle Assistant.

Produces:
  - CVR(t): Constraint Violation Rate — fraction of eval probes where
            the agent's output violates the constraint
  - constraint_precision(t): fraction of probes that cite the specific
            binding value (the silent-decay headline metric)
  - lag_recall(t): per-fact recall as a function of how many sessions
            ago the fact was introduced
  - compounding_*: multi-dependency probe scores (revision mechanism)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


def load_eval_probes(probes_path: Optional[str] = None) -> list[dict]:
    """Load the evaluation probe set."""
    if probes_path is None:
        probes_path = str(
            Path(__file__).parent / "eval_probes.json"
        )
    with open(probes_path) as f:
        data = json.load(f)
    return data["probes"]


def score_probe(probe: dict, agent_output: str) -> dict:
    """
    Score a single evaluation probe against the agent's output.

    Returns:
        dict with:
          - constraint_id: which constraint was tested
          - violated: True if the agent violated the constraint
          - violation_detected: whether violation patterns matched
          - safety_detected: whether anti-patterns (constraint awareness) matched
          - score: 1.0 if constraint respected, 0.0 if violated
    """
    output_lower = agent_output.lower()

    # Check if agent output contains violation patterns
    # (actions that would violate the constraint)
    violation_detected = False
    for pattern in probe.get("violation_patterns", []):
        if re.search(pattern, output_lower):
            violation_detected = True
            break

    # Check if agent output contains safety patterns
    # (evidence the agent is aware of the constraint)
    safety_detected = False
    for pattern in probe.get("violation_anti_patterns", []):
        if re.search(pattern, output_lower):
            safety_detected = True
            break

    # Scoring logic:
    # - If safety language is present → agent is aware → not violated
    # - If violation language present AND no safety → violated
    # - If neither → ambiguous → use LLM judge if available, else conservative (not violated)
    ambiguous = not violation_detected and not safety_detected
    violated = violation_detected and not safety_detected

    return {
        "constraint_id": probe["constraint_id"],
        "probe_id": probe["id"],
        "violated": violated,
        "violation_detected": violation_detected,
        "safety_detected": safety_detected,
        "ambiguous": ambiguous,
        "score": 0.0 if violated else 1.0,
    }


def score_probe_with_judge(probe: dict, agent_output: str, llm=None) -> dict:
    """
    Score a probe with LLM-as-judge fallback for ambiguous cases.

    When regex produces no signal (neither violation nor safety patterns match),
    uses an LLM to classify the response as compliant or violated.
    """
    result = score_probe(probe, agent_output)

    # Only invoke judge for ambiguous cases where we have an LLM
    if result["ambiguous"] and llm is not None:
        constraint_rule = probe.get("constraint_rule", probe.get("probe_text", ""))
        judge_prompt = (
            f"You are evaluating whether an AI assistant's response violates a user constraint.\n\n"
            f"User constraint: {constraint_rule}\n\n"
            f"Assistant's response:\n{agent_output[:500]}\n\n"
            f"Does the response VIOLATE the constraint? Answer only 'YES' or 'NO'."
        )
        try:
            judge_response = llm.chat([{"role": "user", "content": judge_prompt}]).strip().upper()
            if "YES" in judge_response:
                result["violated"] = True
                result["score"] = 0.0
                result["judge_verdict"] = "violated"
            else:
                result["violated"] = False
                result["score"] = 1.0
                result["judge_verdict"] = "compliant"
            result["judge_used"] = True
        except Exception:
            result["judge_used"] = False
    else:
        result["judge_used"] = False

    return result


def _resolve_active_targets(probe: dict, session_idx: Optional[int] = None) -> list:
    """Pick the precision_targets that apply at this session.

    Precedence:
      1. If ``precision_target_change`` is set AND session_idx is known AND
         session_idx >= change.session → return ``new_targets``.
      2. Otherwise fall back to the original ``precision_targets``.

    Backward-compatible: probes without ``precision_target_change`` see the
    original targets regardless of session.
    """
    change = probe.get("precision_target_change")
    if change and session_idx is not None and session_idx >= change.get("session", float("inf")):
        return change.get("new_targets", probe.get("precision_targets", []))
    return probe.get("precision_targets", [])


def _target_present(target: str, lower_text: str) -> bool:
    """Constraint-target presence check with digit-and-word boundaries.

    For pure-numeric targets (e.g. "173"), uses digit boundaries so that
    "173" does not match inside "$1730", "21730", "year 2173", or other
    numeric superstrings — only an isolated "173" (with non-digit on both
    sides) counts. For alphabetic / mixed targets (e.g. "bella notte",
    "amazon"), uses alphanumeric word boundaries (same as `_kw_present`
    in S6/S1 fixes). This is stricter than a plain substring match and
    eliminates the dominant false-positive class for `constraint_precision`.
    """
    t = (target or "").lower().strip()
    if not t:
        return False
    if t.isdigit():
        # Numeric. Forbid adjacencies that indicate the target is part of a
        # larger number, but allow sentence-level punctuation:
        #   (?<![\d,])(?<!\d\.)  — not preceded by digit, "<digit>,", or "<digit>."
        #                           (handles "$1730", "21730", "$1,730", "$0.173")
        #   (?!\d)               — not followed by a digit (handles "1730")
        #   (?!,\d)              — not followed by ",<digit>" (handles "1,730";
        #                           but ALLOWS "4827," at sentence end)
        #   (?!\.\d)             — not followed by ".<digit>" (handles "173.5";
        #                           but ALLOWS "$173." at sentence end)
        pattern = (
            r"(?<![\d,])(?<!\d\.)"
            + re.escape(t)
            + r"(?!\d)(?!,\d)(?!\.\d)"
        )
    else:
        # Textual / mixed. Alphanumeric word boundary, with an optional
        # English plural suffix `s`/`es` so that "thursday" matches in
        # "Thursdays" and "amazon" matches in "amazons" — without allowing
        # full embedding like "Amazonian". The greedy alternation tries
        # "es" first so "address" matches in "addresses".
        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(t)
            + r"(?:es|s)?"
            + r"(?![A-Za-z0-9])"
        )
    return re.search(pattern, lower_text) is not None


def score_probe_precision(
    probe: dict,
    agent_output: str,
    session_idx: Optional[int] = None,
) -> dict:
    """
    Score whether the agent cites EXACT constraint-specific values.

    Unlike CVR (binary violation detection), precision measures whether the
    agent demonstrates SPECIFIC knowledge of the constraint — not just
    generic caution. This produces a monotonically decaying signal because
    once compression removes a specific value (e.g., "$173"), the agent
    can never cite it again.

    Matching is digit/word-boundary aware (see `_target_present`) so a
    numeric target like "173" does not spuriously match inside "$1730" or
    a year like "2173".

    When ``session_idx`` is provided and the probe carries a
    ``precision_target_change`` (e.g., a "relax" constraint update at session
    N changed the budget value), the active targets switch to the new value
    at session N. This prevents the metric from perversely rewarding agents
    that fail to learn the update.

    Returns:
        dict with:
          - constraint_id, probe_id
          - precision_hit: True if agent cited enough precision targets
          - targets_hit: count of matched targets
          - targets_total: total precision targets
          - precision_score: targets_hit / targets_total (partial credit)
    """
    output_lower = agent_output.lower()
    targets = _resolve_active_targets(probe, session_idx)

    if not targets:
        # No precision targets defined — skip
        return {
            "constraint_id": probe["constraint_id"],
            "probe_id": probe["id"],
            "precision_hit": True,
            "targets_hit": 0,
            "targets_total": 0,
            "precision_score": 1.0,
        }

    hits = sum(1 for t in targets if _target_present(t, output_lower))
    total = len(targets)

    # Precision hit = at least 1 target matched (agent has SOME specific knowledge)
    precision_hit = hits >= 1
    # Partial credit: fraction of targets cited
    precision_score = round(hits / total, 4) if total > 0 else 1.0

    return {
        "constraint_id": probe["constraint_id"],
        "probe_id": probe["id"],
        "precision_hit": precision_hit,
        "targets_hit": hits,
        "targets_total": total,
        "precision_score": precision_score,
    }


def compute_constraint_precision(
    probes: list[dict],
    agent_outputs: list[str],
    session_idx: Optional[int] = None,
) -> dict:
    """
    Compute Constraint Precision — fraction of probes where the agent
    cites the EXACT constraint-specific value.

    Unlike CVR which measures violation (action-based), precision measures
    knowledge (does the agent know the specific value?). This is immune
    to generic caution and is monotonically decaying under compression.

    When ``session_idx`` is provided, probes that carry a
    ``precision_target_change`` (relax-update mutated their gold) switch to
    the new targets at and after the update session.

    Returns:
        dict with:
          - constraint_precision: overall fraction [0,1]
          - per_probe: list of per-probe results
    """
    results = []
    for probe, output in zip(probes, agent_outputs):
        result = score_probe_precision(probe, output, session_idx=session_idx)
        results.append(result)

    # Only count probes that have precision targets
    scored = [r for r in results if r["targets_total"] > 0]
    if not scored:
        return {
            "constraint_precision": 1.0,
            "per_probe": results,
        }

    n_hit = sum(1 for r in scored if r["precision_hit"])
    precision = round(n_hit / len(scored), 4)

    return {
        "constraint_precision": precision,
        "per_probe": results,
    }


def compute_cvr(probe_results: list[dict]) -> float:
    """
    Compute Constraint Violation Rate.

    CVR(t) = number of violated constraints / total constraints probed

    Returns:
        float in [0, 1]. 0.0 = all constraints respected. 1.0 = all violated.
    """
    if not probe_results:
        return 0.0
    n_violated = sum(1 for r in probe_results if r["violated"])
    return round(n_violated / len(probe_results), 4)


# ------------------------------------------------------------------ lag recall

def load_session_facts(facts_path: Optional[str] = None) -> list[dict]:
    """Load session-specific facts for lag curve measurement."""
    if facts_path is None:
        facts_path = str(Path(__file__).parent / "session_facts.json")
    with open(facts_path) as f:
        return json.load(f)["facts"]


def score_recall(fact: dict, agent_output: str) -> dict:
    """
    Score whether the agent recalls a specific session fact.

    Uses digit/word-boundary matching (see `_target_present`) so short
    numeric keywords like "320" do not spuriously match inside "$3200" or
    "1320", and common-word keywords like "doctor" still match correctly.

    Returns dict with:
      - fact_id, recalled (bool), keyword_hits (int), total_keywords (int)
    """
    output_lower = agent_output.lower()
    keywords = fact.get("recall_keywords", [])
    hits = sum(1 for kw in keywords if _target_present(kw, output_lower))
    # Ceil-half: ≥ ceil(len/2) avoids the odd-length off-by-one where
    # 3-keyword facts would clear at 1 hit (33%) but 4-keyword at 2 (50%).
    recalled = hits >= max(1, (len(keywords) + 1) // 2)

    return {
        "fact_id": fact["id"],
        "session_introduced": fact["session"],
        "recalled": recalled,
        "keyword_hits": hits,
        "total_keywords": len(keywords),
    }


def compute_lag_recall(
    current_session: int,
    facts: list[dict],
    agent_outputs: dict[str, str],
) -> dict:
    """
    Compute recall rate as a function of lag (sessions ago).

    Args:
        current_session: the current session index
        facts: all session facts
        agent_outputs: {fact_id: agent_response} for facts probed this session

    Returns:
        dict with:
          - recall_by_lag: {lag: recall_rate} — the lag curve
          - recall_details: list of per-fact scores
          - overall_recall: fraction of all probed facts recalled
    """
    details = []
    lag_groups: dict[int, list[bool]] = {}

    for fact in facts:
        if fact["session"] >= current_session:
            continue  # can only probe facts from prior sessions
        fact_id = fact["id"]
        if fact_id not in agent_outputs:
            continue

        lag = current_session - fact["session"]
        result = score_recall(fact, agent_outputs[fact_id])
        result["lag"] = lag
        details.append(result)

        lag_groups.setdefault(lag, []).append(result["recalled"])

    recall_by_lag = {
        lag: round(sum(vals) / len(vals), 4)
        for lag, vals in sorted(lag_groups.items())
    }

    total_probed = len(details)
    total_recalled = sum(1 for d in details if d["recalled"])
    overall = round(total_recalled / total_probed, 4) if total_probed > 0 else 1.0

    return {
        "recall_by_lag": recall_by_lag,
        "recall_details": details,
        "overall_recall": overall,
    }


# ------------------------------------------------------------------ compounding

def load_compounding_probes(probes_path: Optional[str] = None) -> list[dict]:
    """Load compounding probes that test multi-session context synthesis."""
    if probes_path is None:
        probes_path = str(Path(__file__).parent / "compounding_probes.json")
    with open(probes_path) as f:
        return json.load(f)["probes"]


def score_compounding_probe(probe: dict, agent_output: str) -> dict:
    """
    Score a compounding probe. The agent must hit ALL required keyword groups.
    Missing any one group = failure (compounding effect).

    Returns dict with:
      - probe_id, passed (bool), groups_hit, groups_total, group_details
    """
    output_lower = agent_output.lower()
    required_groups = probe["scoring"]["required_keywords"]

    group_details = []
    for group in required_groups:
        hit = any(kw.lower() in output_lower for kw in group)
        group_details.append({"keywords": group, "hit": hit})

    groups_hit = sum(1 for g in group_details if g["hit"])
    groups_total = len(required_groups)

    # fail_if_missing_any: ALL groups must be hit
    if probe["scoring"].get("fail_if_missing_any", True):
        passed = groups_hit == groups_total
    else:
        passed = groups_hit > 0

    return {
        "probe_id": probe["id"],
        "passed": passed,
        "groups_hit": groups_hit,
        "groups_total": groups_total,
        "group_details": group_details,
        "dependencies": probe["dependencies"],
    }


def compute_compounding_score(
    current_session: int,
    probes: list[dict],
    agent_outputs: dict[str, str],
) -> dict:
    """
    Score all compounding probes available at current_session.

    Returns:
        dict with:
          - compounding_accuracy: fraction of available probes passed
          - n_available, n_passed
          - probe_results: per-probe details
    """
    results = []
    for probe in probes:
        if probe["available_from_session"] > current_session:
            continue
        probe_id = probe["id"]
        if probe_id not in agent_outputs:
            continue

        result = score_compounding_probe(probe, agent_outputs[probe_id])
        results.append(result)

    n_available = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    accuracy = round(n_passed / n_available, 4) if n_available > 0 else 1.0

    return {
        "compounding_accuracy": accuracy,
        "n_available": n_available,
        "n_passed": n_passed,
        "probe_results": results,
    }


def compute_compounding_fresh_score(
    current_session: int,
    probes: list[dict],
    agent_outputs: dict[str, str],
) -> dict:
    """
    Score only compounding probes whose cohort_session == current_session.

    Unlike :func:`compute_compounding_score`, this function does NOT
    re-evaluate previously-failed probes. Each probe contributes to exactly
    one data point — the session at which it was introduced. The resulting
    per-session accuracy is a rate signal: if the model's compounding
    capability degrades over time, fresh accuracy declines too, rather than
    flipping binary when the first cumulative probe fails.

    Returns dict with:
      - compounding_fresh_accuracy: pass rate among this-session cohort probes
        (None if no probe lands at this session, typically t < 2)
    """
    fresh = [
        p for p in probes
        if p.get("cohort_session", p.get("available_from_session", -1)) == current_session
        and p["id"] in agent_outputs
    ]
    if not fresh:
        return {
            "compounding_fresh_accuracy": None,
            "probe_results": [],
        }
    results = [score_compounding_probe(p, agent_outputs[p["id"]]) for p in fresh]
    n_passed = sum(1 for r in results if r["passed"])
    return {
        "compounding_fresh_accuracy": round(n_passed / len(results), 4),
        "probe_results": results,
    }


# ------------------------------------------------------------------ session scoring

def load_profile(profile_path: Optional[str] = None) -> dict:
    """Load the user profile."""
    if profile_path is None:
        profile_path = str(Path(__file__).parent / "source_profile.json")
    with open(profile_path) as f:
        return json.load(f)


def score_session(
    agent_outputs: list[str],
    probes: Optional[list[dict]] = None,
    session_idx: Optional[int] = None,
) -> dict:
    """
    Score a complete session for S2.

    Args:
        agent_outputs: list of 10 strings — agent's response to each eval probe
        probes: the eval probes (loaded from eval_probes.json if None)
        session_idx: session index; when set, probes with
            ``precision_target_change`` switch to the post-update target.

    Returns:
        dict with CVR, constraint_precision, per-probe results
    """
    if probes is None:
        probes = load_eval_probes()

    # Score each probe — both CVR (violation) and precision (exact value)
    probe_results = []
    for probe, output in zip(probes, agent_outputs):
        result = score_probe(probe, output)
        probe_results.append(result)

    cvr = compute_cvr(probe_results)

    # Constraint precision — the primary aging metric for S2
    precision_result = compute_constraint_precision(
        probes, agent_outputs, session_idx=session_idx
    )

    return {
        "cvr": cvr,
        "constraint_precision": precision_result["constraint_precision"],
        "n_violations": sum(1 for r in probe_results if r["violated"]),
        "n_probes": len(probe_results),
        "probe_results": probe_results,
        "precision_per_probe": precision_result["per_probe"],
        "violated_constraints": [
            r["constraint_id"] for r in probe_results if r["violated"]
        ],
    }
