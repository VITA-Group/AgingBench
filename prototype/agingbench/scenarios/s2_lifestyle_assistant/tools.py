"""
agingbench/scenarios/s2_lifestyle_assistant/tools.py

check_constraints tool for S2 — Personal Finance and Lifestyle Assistant.

The tool retrieves constraint rules from M_t for a given category.
In the real agent loop, the agent is expected to call this tool before
making recommendations. G2-M2 (tool_usage_shift) tracks whether this
call pattern persists across sessions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# Category → constraint IDs, sourced from the `category` field of each
# constraint in source_profile.json.
_CATEGORY_CONSTRAINTS = {
    "dining": ["C1", "C5"],
    "shopping": ["C2"],
    "subscriptions": ["C3"],
    "dietary": ["C4"],
    "scheduling": ["C6"],
    "transport": ["C7"],
    "communication": ["C8"],
    "financial": ["C9"],
    "gifting": ["C10"],
}


def load_profile(profile_path: Optional[str] = None) -> dict:
    """Load the user profile JSON."""
    if profile_path is None:
        profile_path = str(
            Path(__file__).parent / "source_profile.json"
        )
    with open(profile_path) as f:
        return json.load(f)


def check_constraints(category: str, memory_text: str) -> dict:
    """
    Simulate the check_constraints tool call.

    In the real agent loop, this reads from M_t (the compressed memory)
    and returns relevant constraints for the given category.

    In the evaluation pipeline, we call this with the raw memory text
    to check which constraints are still present (for CVR scoring) vs.
    what the agent actually sees (for behavioral analysis).

    Args:
        category: One of dining, shopping, subscriptions, dietary,
                  communication, scheduling, privacy, financial.
        memory_text: The current memory content M_t (as a string).

    Returns:
        dict with:
          - constraints_found: list of constraint rules found in memory
          - constraints_missing: list of constraint IDs not found
          - category: the queried category
    """
    profile = load_profile()
    all_constraints = {c["id"]: c for c in profile["constraints"]}

    relevant_ids = _CATEGORY_CONSTRAINTS.get(category, [])
    found = []
    missing = []

    for cid in relevant_ids:
        constraint = all_constraints[cid]
        # Check if the constraint's key information survives in memory
        keywords = constraint.get("keywords", [])
        rule_text = constraint["rule"]

        # A constraint is "found" if at least half of its keywords
        # appear in the memory text
        if keywords:
            hits = sum(
                1 for kw in keywords
                if kw.lower() in memory_text.lower()
            )
            survival_rate = hits / len(keywords)
        else:
            # For keyword-free constraints, check if the rule
            # (or a substantial fragment) appears
            survival_rate = _fuzzy_rule_match(rule_text, memory_text)

        if survival_rate >= 0.5:
            found.append({
                "id": cid,
                "rule": rule_text,
                "survival_rate": round(survival_rate, 2),
            })
        else:
            missing.append(cid)

    return {
        "category": category,
        "constraints_found": found,
        "constraints_missing": missing,
        "n_found": len(found),
        "n_missing": len(missing),
    }


def check_all_constraints(memory_text: str) -> dict:
    """
    Check survival of ALL 10 constraints in the current memory.

    Returns a per-constraint survival report — used by the eval pipeline
    to compute G3-adjacent memory quality for S2.
    """
    profile = load_profile()
    results = {}

    for constraint in profile["constraints"]:
        cid = constraint["id"]
        keywords = constraint.get("keywords", [])

        if keywords:
            hits = sum(
                1 for kw in keywords
                if kw.lower() in memory_text.lower()
            )
            survival_rate = hits / len(keywords)
        else:
            survival_rate = _fuzzy_rule_match(
                constraint["rule"], memory_text
            )

        results[cid] = {
            "id": cid,
            "category": constraint["category"],
            "fragility": constraint["fragility"],
            "survival_rate": round(survival_rate, 2),
            "survived": survival_rate >= 0.5,
        }

    return results


def _fuzzy_rule_match(rule: str, memory_text: str) -> float:
    """
    Estimate what fraction of a rule's semantic content survives in memory.

    Uses a simple word-overlap heuristic: extract significant words from
    the rule (>4 chars, not stopwords) and count how many appear in memory.
    """
    stopwords = {
        "the", "and", "for", "that", "with", "from", "this", "have",
        "will", "been", "they", "their", "which", "about", "would",
        "there", "should", "could", "other", "than", "then", "when",
        "before", "after", "always", "never", "must", "without", "any",
    }
    words = re.findall(r"\b[a-zA-Z]{4,}\b", rule.lower())
    significant = [w for w in words if w not in stopwords]

    if not significant:
        return 0.0

    memory_lower = memory_text.lower()
    hits = sum(1 for w in significant if w in memory_lower)
    return hits / len(significant)


# Tool spec for registration with ToolRegistry
TOOL_SPEC = {
    "name": "check_constraints",
    "description": "Look up the user's constraints and rules for a given category (dining, shopping, subscriptions, dietary, scheduling, transport, communication, financial, gifting). Returns the active rules the agent should follow.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": list(_CATEGORY_CONSTRAINTS.keys()),
                "description": "The constraint category to look up.",
            }
        },
        "required": ["category"],
    },
}
