"""
agingbench/scenarios/s2_hard/tools.py

check_constraints tool for S2-hard — 40 constraints across 13 categories.
Identical interface to s2_lifestyle_assistant/tools.py, just more categories.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


_CATEGORY_CONSTRAINTS = {
    "dining": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    "dietary": ["C4", "C5", "C6", "C7"],
    "shopping": ["C8", "C9", "C10", "C11"],
    "subscriptions": ["C12", "C13", "C14"],
    "communication": ["C15", "C16", "C17"],
    "scheduling": ["C18", "C19", "C20", "C21"],
    "privacy": ["C22", "C23", "C24", "C25"],
    "financial": ["C26", "C27", "C28", "C29", "C30"],
    "transportation": ["C31", "C32"],
    "health": ["C33", "C34"],
    "home": ["C35", "C36"],
    "pets": ["C37", "C38"],
    "entertainment": ["C39", "C40"],
}


def load_profile(profile_path: Optional[str] = None) -> dict:
    if profile_path is None:
        profile_path = str(Path(__file__).parent / "source_profile.json")
    with open(profile_path) as f:
        return json.load(f)


def check_constraints(category: str, memory_text: str) -> dict:
    profile = load_profile()
    all_constraints = {c["id"]: c for c in profile["constraints"]}

    relevant_ids = _CATEGORY_CONSTRAINTS.get(category, [])
    found = []
    missing = []

    for cid in relevant_ids:
        constraint = all_constraints[cid]
        keywords = constraint.get("keywords", [])
        rule_text = constraint["rule"]

        if keywords:
            hits = sum(
                1 for kw in keywords
                if kw.lower() in memory_text.lower()
            )
            survival_rate = hits / len(keywords)
        else:
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


TOOL_SPEC = {
    "name": "check_constraints",
    "description": "Look up the user's constraints and rules for a given category. Categories: dining, dietary, shopping, subscriptions, communication, scheduling, privacy, financial, transportation, health, home, pets, entertainment.",
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
