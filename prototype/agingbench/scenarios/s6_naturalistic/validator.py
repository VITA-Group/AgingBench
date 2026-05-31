"""
agingbench/scenarios/s6_naturalistic/validator.py — Scoring for S6.

Two scoring modes:
  1. Task evaluation: keyword matching against eval_keywords (primary task answer).
  2. Recall probe evaluation: keyword matching against probe keywords (memory recall).

Both use case-insensitive substring matching — same approach as S1 validator.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_S6_DIR = Path(__file__).parent


def load_session_tasks() -> list[dict]:
    """Load all session definitions from session_tasks.json."""
    with open(_S6_DIR / "session_tasks.json") as f:
        data = json.load(f)
    return data["sessions"]


def load_system_prompt() -> str:
    """Load the system prompt for the S6 agent."""
    with open(_S6_DIR / "session_tasks.json") as f:
        data = json.load(f)
    return data["system_prompt"]


def _kw_present(keyword: str, lower_text: str) -> bool:
    """Word-boundary-aware presence check.

    Prevents short-keyword substring collisions like "234" matching inside
    "$2340" or "1234", and "37" matching inside "$3700". Adjacent
    alphanumeric characters on either side of the keyword block the match;
    punctuation/whitespace boundaries allow it.
    """
    kw = (keyword or "").lower().strip()
    if not kw:
        return False
    return re.search(
        r"(?<![A-Za-z0-9])"
        + re.escape(kw)
        + r"(?:es|s)?"
        + r"(?![A-Za-z0-9])",
        lower_text,
    ) is not None


def score_keywords(text: str, keywords: list[str]) -> int:
    """Return 1 if ANY keyword is found (case-insensitive, word-bounded) in text, else 0."""
    text_lower = (text or "").lower()
    return int(any(_kw_present(kw, text_lower) for kw in keywords))


def score_task(agent_output: str, session: dict) -> dict:
    """
    Score the primary task output against eval_keywords.

    Returns dict with:
      - task_score: fraction of eval_keywords found in output (word-bounded match)
      - keywords_found: list of matched keywords
      - keywords_missing: list of unmatched keywords
    """
    eval_kws = session["task"].get("eval_keywords", [])
    if not eval_kws:
        return {"task_score": 1.0, "keywords_found": [], "keywords_missing": []}

    output_lower = (agent_output or "").lower()
    found, missing = [], []
    for kw in eval_kws:
        (found if _kw_present(kw, output_lower) else missing).append(kw)

    score = len(found) / len(eval_kws) if eval_kws else 1.0
    return {
        "task_score": round(score, 4),
        "keywords_found": found,
        "keywords_missing": missing,
    }


def score_recall_probe(agent_output: str, probe: dict) -> dict:
    """
    Score a single recall probe response.

    Returns dict with:
      - probe_id: str
      - recalled: 1 if any keyword found, 0 otherwise
      - keywords: the probe's keyword list
    """
    recalled = score_keywords(agent_output, probe["keywords"])
    return {
        "probe_id": probe["probe_id"],
        "recalled": recalled,
        "keywords": probe["keywords"],
    }


def score_recall_batch(
    probe_outputs: list[str],
    probes: list[dict],
) -> dict:
    """
    Score a batch of recall probes.

    Returns dict with:
      - recall_rate: fraction of probes where at least one keyword was recalled
      - per_probe: list of per-probe results
      - n_recalled: count of probes recalled
      - n_total: total probes
    """
    if not probes:
        return {
            "recall_rate": 1.0,
            "per_probe": [],
            "n_recalled": 0,
            "n_total": 0,
        }

    per_probe = []
    for output, probe in zip(probe_outputs, probes):
        per_probe.append(score_recall_probe(output, probe))

    n_recalled = sum(1 for p in per_probe if p["recalled"])
    return {
        "recall_rate": round(n_recalled / len(probes), 4) if probes else 1.0,
        "per_probe": per_probe,
        "n_recalled": n_recalled,
        "n_total": len(probes),
    }


def build_recall_matrix_entry(
    session_idx: int,
    all_sessions: list[dict],
    probe_results: list[dict],
) -> dict[int, float]:
    """
    Build one row of the recall matrix: at evaluation time `session_idx`,
    what is the recall rate for facts from each prior session?

    probe_results is a flat list of per-probe results for all probes run
    at this evaluation time.  We group them by origin session.

    Returns {origin_session: recall_rate} mapping.
    """
    # Map probe_id prefix (e.g. "s3_") to session_id
    row = {}
    for result in probe_results:
        pid = result["probe_id"]
        # Extract origin session from probe_id: "s3_p0" → 3
        origin = int(pid.split("_")[0][1:])
        row.setdefault(origin, []).append(result["recalled"])

    # Average recall per origin session
    return {
        origin: round(sum(scores) / len(scores), 4)
        for origin, scores in row.items()
    }
