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

    def _match(k: str, t: str) -> bool:
        # Exact word-bounded match (no optional plural). The previous
        # `(?:es|s)?` suffix over-matched: keyword "john" hit "johns"
        # ("Johns Hopkins"), crediting recall for the wrong token (a
        # false-positive that hides aging). Exact match is the safe direction.
        return re.search(
            r"(?<![A-Za-z0-9])" + re.escape(k) + r"(?![A-Za-z0-9])",
            t,
        ) is not None

    if _match(kw, lower_text):
        return True
    # Comma-normalized numeric retry: '23,800' matches '23800' (and vice-versa).
    # Strip commas only BETWEEN digits so the word-boundary guard is preserved.
    kw_n = re.sub(r"(?<=\d),(?=\d)", "", kw)
    text_n = re.sub(r"(?<=\d),(?=\d)", "", lower_text)
    if kw_n == kw and text_n == lower_text:
        return False
    return _match(kw_n, text_n)


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


def _active_keywords(probe: dict, at_session=None) -> list:
    """Keywords valid at evaluation session ``at_session`` (time-correct
    revision scoring). When the probe carries ``keywords_history`` (a list of
    (session, kws) tuples recorded by the generator on each
    version_random_facts revision), return the latest entry whose session is
    <= ``at_session`` — so a probe re-asked BEFORE a revision is scored against
    the pre-revision value and one re-asked AFTER against the post-revision
    value. Falls back to ``probe["keywords"]`` when there is no history
    (non-revised probes) or no session. Mirrors s3 validator._active_keywords.
    """
    history = probe.get("keywords_history")
    if not history or at_session is None:
        return probe.get("keywords", [])
    active = probe.get("keywords", [])
    for sess, kws in history:
        if sess <= at_session:
            active = kws
        else:
            break
    return active


def score_recall_probe(agent_output: str, probe: dict, at_session=None) -> dict:
    """
    Score a single recall probe response.

    When ``at_session`` is given and the probe carries ``keywords_history``,
    scores against the value active at that evaluation session (time-correct
    revision scoring — fixes penalizing a correct contemporaneous answer);
    otherwise against ``probe["keywords"]``.

    Returns dict with:
      - probe_id: str
      - recalled: 1 if any keyword found, 0 otherwise
      - keywords: the keyword list actually scored against
    """
    active = _active_keywords(probe, at_session)
    recalled = score_keywords(agent_output, active)
    return {
        "probe_id": probe["probe_id"],
        "recalled": recalled,
        "keywords": active,
    }


def score_recall_batch(
    probe_outputs: list[str],
    probes: list[dict],
    at_session=None,
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
        per_probe.append(score_recall_probe(output, probe, at_session=at_session))

    n_recalled = sum(1 for p in per_probe if p["recalled"])
    return {
        "recall_rate": round(n_recalled / len(probes), 4) if probes else 1.0,
        "per_probe": per_probe,
        "n_recalled": n_recalled,
        "n_total": len(probes),
    }


def partition_recall(probe_results: list[dict]) -> dict:
    """De-blend a session's per-probe recall into the four-mechanism axes.

    The S6 headline used to be the recall rate over ALL probes, which silently
    folded three distinct mechanisms into one number:
      - interference (confusable-name binding probes),
      - revision (probes whose fact was re-versioned, carrying keywords_history),
      - compression (plain stable facts).
    It also (a) scored an empty probe pool as perfect recall (1.0) and
    (b) counted agent crash / turn-exhaustion (``mechanics_failure``) as a
    recall miss — an availability failure, not a memory outcome.

    This returns the cleanly separated rates (each ``None`` when its pool is
    empty, so the caller can SKIP rather than fabricate a score):

      - recall_all          : every live probe (back-compat blended view)
      - recall_excl_binding  : live, non-binding probes
      - recall_compression   : live, non-binding, non-revised (HEADLINE) — the
                               isolated compression signal
      - n_total / n_live / n_mechanics_failures : pool bookkeeping

    A probe is "live" when ``mechanics_failure`` is falsy. Binding probes carry
    ``probe_type == "interference_binding"``; revised probes carry a truthy
    ``is_revised`` flag (set by the runner from ``keywords_history``).
    """
    def _rate(ps):
        return (round(sum(p["recalled"] for p in ps) / len(ps), 4)
                if ps else None)

    live = [p for p in probe_results if not p.get("mechanics_failure")]
    non_binding = [p for p in live
                   if p.get("probe_type") != "interference_binding"]
    stable = [p for p in non_binding if not p.get("is_revised")]
    return {
        "recall_all": _rate(live),
        "recall_excl_binding": _rate(non_binding),
        "recall_compression": _rate(stable),
        "n_total": len(probe_results),
        "n_live": len(live),
        "n_mechanics_failures": len(probe_results) - len(live),
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
