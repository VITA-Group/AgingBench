"""
agingbench/scenarios/s1_research_literature/task_validator.py

Runs 10 compliance-decision tasks against the agent and scores each answer.

Two complementary aging signals:
  keyword_m(k)  — passive:  is the constraint value still present in compressed text?
  task_m(k)     — active:   can the agent apply it correctly to make a right decision?

Task format (tasks.jsonl):
  task_id, constraint_id, query, correct_answer, correct_value, wrong_if

Answer parsing:
  - "yes"/"no" tasks: look for the first yes/no word in the response
  - Open-answer tasks (e.g. "who must approve"): check if correct_value keywords appear

Deterministic: uses do_sample=False; answer is parsed via regex, no judge.
"""

from __future__ import annotations

import re
from typing import Optional


TASK_SYSTEM = """You are a project compliance checker. You have the following project specification in your memory:

{memory}

Answer each question based strictly on the specification above.
For yes/no questions: start your answer with YES or NO, then give one sentence of reasoning.
For open questions: state the name or value directly, then give one sentence of reasoning.
Be concise."""


def _parse_yes_no(text: str) -> Optional[str]:
    """Extract yes/no from the start of a response."""
    t = text.strip().lower()
    if re.match(r"\byes\b", t):
        return "yes"
    if re.match(r"\bno\b", t):
        return "no"
    # Also check first sentence
    first = re.split(r"[.!?\n]", t)[0]
    if "yes" in first.split():
        return "yes"
    if "no" in first.split():
        return "no"
    return None


def score_task(response: str, task: dict) -> int:
    """
    Returns 1 if the agent's response is correct, 0 otherwise.

    For yes/no tasks: parsed answer must match task["correct_answer"].
    For open-answer tasks (correct_answer is a name/value):
      check if task["correct_value"] keywords appear in the response.
    """
    correct = task["correct_answer"].lower()

    if correct in ("yes", "no"):
        parsed = _parse_yes_no(response)
        if parsed is None:
            return 0  # unparseable = wrong
        return int(parsed == correct)
    else:
        # Open answer — keyword check
        r = response.lower()
        return int(correct in r)


def run_tasks(
    tasks: list[dict],
    memory_text: str,
    llm,
) -> tuple[list[int], float, list[dict]]:
    """
    Run all tasks against the current memory snapshot.

    Parameters
    ----------
    tasks       : list of task dicts loaded from tasks.jsonl
    memory_text : current compressed memory (or source_doc at cycle 0)
    llm         : BaseLLM instance

    Returns
    -------
    scores      : list[int]  — 0 or 1 per task
    task_m      : float      — fraction correct
    details     : list of {task_id, query, response, correct, score}
    """
    system_msg = TASK_SYSTEM.format(memory=memory_text or "(empty)")
    scores = []
    details = []

    for task in tasks:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": task["query"]},
        ]
        response = llm.chat(messages)
        s = score_task(response, task)
        scores.append(s)
        details.append({
            "task_id": task["task_id"],
            "constraint_id": task["constraint_id"],
            "query": task["query"],
            "correct_answer": task["correct_answer"],
            "correct_value": task["correct_value"],
            "response": response[:200],
            "score": s,
        })

    task_m = sum(scores) / len(scores) if scores else 0.0
    return scores, task_m, details


_TREND_PROBE_SYSTEM = """You are answering questions about a series of engineering reports stored in your memory:

{memory}

Answer each question concisely using the information in memory. If a value has been updated, cite the latest value, not the original."""


def run_keyword_probes(
    probes: list[dict],
    memory_text: str,
    llm,
) -> tuple[list[int], list[dict]]:
    from .validator import score_probe
    system_msg = _TREND_PROBE_SYSTEM.format(memory=memory_text or "(empty)")
    scores, details = [], []
    for probe in probes:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": probe["question"]},
        ]
        response = llm.chat(messages)
        s = score_probe(response, probe)
        scores.append(s)
        details.append({
            "probe_id": probe["probe_id"],
            "response": response[:300],
            "score": s,
            "expected": probe["keywords"],
        })
    return scores, details


def run_trend_probes(
    trend_probes: list[dict],
    memory_text: str,
    llm,
) -> tuple[list[int], list[dict]]:
    from .validator import score_probe
    system_msg = _TREND_PROBE_SYSTEM.format(memory=memory_text or "(empty)")
    scores, details = [], []
    for probe in trend_probes:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": probe["question"]},
        ]
        response = llm.chat(messages)
        s = score_probe(response, probe)
        scores.append(s)
        details.append({
            "probe_id": probe["probe_id"],
            "response": response[:300],
            "score": s,
            "expected": probe["keywords"],
            "forbidden": probe.get("forbidden_keywords", []),
        })
    return scores, details


def report(details: list[dict], task_m: float) -> None:
    """Pretty-print task results for one cycle."""
    print(f"\n  Task compliance: {sum(d['score'] for d in details)}/{len(details)}  (task_m = {task_m:.3f})")
    for d in details:
        status = "OK  " if d["score"] else "FAIL"
        print(f"    [{status}] {d['task_id']}  expected={d['correct_answer']!r:<6}  "
              f"response={d['response'][:60]!r}")
