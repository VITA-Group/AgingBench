"""
S4 Software Engineering Agent — Validator

Scores agent modifications against ground-truth impact sets and test suites.
Produces G4 metrics: FASR, RR, CFR, LA, shock, recovery.
"""

from __future__ import annotations


def compute_files_modified(before_snapshot: dict, after_snapshot: dict) -> set[str]:
    """Detect which files the agent actually changed."""
    modified = set()
    all_files = set(before_snapshot.keys()) | set(after_snapshot.keys())
    for f in all_files:
        old = before_snapshot.get(f, "")
        new = after_snapshot.get(f, "")
        if old != new:
            modified.add(f)
    return modified


def compute_la(predicted_impact: set[str], actual_impact: set[str]) -> float:
    """G4-M4: Lookahead Accuracy — Jaccard similarity."""
    if not predicted_impact and not actual_impact:
        return 1.0
    intersection = predicted_impact & actual_impact
    union = predicted_impact | actual_impact
    return len(intersection) / len(union) if union else 0.0


def score_tests(test_results: dict[str, str]) -> tuple[set[str], set[str]]:
    """Split test results into passing and failing sets."""
    passing = {name for name, status in test_results.items() if status == "pass"}
    failing = {name for name, status in test_results.items() if status != "pass"}
    return passing, failing


def compute_cfr(passing_before: set[str], passing_after: set[str]) -> float:
    """G4-M3: Cascading Failure Rate — fraction of previously passing tests now failing."""
    if not passing_before:
        return 0.0
    broken = passing_before - passing_after
    return len(broken) / len(passing_before)


def score_session(
    task: dict,
    agent_output: str,
    files_modified: set[str],
    tests_before: dict[str, str],
    tests_after: dict[str, str],
    n_attempts: int = 1,
    predicted_impact: set[str] | None = None,
) -> dict:
    """
    Score one session's agent work.

    Returns dict with FASR, RR, CFR, LA for this session.
    """
    ground_truth_impact = set(task["impact_set"])

    # LA: did the agent predict the right files?
    if predicted_impact is None:
        predicted_impact = files_modified
    la = compute_la(predicted_impact, ground_truth_impact)

    # Test results
    passing_before, _ = score_tests(tests_before)
    passing_after, failing_after = score_tests(tests_after)

    # CFR: did the agent break previously passing tests?
    cfr = compute_cfr(passing_before, passing_after)

    # FASR: 1 if succeeded on first attempt, 0 otherwise
    fasr = 1.0 if n_attempts == 1 and not failing_after else 0.0

    # RR: number of revision cycles
    rr = float(n_attempts)

    return {
        "fasr": fasr,
        "rr": rr,
        "cfr": cfr,
        "la": la,
        "files_modified": sorted(files_modified),
        "ground_truth_impact": sorted(ground_truth_impact),
        "tests_passing": len(passing_after),
        "tests_failing": len(failing_after),
        "tests_broken": sorted(passing_before - passing_after),
    }
