"""Regression tests for two S1 generator/scorer fixes (2026-05-30).

Bug 1 — phantom keywords inflating m0.
  ``S1Generator._extract_unique_keywords`` previously read values from a fixed
  key list in `vals` without checking whether the chosen batch template
  actually rendered those values. Templates like Security Audit and Migration
  populate only a subset of the value pool, so 30-50% of cycle-0 keywords
  were never recoverable from the content, producing a structural m0=0.5
  floor across the fig5a 20-session OS-model runs.

Bug 2 — version_accuracy proxying through session-wide keyword_m.
  Even when the generator emitted trend dep tasks (``dependency_type='trend'``,
  ``fact_versions_required[fid]>1``), ``dependency_scorer.version_accuracy``
  pulled the SESSION-WIDE primary score (``keyword_m``) as a correctness
  proxy rather than the per-probe verdict. The metric correlated with general
  recall, not with actually citing the latest version.

These tests pin: (a) every extracted keyword appears in the rendered content,
(b) trend probes carry forbidden_keywords (the stale pre-revision value),
(c) the validator scores 0 when the agent cites the forbidden value, and
(d) version_accuracy uses per-probe trend_probe_results when present.
"""
from __future__ import annotations

from agingbench.generators.pressure_config import PressureConfig
from agingbench.generators.s1_generator import S1Generator
from agingbench.scenarios.s1_research_literature.validator import (
    score_probe, score_all,
)
from agingbench.metrics.dependency_scorer import version_accuracy


def _build_session_lookup_from_results(results):
    return {r["session"]: r for r in results}


# ---------------------------------------------------------------------------
# Bug 1: keyword extraction matches content
# ---------------------------------------------------------------------------

def test_every_extracted_keyword_appears_in_its_batch_content():
    """For every batch, every keyword must be a substring of content.

    Pre-fix: Security Audit / Migration templates emitted phantom keywords
    that were not in content, producing the m0=0.5 floor.
    """
    p = PressureConfig.medium(); p.warmup_sessions = 0
    r = S1Generator(seed=42, pressure=p).generate(n_sessions=8)

    for b in r["paper_batches"]["batches"]:
        content_lower = b["content"].lower()
        for kw in b["keywords"]:
            assert kw.lower() in content_lower, (
                f"phantom keyword {kw!r} for cycle {b['cycle']} "
                f"({b['title'][:40]}): not in content"
            )


def test_m0_reaches_1_when_no_aging_pressure():
    """Under PressureConfig.none() and a fixed eval_text == batch content, the
    snapshot validator should find every keyword (m_snapshot == 1.0). Pre-fix,
    several template draws produced m=0.5 here even though no aging applied.
    """
    p = PressureConfig.none()
    r = S1Generator(seed=42, pressure=p).generate(n_sessions=5)
    batches = r["paper_batches"]["batches"]

    for b in batches:
        # Take only this cycle's probes (probe_id encodes cycle).
        cycle_probes = [
            pr for pr in r["probes"]
            if pr["probe_id"].startswith(f"s1_c{b['cycle']}_p")
        ]
        if not cycle_probes:
            continue
        _, m = score_all(b["content"], cycle_probes)
        assert m == 1.0, (
            f"cycle {b['cycle']}: expected m=1.0 against own content, got {m}"
        )


# ---------------------------------------------------------------------------
# Bug 2: faithful revision-via-DAG scoring
# ---------------------------------------------------------------------------

def test_trend_probes_carry_forbidden_pre_revision_keyword():
    """Trend dep probes must carry forbidden_keywords (= pre-revision value)
    so the validator can fail responses citing the stale version."""
    # Use heavy update_rate + low warmup so at least one trend probe fires.
    p = PressureConfig.medium()
    p.warmup_sessions = 0
    p.dependency_density = 1.0
    p.update_rate = 1.0
    r = S1Generator(seed=42, pressure=p).generate(n_sessions=10)

    trend_probes = [pr for pr in r["probes"] if pr.get("dep_type") == "trend"]
    assert trend_probes, "generator did not emit any trend dep probes"
    for tp in trend_probes:
        assert tp.get("forbidden_keywords"), (
            f"trend probe {tp['probe_id']} missing forbidden_keywords"
        )
        # Forbidden value must not also be in the canonical expected keywords —
        # otherwise scoring would be undefined.
        for fk in tp["forbidden_keywords"]:
            assert fk not in tp["keywords"], (
                f"trend probe {tp['probe_id']}: forbidden value {fk!r} "
                f"overlaps expected keywords"
            )


def test_validator_normalizes_format_equivalences():
    """Format normalization should make '73.9%' match '73.9 percent',
    '40,922' match '40922', '156ms' match '156 milliseconds', and
    '$429,374' match '429,374 dollars'."""
    pct = {"probe_id": "p", "keywords": ["73.9%"]}
    assert score_probe("the cache hit rate was 73.9 percent", pct) == 1
    assert score_probe("the cache hit rate was 73.9%", pct) == 1
    assert score_probe("around 74 percent", pct) == 0   # different number

    big_int = {"probe_id": "p", "keywords": ["40,922"]}
    assert score_probe("processed 40922 requests per second", big_int) == 1
    assert score_probe("processed 40,922 requests per second", big_int) == 1
    assert score_probe("processed 41,000 requests", big_int) == 0

    latency = {"probe_id": "p", "keywords": ["156ms"]}
    assert score_probe("the latency was 156 milliseconds", latency) == 1
    assert score_probe("the latency was 156ms", latency) == 1
    assert score_probe("the latency was 156 ms", latency) == 1

    dollars = {"probe_id": "p", "keywords": ["$429,374"]}
    assert score_probe("budget of 429,374 dollars", dollars) == 1
    assert score_probe("budget of $429374", dollars) == 1
    assert score_probe("budget of $430,000", dollars) == 0


def test_validator_normalizes_forbidden_keywords_too():
    """forbidden_keywords also go through normalization so a stale percent
    cited as 'X percent' is correctly penalized."""
    probe = {"probe_id": "p", "keywords": ["47%"], "forbidden_keywords": ["73.9%"]}
    assert score_probe("the updated rate is 47%", probe) == 1
    assert score_probe("it was 73.9 percent originally, now 47 percent", probe) == 0
    assert score_probe("it was 73.9% originally, now 47%", probe) == 0


def test_validator_penalizes_forbidden_keyword():
    """Citing the pre-revision keyword must yield score=0 even when the
    expected (post-revision) keyword is also present."""
    probe = {
        "probe_id": "test_trend",
        "keywords": ["NEW_VALUE"],
        "forbidden_keywords": ["OLD_VALUE"],
        "dep_type": "trend",
    }
    # Just the expected → 1
    assert score_probe("the answer is NEW_VALUE", probe) == 1
    # Only forbidden → 0
    assert score_probe("the answer is OLD_VALUE", probe) == 0
    # BOTH present → 0 (forbidden penalty wins)
    assert score_probe("was OLD_VALUE, now NEW_VALUE", probe) == 0
    # Neither → 0
    assert score_probe("totally unrelated", probe) == 0


def test_version_accuracy_uses_per_probe_trend_results_when_present():
    """When session_results carry trend_probe_results, version_accuracy must
    aggregate those binary verdicts instead of falling back to the session-wide
    keyword_m proxy."""
    graph = {
        "tasks": {
            "t1": {
                "session": 3,
                "dependency_type": "trend",
                "fact_versions_required": {"f1": 2},
            },
            "t2": {
                "session": 5,
                "dependency_type": "trend",
                "fact_versions_required": {"f2": 3},
            },
        },
        "facts": {"f1": {}, "f2": {}},
    }
    # Two sessions, each with the *session-wide proxy* (keyword_m) being LOW
    # (would force 0/N under legacy path) but trend probes themselves SUCCEED.
    session_results = [
        {
            "session": 3,
            "keyword_m": 0.1,
            "trend_probe_results": [{"probe_id": "t1", "score": 1.0}],
        },
        {
            "session": 5,
            "keyword_m": 0.1,
            "trend_probe_results": [{"probe_id": "t2", "score": 1.0}],
        },
    ]
    lookup = _build_session_lookup_from_results(session_results)
    va = version_accuracy(lookup, graph["tasks"], graph["facts"])
    assert va == 1.0, (
        f"per-probe path should give 1.0 even though session proxy is 0.1, "
        f"got {va}"
    )


def test_version_accuracy_falls_back_to_proxy_when_no_trend_results():
    """Backward compatibility: when trend_probe_results is absent (older
    runners or runs without the new field), version_accuracy should still
    work via the session-wide proxy path."""
    graph = {
        "tasks": {
            "t1": {
                "session": 3,
                "dependency_type": "trend",
                "fact_versions_required": {"f1": 2},
            },
        },
        "facts": {"f1": {}},
    }
    # No trend_probe_results → falls back to keyword_m proxy. 0.6 > 0.5 = pass.
    session_results = [{"session": 3, "keyword_m": 0.6}]
    lookup = _build_session_lookup_from_results(session_results)
    assert version_accuracy(lookup, graph["tasks"], graph["facts"]) == 1.0

    # 0.2 < 0.5 = fail under proxy path.
    session_results = [{"session": 3, "keyword_m": 0.2}]
    lookup = _build_session_lookup_from_results(session_results)
    assert version_accuracy(lookup, graph["tasks"], graph["facts"]) == 0.0


class _StubLLM:
    """Minimal LLM stub: chat() returns a canned answer keyed off the
    user message. Used to test response-based trend scoring without API calls."""
    def __init__(self, responses: dict[str, str]):
        self._responses = responses
        self.tracer = None
    def chat(self, messages):
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        for substr, reply in self._responses.items():
            if substr in user:
                return reply
        return "I don't know."


def test_run_trend_probes_scores_response_against_keywords_and_forbidden():
    from agingbench.scenarios.s1_research_literature.task_validator import (
        run_trend_probes,
    )
    probes = [
        {"probe_id": "p1", "question": "What is the current latency for X?",
         "keywords": ["120ms"], "forbidden_keywords": ["200ms"]},
        {"probe_id": "p2", "question": "What is the current cost for Y?",
         "keywords": ["$50"], "forbidden_keywords": ["$80"]},
        {"probe_id": "p3", "question": "What is the current size for Z?",
         "keywords": ["10GB"], "forbidden_keywords": ["5GB"]},
    ]
    llm = _StubLLM({
        "latency for X": "the latest measurement is 120ms",        # correct
        "cost for Y":    "it was $80, now $50",                    # stale residue
        "size for Z":    "the size is unknown",                    # no answer
    })
    scores, details = run_trend_probes(probes, "(memory)", llm)
    assert scores == [1, 0, 0]
    assert details[0]["score"] == 1
    assert "120ms" in details[0]["response"]
    assert details[1]["score"] == 0  # forbidden penalty bites
    assert details[2]["score"] == 0  # neither expected nor forbidden


def test_run_keyword_probes_scores_response_against_expected_keywords():
    from agingbench.scenarios.s1_research_literature.task_validator import (
        run_keyword_probes,
    )
    probes = [
        {"probe_id": "kp1", "question": "What was the test coverage?",
         "keywords": ["68.7%"]},
        {"probe_id": "kp2", "question": "How many vulns were found?",
         "keywords": ["12"]},
        {"probe_id": "kp3", "question": "What was the latency?",
         "keywords": ["95ms"]},
    ]
    llm = _StubLLM({
        "test coverage": "the test coverage was 68.7% per the report",
        "vulns":         "12 vulnerabilities were identified",
        "latency":       "I don't have that data.",
    })
    scores, details = run_keyword_probes(probes, "(memory)", llm)
    assert scores == [1, 1, 0]
    assert details[0]["score"] == 1
    assert details[2]["score"] == 0


def test_runner_response_path_overrides_memory_score_in_version_accuracy():
    """When response-based scoring runs, the per-probe `score` field in
    trend_probe_results reflects the LLM verdict, not the memory check.
    version_accuracy then aggregates the response verdicts directly."""
    graph = {
        "tasks": {
            "t1": {"session": 0, "dependency_type": "trend",
                   "fact_versions_required": {"f1": 2}},
        },
        "facts": {"f1": {}},
    }
    session_results = [{
        "session": 0,
        "keyword_m": 0.9,
        "trend_probe_results": [{
            "probe_id": "p1",
            "score_memory": 0.0,     # memory had stale residue
            "score_response": 1.0,   # but agent answered correctly
            "score": 1.0,            # runner writes the response score here
        }],
    }]
    lookup = {r["session"]: r for r in session_results}
    assert version_accuracy(lookup, graph["tasks"], graph["facts"]) == 1.0


def test_version_accuracy_mixed_faithful_and_proxy_paths():
    """When some sessions carry trend_probe_results and others don't, each
    session should use its appropriate path. Sanity-check aggregation."""
    graph = {
        "tasks": {
            "t1": {
                "session": 1,
                "dependency_type": "trend",
                "fact_versions_required": {"f1": 2},
            },
            "t2": {
                "session": 2,
                "dependency_type": "trend",
                "fact_versions_required": {"f2": 2},
            },
        },
        "facts": {"f1": {}, "f2": {}},
    }
    session_results = [
        # Faithful: 2 probes, 1 correct
        {
            "session": 1,
            "keyword_m": 0.9,
            "trend_probe_results": [
                {"probe_id": "a", "score": 1.0},
                {"probe_id": "b", "score": 0.0},
            ],
        },
        # Proxy: keyword_m = 0.7 → counts as 1 correct
        {"session": 2, "keyword_m": 0.7},
    ]
    lookup = _build_session_lookup_from_results(session_results)
    # 2 correct of 3 total (2 faithful probes + 1 proxy session) = 0.6667
    va = version_accuracy(lookup, graph["tasks"], graph["facts"])
    assert va == 0.6667, va
