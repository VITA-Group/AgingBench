"""
Unit-level validation that C1/C2/C3/C4 produce distinct memory_text content
in S2Runner. Does not run the LLM — exercises only the memory-text builders
and state accumulators.

Run:
    cd prototype && uv run pytest tests/test_attribution_modes.py -v
"""

import pytest

from agingbench.core.memory.append_only import AppendOnlyPolicy
from agingbench.core.memory.no_memory import NoMemoryPolicy
from agingbench.runner.s2_runner import S2Runner


class _StubLLM:
    """Minimal stub so S2Runner.__init__ can compute model_id/_provider."""
    model_id = "stub"
    model = "stub"


class _StubTracer:
    def log(self, *a, **kw): pass
    def log_llm_call(self, *a, **kw): pass


def _make_runner(**attrib_flags):
    """Build an S2Runner with minimal fake generated_data and the given flags."""
    fake_data = {
        "source_profile": {"profile_text": "PROFILE: Dr. Q\n- [C1] do X\n- [C2] never Y"},
        "session_tasks": {"sessions": []},
        "constraint_updates": {"updates": []},
        "eval_probes": {"probes": []},
        "session_facts": {"facts": [
            {"session": 0, "content": "fact-0 content", "keywords": ["f0"]},
            {"session": 1, "content": "fact-1 content", "keywords": ["f1"]},
            {"session": 2, "content": "fact-2 content", "keywords": ["f2"]},
        ]},
        "compounding_probes": {"probes": []},
    }
    return S2Runner(
        memory_policy=NoMemoryPolicy(),  # irrelevant for C2/C3/C4; used by C1
        llm=_StubLLM(),
        tracer=_StubTracer(),
        sut_id="test_sut",
        generated_data=fake_data,
        **attrib_flags,
    )


def test_c2_oracle_retrieval_injects_gold_facts():
    r = _make_runner(oracle_retrieval=True)
    # At session 2, the gold text should contain session 0 and 1 facts.
    text = r._build_gold_text(up_to_session=2)
    assert "GOLD REFERENCE (oracle retrieval)" in text
    assert "fact-0 content" in text
    assert "fact-1 content" in text
    # Should NOT leak session 2 (the current session, before it has happened)
    assert "fact-2 content" not in text


def test_c3_oracle_store_uses_runner_owned_appendonly():
    r = _make_runner(oracle_store=True)
    # Seed with two raw session outputs.
    r._write_c3("SESSION 0 RAW: value = 42")
    r._write_c3("SESSION 1 RAW: budget = $100")
    text = r._read_c3(query=None)
    assert text.startswith("=== ORACLE STORE (C3, raw-stored + top-k cosine) ===")
    # Either cosine retrieval returns something or fallback returns last-k.
    assert "SESSION 1 RAW" in text or "SESSION 0 RAW" in text
    # The underlying store must be an AppendOnly, not the SUT's NoMemoryPolicy.
    assert isinstance(r._c3_store, AppendOnlyPolicy)


def test_c4_incontext_ceiling_concatenates_full_history():
    r = _make_runner(incontext_ceiling=True)
    r._append_c4("SESSION 0 RAW: alpha")
    r._append_c4("SESSION 1 RAW: beta")
    r._append_c4("SESSION 2 RAW: gamma")
    text = r._read_c4()
    assert text.startswith("=== IN-CONTEXT CEILING (C4)")
    # All three sessions must be visible (ceiling is far above content size).
    assert "alpha" in text and "beta" in text and "gamma" in text
    # Profile should also be present so the agent has its baseline.
    assert "PROFILE: Dr. Q" in text


def test_c4_truncation_drops_oldest_when_over_budget():
    r = _make_runner(incontext_ceiling=True, ceiling_max_tokens=50)  # ~200 chars
    r._append_c4("SESSION 0: " + "X" * 300)
    r._append_c4("SESSION 1: " + "Y" * 300)
    r._append_c4("SESSION 2: " + "Z" * 300)
    text = r._read_c4()
    # Truncation marker present
    assert "head-truncated" in text or "truncated" in text
    # Tail (most recent) content survives; oldest is dropped
    assert "Z" in text  # session 2
    # Session 0's large X block should be mostly dropped
    assert text.count("X" * 100) == 0


def test_four_modes_produce_materially_distinct_prefixes():
    """Each mode's memory_text must begin with a distinguishable marker."""
    r_c2 = _make_runner(oracle_retrieval=True)
    r_c3 = _make_runner(oracle_store=True)
    r_c4 = _make_runner(incontext_ceiling=True)

    prefix_c2 = r_c2._build_gold_text(up_to_session=1)[:60]
    r_c3._write_c3("sess-0-raw")
    prefix_c3 = r_c3._read_c3()[:60]
    r_c4._append_c4("sess-0-raw")
    prefix_c4 = r_c4._read_c4()[:60]

    assert "GOLD REFERENCE" in prefix_c2
    assert "ORACLE STORE" in prefix_c3
    assert "IN-CONTEXT CEILING" in prefix_c4
    # Pairwise distinct
    assert prefix_c2 != prefix_c3 != prefix_c4 != prefix_c2


def test_oracle_mode_legacy_alias_maps_to_oracle_store():
    """Back-compat: legacy oracle_mode=True should alias to C3 (oracle_store)."""
    r = _make_runner(oracle_mode=True)
    assert r.oracle_store is True, "oracle_mode should alias to oracle_store"


def test_mutual_exclusion_is_not_enforced_at_runner_level():
    """CLI enforces mutual exclusion; runner accepts multi-flag state silently.

    This test documents current behavior; enforcement lives in the CLI.
    """
    r = _make_runner(oracle_retrieval=True, oracle_store=True)
    assert r.oracle_retrieval is True
    assert r.oracle_store is True


# ---------------------------------------------------------------------------
# Milestone 2: parity tests for S1, S4, S6 runners.
# ---------------------------------------------------------------------------

def test_s4_runner_exposes_c3_c4_helpers_and_state():
    from agingbench.runner.s4_runner import S4Runner
    from agingbench.core.memory.no_memory import NoMemoryPolicy
    from agingbench.core.memory.append_only import AppendOnlyPolicy

    fake = {"tasks": {"sessions": [], "life_event": {}}, "snapshots": {"snapshots": []}}
    r = S4Runner(
        memory_policy=NoMemoryPolicy(), llm=_StubLLM(), tracer=_StubTracer(),
        generated_data=fake,
        oracle_store=True, ceiling_max_tokens=50_000,
    )
    r._write_c3("sprint-0-raw")
    assert isinstance(r._c3_store, AppendOnlyPolicy)
    out = r._read_c3()
    assert out.startswith("=== ORACLE STORE (C3, raw-stored")
    r._append_c4("sprint-0-raw")
    out4 = r._read_c4()
    assert out4.startswith("=== IN-CONTEXT CEILING (C4)")


def test_s1_runner_accepts_new_flags_and_aliases_legacy_oracle_mode():
    from agingbench.runner.s1_runner import S1Runner
    from agingbench.core.memory.no_memory import NoMemoryPolicy

    r_new = S1Runner(
        source_doc_text="doc", probes=[], validator_fn=lambda *a, **k: ([], 0),
        memory_policy=NoMemoryPolicy(), llm=_StubLLM(), tracer=_StubTracer(),
        oracle_store=True, incontext_ceiling=False,
    )
    assert r_new.oracle_store is True
    assert r_new.incontext_ceiling is False
    # Legacy oracle_mode aliases to oracle_store
    r_legacy = S1Runner(
        source_doc_text="doc", probes=[], validator_fn=lambda *a, **k: ([], 0),
        memory_policy=NoMemoryPolicy(), llm=_StubLLM(), tracer=_StubTracer(),
        oracle_mode=True,
    )
    assert r_legacy.oracle_store is True


def _make_s1_runner(**attrib_flags):
    """Build a minimal S1Runner with paper_batches so all_content_so_far has content."""
    from agingbench.runner.s1_runner import S1Runner
    fake_data = {
        "paper_batches": {
            "batches": [
                {"title": f"Paper {i}",
                 "content": f"Batch {i} body " + ("alpha_kw " * 20),
                 "keywords": [f"kw{i}"]}
                for i in range(3)
            ],
            "cross_cycle_queries": [],
        },
        "session_facts": {"facts": []},
    }
    return S1Runner(
        source_doc_text="doc",
        probes=[],
        validator_fn=lambda *a, **k: ([], 0.0),
        memory_policy=NoMemoryPolicy(),
        llm=_StubLLM(),
        tracer=_StubTracer(),
        sut_id="test_sut",
        generated_data=fake_data,
        **attrib_flags,
    )


def test_s1_c4_truncates_eval_text_when_ceiling_bites():
    """S1 C_4 must tail-truncate eval_text to ceiling_max_tokens * 4 chars."""
    # Three batches, each ~200 chars → all_content_so_far ~600 chars by cycle 2.
    # Set ceiling so the budget bites at the last cycle.
    r = _make_s1_runner(incontext_ceiling=True, ceiling_max_tokens=30)  # 120 chars
    out = r.run(n_cycles=2, seed=42)
    final_len = out["session_results"][-1]["eval_text_len"]
    assert final_len <= 30 * 4, (
        f"C_4 should truncate to <= {30*4} chars at budget, got {final_len}"
    )
    assert out["attribution_mode"] == "c4_incontext_ceiling"
    assert out["ceiling_max_tokens"] == 30


def test_s1_c4_passthrough_when_corpus_fits():
    """S1 C_4 must NOT truncate when ceiling_max_tokens is generous."""
    r = _make_s1_runner(incontext_ceiling=True, ceiling_max_tokens=100_000)
    out = r.run(n_cycles=2, seed=42)
    # Compare against an unconstrained C_3 on the same data: lengths must match.
    r_c3 = _make_s1_runner(oracle_store=True)
    out_c3 = r_c3.run(n_cycles=2, seed=42)
    for s_c4, s_c3 in zip(out["session_results"], out_c3["session_results"]):
        assert s_c4["eval_text_len"] == s_c3["eval_text_len"]


def test_s1_c2_oracle_retrieval_sets_abstain_flag():
    """S1 C_2 is aliased to C_3; runs must emit c2_abstain_s1=True."""
    r = _make_s1_runner(oracle_retrieval=True)
    out = r.run(n_cycles=1, seed=42)
    assert out["c2_abstain_s1"] is True
    # attribution_mode is kept stable across runners for downstream filtering.
    assert out["attribution_mode"] == "c2_oracle_retrieval"


def test_s1_c2_and_c3_produce_identical_eval_text_by_design():
    """Document the abstain: C_2 and C_3 share the same scoring surface in S1."""
    r_c2 = _make_s1_runner(oracle_retrieval=True)
    r_c3 = _make_s1_runner(oracle_store=True)
    out_c2 = r_c2.run(n_cycles=2, seed=42)
    out_c3 = r_c3.run(n_cycles=2, seed=42)
    for s_c2, s_c3 in zip(out_c2["session_results"], out_c3["session_results"]):
        assert s_c2["eval_text_len"] == s_c3["eval_text_len"]
        assert s_c2["keyword_m"] == s_c3["keyword_m"]
    # The abstain flag is the signal that distinguishes them for downstream.
    assert out_c2["c2_abstain_s1"] is True
    assert out_c3["c2_abstain_s1"] is False


def test_s1_non_s1_runs_do_not_emit_c2_abstain_true():
    """Non-C_2 S1 runs must report c2_abstain_s1=False."""
    for flag in (
        {},
        {"oracle_store": True},
        {"incontext_ceiling": True, "ceiling_max_tokens": 50_000},
    ):
        out = _make_s1_runner(**flag).run(n_cycles=1, seed=42)
        assert out["c2_abstain_s1"] is False, f"flag={flag}"


def test_all_runners_stamp_attribution_schema_v2_clean():
    """Every runner's returned dict must carry attribution_schema=v2_clean."""
    import inspect
    from agingbench.runner.s1_runner import S1Runner
    from agingbench.runner.s2_runner import S2Runner
    from agingbench.runner.s4_runner import S4Runner
    from agingbench.runner.s6_runner import S6Runner

    # Sanity: each runner's run() body contains the v2_clean stamp.
    for cls in (S1Runner, S2Runner, S4Runner, S6Runner):
        src = inspect.getsource(cls.run)
        assert '"attribution_schema": "v2_clean"' in src, (
            f"{cls.__name__}.run() missing attribution_schema stamp"
        )
        assert '"attribution_mode"' in src, (
            f"{cls.__name__}.run() missing attribution_mode stamp"
        )
