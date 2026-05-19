"""
Regression tests for the policy-aware write pattern in S2 and S6 runners.

Before fix: both runners used a single cumulative write path
    current_mem = policy.read(); policy.write(current_mem + "\n\n" + new)
which was correct for SummarizeStore (re-compresses against full history)
but wrong for AppendOnly. For AppendOnly it turned "append each session
as an episode" into O(N^2) cumulative snapshots.

These tests assert that:
 * AppendOnly receives *episodic* writes — each call gets only the new
   session's interaction_history, so the store has N entries each of
   bounded size after N sessions.
 * SummarizeStore still receives *cumulative* writes (old + new).
"""
from __future__ import annotations

import pytest

from agingbench.core.memory.append_only import AppendOnlyPolicy
from agingbench.core.memory.summarize_store import SummarizeStorePolicy


class _StubLLM:
    model_id = "stub"
    model = "stub"

    # SummarizeStorePolicy routes writes through llm.chat(); stub returns a
    # fixed summary so the test can detect it in the stored memory.
    def chat(self, messages):
        return "[stub summary]"

    def generate(self, *args, **kwargs):
        return "[stub summary]"

    last_thought = ""


class _StubTracer:
    def __init__(self):
        self.path = type("P", (), {"parent": __import__("pathlib").Path("/tmp")})()

    def log(self, *a, **kw):
        return "stub-span"

    def log_llm_call(self, *a, **kw):
        pass


class _SpyAppendOnly(AppendOnlyPolicy):
    """Captures each write() payload so tests can assert on them."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.write_payloads: list[str] = []

    def write(self, new_content: str, llm=None) -> None:  # type: ignore[override]
        self.write_payloads.append(new_content)
        super().write(new_content, llm=llm)


class _SpySummarizeStore(SummarizeStorePolicy):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.write_payloads: list[str] = []

    def write(self, new_content: str, llm=None) -> None:  # type: ignore[override]
        self.write_payloads.append(new_content)
        super().write(new_content, llm=llm)


def _make_s2(policy):
    from agingbench.runner.s2_runner import S2Runner

    # Minimal fake scenario: 3 sessions, 1 task each, no eval probes, no
    # session facts, no updates. Keeps the runner loop tight so the test
    # focuses on the write-path behavior.
    fake = {
        "source_profile": {"profile_text": "PROFILE"},
        "session_tasks": {
            "sessions": [
                {"session": i, "tasks": [
                    {"id": f"t{i}",
                     "text": f"Task session {i}",
                     "constraints_tested": []},
                ]}
                for i in range(3)
            ]
        },
        "constraint_updates": {"updates": []},
        "eval_probes": {"probes": []},
        "session_facts": {"facts": []},
        "compounding_probes": {"probes": []},
    }

    # Minimal agent: echoes the task text as the output, no tool calls.
    class _StubAgent:
        def __init__(self, **kw):
            self.max_turns = 8

        def run_session(self, task_text, session_id=None):
            return {"output": f"RESPONSE[{task_text[:20]}]",
                    "turns": 1, "tool_calls": []}

    return S2Runner(
        memory_policy=policy,
        llm=_StubLLM(),
        tracer=_StubTracer(),
        sut_id="test",
        generated_data=fake,
        agent_class=_StubAgent,
    )


def test_s2_appendonly_receives_episodic_writes():
    """Each write payload should be one session's interaction history, not
    a cumulative snapshot."""
    policy = _SpyAppendOnly(
        db_path=":memory:", embedding_model=None, top_k=5, max_input_tokens=100_000,
    )
    r = _make_s2(policy)
    r.run(n_sessions=3, seed=42)

    # The runner writes the profile at session 0, then one write per session.
    # So we expect 4 payloads total: [profile, sess0, sess1, sess2].
    assert len(policy.write_payloads) == 4, (
        f"expected 4 writes (profile + 3 sessions), got {len(policy.write_payloads)}"
    )
    # First write is the profile seed.
    assert "PROFILE" in policy.write_payloads[0]
    # Subsequent writes are per-session and must NOT contain prior sessions.
    for idx, payload in enumerate(policy.write_payloads[1:], start=0):
        assert f"Session {idx}" in payload or f"session {idx}" in payload
        # Must not contain prior sessions (the cumulative-write bug).
        for prior in range(idx):
            assert f"--- Session {prior} ---" not in payload, (
                f"session {idx} write contains prior session {prior} — "
                f"cumulative regression!"
            )


def test_s2_summarize_store_still_cumulative():
    """SummarizeStore must still receive 'current + new' so re-compression
    works. This test guards against over-correction of the above fix."""
    policy = _SpySummarizeStore(
        prompt_template="Summarize the following:\n{text}\nSUMMARY:",
        word_budget=200,
    )
    r = _make_s2(policy)
    r.run(n_sessions=2, seed=42)

    # SummarizeStore's write is called with "current + new" at each session
    # (plus the initial profile write at session 0). The second session's
    # write payload must reference something from session 0 (via current_mem).
    assert len(policy.write_payloads) == 3
    # First (profile), second (current=stub summary from prior + new s0), etc.
    # The second and third writes should contain the stub summary (proof of
    # concat with current_mem) AND the session text.
    assert "[stub summary]" in policy.write_payloads[1], (
        "SummarizeStore second write should include 'current_mem' "
        "(the prior summary), but payload did not contain the stub."
    )
    assert "Session 0" in policy.write_payloads[1]


def _make_s6(policy):
    from agingbench.runner.s6_runner import S6Runner

    fake = {
        "session_tasks": {
            "system_prompt": "You are a helpful assistant.",
            "sessions": [
                {
                    "session": i,
                    "domain": "general",
                    "recall_probes": [],
                    "task": {"id": f"t{i}", "text": f"Task for session {i}"},
                    "environment_data": "",
                }
                for i in range(3)
            ],
        }
    }

    class _StubAgent:
        def __init__(self, **kw):
            self.max_turns = 8

        def run_session(self, task_text, session_id=None):
            return {"output": f"RESPONSE[{task_text[:20]}]",
                    "turns": 1, "tool_calls": []}

    return S6Runner(
        memory_policy=policy,
        llm=_StubLLM(),
        tracer=_StubTracer(),
        sut_id="test",
        generated_data=fake,
        agent_class=_StubAgent,
    )


def test_s6_appendonly_receives_episodic_writes():
    policy = _SpyAppendOnly(
        db_path=":memory:", embedding_model=None, top_k=5, max_input_tokens=100_000,
    )
    r = _make_s6(policy)
    r.run(n_sessions=3, seed=42)

    # S6 does NOT do a profile seed write, so we expect exactly 3 writes —
    # one per session.
    assert len(policy.write_payloads) == 3, (
        f"expected 3 session writes, got {len(policy.write_payloads)}"
    )
    for idx, payload in enumerate(policy.write_payloads):
        # Must not contain prior session markers (cumulative regression check).
        for prior in range(idx):
            assert f"--- Session {prior} ---" not in payload, (
                f"S6 session {idx} write contains prior session {prior}"
            )
