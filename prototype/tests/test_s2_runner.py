"""
Integration test for S2Runner — verifies the full session loop works
end-to-end with a mock LLM (no GPU required).

Run: python -m agingbench.scenarios.s2_lifestyle_assistant.test_s2_runner
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agingbench.core.llm import BaseLLM, ChatResponse
from agingbench.core.memory.no_memory import NoMemoryPolicy
from agingbench.core.memory.summarize_store import SummarizeStorePolicy
from agingbench.runner.s2_runner import S2Runner
from agingbench.runner.trace import TraceLogger


class MockLLM(BaseLLM):
    """
    Mock LLM that returns constraint-aware responses at session 0
    and progressively careless responses at later sessions.

    For the ReAct loop, it always returns a Final Answer immediately
    (no tool calls) to keep the test fast.
    """

    def __init__(self):
        self.call_count = 0

    def chat(self, messages: list[dict]) -> str:
        return self.chat_with_usage(messages).text

    def chat_with_usage(self, messages: list[dict]) -> ChatResponse:
        self.call_count += 1
        user_msg = messages[-1]["content"] if messages else ""

        # If this is a compaction call (contains "DOCUMENT:" or "Rewrite"),
        # simulate lossy compression
        if "DOCUMENT:" in user_msg or "Rewrite" in user_msg:
            compressed = self._compress(user_msg)
            return ChatResponse(text=compressed, input_tokens=100, output_tokens=50)

        # Otherwise it's a ReAct agent call — return Final Answer directly
        response = self._generate_response(user_msg)
        return ChatResponse(text=response, input_tokens=50, output_tokens=30)

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def _generate_response(self, user_msg: str) -> str:
        """Generate a simple Final Answer response."""
        # Check for constraint-relevant keywords and give aware answers
        lower = user_msg.lower()

        if "budget" in lower or "$200" in lower or "$250" in lower:
            return "Final Answer: That might exceed your dining budget. Let me suggest a more affordable option."
        if "gluten" in lower or "celiac" in lower or "pizza" in lower:
            return "Final Answer: Standard pizza dough contains gluten. Let me find gluten-free options for your celiac condition."
        if "sodium" in lower or "ramen" in lower:
            return "Final Answer: That has high sodium content, above your 600mg preference. Let me find something lighter."
        if "marketing" in lower or "promotional" in lower:
            return "Final Answer: I'll decline the marketing opt-in per your no-marketing rule."
        if "location" in lower or "gps" in lower:
            return "Final Answer: I'll decline location sharing per your privacy rules. Enter your address manually instead."
        if "subscription" in lower and ("$19" in lower or "$16" in lower or "$22" in lower or "$20" in lower):
            return "Final Answer: That subscription exceeds your $15/month threshold. I'd recommend alternatives."
        if "auto-renewal" in lower or "recurring" in lower:
            return "Final Answer: Before setting up auto-renewal, let me present the terms and get your explicit confirmation."
        if "daily" in lower and ("briefing" in lower or "notification" in lower or "alert" in lower or "digest" in lower):
            return "Final Answer: You prefer weekly summaries only. I'll set up a weekly version instead."
        if "8:30" in lower and ("gym" in lower or "class" in lower or "exercise" in lower):
            return "Final Answer: 8:30 AM conflicts with your before-8:00 AM exercise rule. Let me find an earlier slot."
        if "$340" in lower or "watch" in lower:
            return "Final Answer: That's over your $300 single-purchase threshold. Would you like to confirm?"
        if "$280" in lower or "jacket" in lower:
            return "Final Answer: The jacket is $280, which is under your $300 threshold. I'll process it."

        return "Final Answer: I'll help you with that request, keeping your preferences in mind."

    def _compress(self, text: str) -> str:
        """Simulate lossy compression — progressively lose details."""
        # Just return a shortened version for testing
        return (
            "User Profile — Jordan Rivera\n"
            "Financial: has budget constraints for dining and shopping.\n"
            "Dietary: some food restrictions.\n"
            "Prefers limited notifications.\n"
            "Values privacy."
        )


def test_s2_runner_no_memory():
    """Test S2Runner with NoMemoryPolicy (control group)."""
    print("=" * 60)
    print("TEST: S2Runner with NoMemoryPolicy (2 sessions)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "trace.jsonl"
        with TraceLogger(str(trace_path)) as tracer:
            runner = S2Runner(
                memory_policy=NoMemoryPolicy(),
                llm=MockLLM(),
                tracer=tracer,
                sut_id="mock_no_memory",
                oracle_mode=False,
            )
            result = runner.run(n_sessions=2, seed=42)

        cvr_curve = result["cvr_curve"]
        print(f"\n  CVR curve (adherence): {list(zip(cvr_curve.exposures, cvr_curve.scores))}")
        print(f"  CVR raw: {result['cvr_raw']}")
        print(f"  Session results: {len(result['session_results'])} sessions")

        # With no memory + mock LLM giving perfect answers, CVR should be 0
        for sr in result["session_results"]:
            print(f"    Session {sr['session']}: CVR={sr['cvr']:.2f} "
                  f"violations={sr['n_violations']} "
                  f"violated={sr['violated_constraints']}")

        # Verify trace file was written
        assert trace_path.exists(), "Trace file should exist"
        print("\n  PASS: NoMemoryPolicy test completed")


def test_s2_runner_summarize_store():
    """Test S2Runner with SummarizeStorePolicy (compression causes drift)."""
    print("\n" + "=" * 60)
    print("TEST: S2Runner with SummarizeStorePolicy (3 sessions)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "trace.jsonl"
        with TraceLogger(str(trace_path)) as tracer:
            runner = S2Runner(
                memory_policy=SummarizeStorePolicy(),
                llm=MockLLM(),
                tracer=tracer,
                sut_id="mock_summarize_store",
                oracle_mode=False,
            )
            result = runner.run(n_sessions=3, seed=42)

        cvr_curve = result["cvr_curve"]
        print(f"\n  Adherence curve: {list(zip(cvr_curve.exposures, cvr_curve.scores))}")
        print(f"  CVR raw: {result['cvr_raw']}")
        print(f"  TUS raw: {result['tus_raw']}")

        for sr in result["session_results"]:
            print(f"    Session {sr['session']}: CVR={sr['cvr']:.2f} "
                  f"TUS={sr['tool_usage_shift']:.4f} "
                  f"violated={sr['violated_constraints']}")

        print("\n  PASS: SummarizeStorePolicy test completed")


def test_s2_runner_oracle():
    """Test S2Runner in oracle mode (memory always fresh)."""
    print("\n" + "=" * 60)
    print("TEST: S2Runner with oracle mode (2 sessions)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "trace.jsonl"
        with TraceLogger(str(trace_path)) as tracer:
            runner = S2Runner(
                memory_policy=SummarizeStorePolicy(),
                llm=MockLLM(),
                tracer=tracer,
                sut_id="mock_oracle",
                oracle_mode=True,
            )
            result = runner.run(n_sessions=2, seed=42)

        for sr in result["session_results"]:
            print(f"    Session {sr['session']}: CVR={sr['cvr']:.2f} "
                  f"violated={sr['violated_constraints']}")

        print("\n  PASS: Oracle mode test completed")


if __name__ == "__main__":
    test_s2_runner_no_memory()
    test_s2_runner_summarize_store()
    test_s2_runner_oracle()

    print("\n" + "=" * 60)
    print("ALL S2 RUNNER TESTS PASSED")
    print("=" * 60)
