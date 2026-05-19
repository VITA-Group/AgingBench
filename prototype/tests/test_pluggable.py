"""
tests/test_pluggable.py — Verify that external users can plug in custom agents
and memory policies via CLI flags and SUT YAML config.

Tests:
  1. AgentInterface ABC can be subclassed
  2. Custom agent can be loaded via importlib (module:Class)
  3. Custom memory policy can be loaded via build_memory_policy(type="custom")
  4. Runners accept agent_class parameter
  5. CLI --agent flag resolves correctly
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import Optional

# ---- 1. Custom agent that satisfies AgentInterface ----

from agingbench.core.agent import AgentInterface, ReferenceAgent
from agingbench.core.tools import ToolRegistry
from agingbench.core.memory.base import MemoryPolicy, build_memory_policy


class EchoAgent(AgentInterface):
    """Minimal test agent: echoes the task + memory back as output."""

    def __init__(self, llm, memory_policy, tools=None, max_turns=8):
        self.llm = llm
        self.memory_policy = memory_policy
        self.tools = tools or ToolRegistry()
        self.max_turns = max_turns

    def run_session(self, task: str, session_id: int = 0) -> dict:
        memory_content = self.memory_policy.read() or "(empty)"
        output = f"ECHO: task={task[:100]} memory={memory_content[:200]}"
        return {"output": output, "tool_calls": [], "turns": 1}


# ---- 2. Custom memory policy ----

class FixedMemoryPolicy(MemoryPolicy):
    """Test memory policy: always returns a fixed string."""

    def __init__(self, fixed_text: str = "FIXED_MEMORY_CONTENT"):
        self._text = fixed_text

    def read(self, query: Optional[str] = None) -> str:
        return self._text

    def write(self, new_content: str, llm=None) -> None:
        pass  # no-op

    def reset(self) -> None:
        self._text = ""


# ---- Tests ----

def test_agent_interface_subclass():
    """EchoAgent satisfies AgentInterface."""
    assert issubclass(EchoAgent, AgentInterface)
    assert issubclass(ReferenceAgent, AgentInterface)


def test_custom_agent_instantiation():
    """EchoAgent can be instantiated with the same args as ReferenceAgent."""
    from agingbench.core.memory.no_memory import NoMemoryPolicy

    agent = EchoAgent(
        llm=None,
        memory_policy=NoMemoryPolicy(),
        tools=ToolRegistry(),
        max_turns=4,
    )
    result = agent.run_session("What is 2+2?")
    assert "output" in result
    assert "ECHO:" in result["output"]
    assert result["turns"] == 1
    print("  [PASS] Custom agent instantiation and run_session")


def test_custom_memory_policy_via_factory():
    """build_memory_policy can load a custom class via type='custom'."""
    policy = build_memory_policy({
        "type": "custom",
        "class": "tests.test_pluggable:FixedMemoryPolicy",
        "fixed_text": "hello from custom policy",
    })
    # Check by class name (avoids __main__ vs module identity mismatch)
    assert type(policy).__name__ == "FixedMemoryPolicy"
    assert isinstance(policy, MemoryPolicy)
    assert policy.read() == "hello from custom policy"
    print("  [PASS] Custom memory policy via factory")


def test_agent_loader():
    """_load_agent_class resolves module:Class correctly."""
    from agingbench.cli import _load_agent_class

    # Default (None) → ReferenceAgent
    cls = _load_agent_class(None)
    assert cls is ReferenceAgent
    print("  [PASS] Default agent loader → ReferenceAgent")

    # Custom spec
    cls = _load_agent_class("test_pluggable:EchoAgent")
    assert cls.__name__ == "EchoAgent"
    assert issubclass(cls, AgentInterface)
    print("  [PASS] Custom agent loader → EchoAgent")


def test_runner_accepts_agent_class():
    """S6Runner accepts agent_class parameter."""
    from agingbench.runner.s6_runner import S6Runner
    from agingbench.core.memory.no_memory import NoMemoryPolicy

    # Just verify constructor doesn't crash — no actual run
    class FakeLLM:
        model_id = "test"
        def chat(self, messages): return "Final Answer: test"
        def chat_with_usage(self, messages):
            from agingbench.core.llm import ChatResponse
            return ChatResponse(text="Final Answer: test", input_tokens=0, output_tokens=0)
        def count_tokens(self, text): return len(text.split())

    from agingbench.runner.trace import TraceLogger

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        trace_path = f.name

    with TraceLogger(trace_path) as tracer:
        runner = S6Runner(
            memory_policy=NoMemoryPolicy(),
            llm=FakeLLM(),
            tracer=tracer,
            sut_id="test",
            agent_class=EchoAgent,
        )
        assert runner.agent_class is EchoAgent
        print("  [PASS] S6Runner accepts agent_class=EchoAgent")


def test_custom_memory_in_sut_yaml():
    """SUT YAML with type=custom can load a custom memory policy."""
    import yaml

    sut_yaml = {
        "sut_id": "test_custom_memory",
        "model": {"provider": "local_hf", "model_id": "test"},
        "memory_policy": {
            "type": "custom",
            "class": "tests.test_pluggable:FixedMemoryPolicy",
            "fixed_text": "yaml-configured memory",
        },
    }

    policy = build_memory_policy(sut_yaml["memory_policy"])
    assert type(policy).__name__ == "FixedMemoryPolicy"
    assert policy.read() == "yaml-configured memory"
    print("  [PASS] SUT YAML with custom memory policy")


if __name__ == "__main__":
    print("\n=== AgingBench Pluggability Tests ===\n")
    test_agent_interface_subclass()
    print("  [PASS] AgentInterface subclass check")
    test_custom_agent_instantiation()
    test_custom_memory_policy_via_factory()
    test_agent_loader()
    test_runner_accepts_agent_class()
    test_custom_memory_in_sut_yaml()
    print("\n=== All pluggability tests passed ===\n")


