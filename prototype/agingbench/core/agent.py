"""
agingbench/baselines/agent.py — Reference agent (PDF §9.1, §6.3 baselines/).

Depends on BaseLLM (not a concrete provider) per §6.1.1.
Uses ToolRegistry / ToolSpec (not plain dicts) per §6.1.3.
"""

import re
import json
from abc import ABC, abstractmethod
from typing import Optional
from .llm import BaseLLM
from .tools import ToolRegistry
from .memory.base import MemoryPolicy


_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)\s*", re.DOTALL | re.IGNORECASE)


def _llm_model_id(llm: BaseLLM) -> str:
    return getattr(llm, "model_id", None) or getattr(llm, "model", "") or ""


def strip_thinking(text: str, llm: BaseLLM) -> str:
    """Strip <think>…</think> blocks from assistant text if the current model's
    ModelConfig requires it (e.g. Gemma 4). No-op otherwise."""
    from .model_config import get_model_config
    if not get_model_config(_llm_model_id(llm)).strip_thinking:
        return text
    return _THINK_BLOCK.sub("", text).strip()


class AgentInterface(ABC):
    """
    Abstract interface that any pluggable agent must satisfy.

    External users implement this to test their own agent on AgingBench
    scenarios.  The runner calls ``run_session()`` once per session and
    manages memory externally (the agent never calls memory.write()).

    For stateful multi-turn agents (e.g. Claude Code, Codex) that manage
    their own workspace, see ``AgentAdapter`` in ``core/agent_adapter.py``.

    Constructor contract (runners pass these keyword arguments):
        llm            : BaseLLM instance
        memory_policy  : MemoryPolicy instance
        tools          : ToolRegistry (scenario-specific tools)
        max_turns      : int (budget for multi-turn reasoning)
    """

    @abstractmethod
    def run_session(self, task: str, session_id: int = 0) -> dict:
        """
        Execute one session and return results.

        Returns
        -------
        dict with at least:
            "output"     : str   — the agent's final answer
            "tool_calls" : list  — tool invocations made during the session
            "turns"      : int   — number of reasoning turns used
        """


REACT_SYSTEM = """You are a helpful agent that completes tasks step by step.
You have access to the following tools:
{tool_descriptions}

To use a tool, respond with:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON object with tool arguments>

When you have a final answer, respond with:
Thought: I have completed the task.
Final Answer: <your answer>

Your memory from previous sessions (may be empty):
{memory}"""


class ReferenceAgent(AgentInterface):
    """
    Minimal ReAct-style reference agent (PDF §9.1).

    Session contract (memory.md pattern):
      session input  = system_prompt + memory.read() + task
      session output = final_answer

    The runner — not the agent — calls memory.write() after each session.
    The agent is stateless between sessions.
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory_policy: MemoryPolicy,
        tools: Optional[ToolRegistry] = None,
        max_turns: int = 8,
    ):
        self.llm = llm
        self.memory_policy = memory_policy
        self.tools: ToolRegistry = tools or ToolRegistry()
        self.max_turns = max_turns

    # ---------------------------------------------------------- P2 entry point

    def compress(self, text: str) -> str:
        """Single-call compression for P2. Returns compressed text."""
        from .memory.summarize_store import COMPACT_MEDIUM
        prompt = COMPACT_MEDIUM.format(text=text)
        return self.llm.chat([{"role": "user", "content": prompt}])

    # ---------------------------------------------------- Full ReAct session

    def run_session(self, task: str, session_id: int = 0) -> dict:
        """
        Execute one session. Returns {"output": str, "tool_calls": list, "turns": int}.
        """
        system_msg = REACT_SYSTEM.format(
            tool_descriptions=self.tools.prompt_descriptions(),
            memory=self.memory_policy.read() or "(empty)",
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": task},
        ]
        tool_calls = []

        for turn in range(self.max_turns):
            response = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": strip_thinking(response, self.llm)})

            final = self._parse_final_answer(response)
            if final is not None:
                return {"output": final, "tool_calls": tool_calls, "turns": turn + 1}

            parsed = self._parse_action(response)
            if parsed:
                tool_name, tool_input = parsed
                spec = self.tools.get(tool_name)
                if spec:
                    result = spec.call(tool_input)
                    tool_calls.append({"tool": tool_name, "version": spec.version,
                                       "input": tool_input, "result": result})
                    messages.append({"role": "user", "content": f"Observation: {result}"})
                else:
                    messages.append(
                        {"role": "user", "content": f"Error: unknown tool '{tool_name}'"}
                    )
            else:
                # No action or final answer — nudge the model to continue.
                # Ensures messages end with a user turn (required by some providers).
                messages.append(
                    {"role": "user", "content": "Continue. Use the Action/Final Answer format."}
                )

        return {"output": response, "tool_calls": tool_calls, "turns": self.max_turns}

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _parse_action(text: str) -> Optional[tuple[str, dict]]:
        action = re.search(r"Action:\s*(\w+)", text)
        inp = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
        if action and inp:
            try:
                return action.group(1), json.loads(inp.group(1))
            except json.JSONDecodeError:
                return action.group(1), {"raw": inp.group(1)}
        return None

    @staticmethod
    def _parse_final_answer(text: str) -> Optional[str]:
        m = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        return m.group(1).strip() if m else None
