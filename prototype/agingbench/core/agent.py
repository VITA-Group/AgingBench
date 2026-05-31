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


class _ParseFailure:
    """Sentinel returned by ``_parse_action`` when the action name parsed but
    the JSON args did not. The run loop surfaces ``reason`` as an error
    Observation so the model can re-emit valid JSON next turn.
    """
    __slots__ = ("reason",)

    def __init__(self, reason: str):
        self.reason = reason


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

        # Local cache: identical (tool, args) calls reuse the result so the
        # model is not punished with extra LLM turns for re-issuing the
        # same call. Discarded when run_session returns.
        tool_cache: dict[tuple, object] = {}

        def _tool_cache_key(name: str, args: dict) -> tuple:
            try:
                return (name, json.dumps(args, sort_keys=True, default=str))
            except Exception:
                return (name, repr(args))

        for turn in range(self.max_turns):
            response = self.llm.chat(messages)
            # Parse the visible (stripped) content, not the raw response —
            # otherwise Actions or Final Answers inside <think> blocks of a
            # thinking model (Gemma 4, DeepSeek R1) get treated as committed
            # protocol output. Visible is also what we add to history.
            visible = strip_thinking(response, self.llm)
            messages.append({"role": "assistant", "content": visible})

            final = self._parse_final_answer(visible)
            if final is not None:
                return {"output": final, "tool_calls": tool_calls, "turns": turn + 1}

            parsed = self._parse_action(visible)
            if parsed:
                tool_name, tool_input = parsed
                if isinstance(tool_input, _ParseFailure):
                    messages.append({
                        "role": "user",
                        "content": f"Observation: ERROR: {tool_input.reason}",
                    })
                    continue
                spec = self.tools.get(tool_name)
                if spec:
                    cache_key = _tool_cache_key(tool_name, tool_input)
                    if cache_key in tool_cache:
                        result = tool_cache[cache_key]
                        cached_hint = (
                            " [repeated call — same result as before; "
                            "try a different argument or commit a Final Answer]"
                        )
                        obs_text = f"Observation: {result}{cached_hint}"
                    else:
                        try:
                            result = spec.call(tool_input)
                        except Exception as e:
                            result = f"ERROR: {type(e).__name__}: {e}"
                        tool_cache[cache_key] = result
                        obs_text = f"Observation: {result}"
                    tool_calls.append({"tool": tool_name, "version": spec.version,
                                       "input": tool_input, "result": result})
                    messages.append({"role": "user", "content": obs_text})
                else:
                    messages.append(
                        {"role": "user", "content": f"Error: unknown tool '{tool_name}'"}
                    )
            else:
                # No Action and no Final Answer; ensure messages end with a
                # user turn (required by some providers) and nudge.
                messages.append(
                    {"role": "user", "content": "Continue. Use the Action/Final Answer format."}
                )

        # max_turns exhausted without a Final Answer; return the stripped
        # last response so downstream scoring doesn't see <think> blocks.
        return {"output": visible, "tool_calls": tool_calls, "turns": self.max_turns}

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _extract_balanced_json(text: str, start: int) -> Optional[str]:
        """Find a balanced ``{...}`` JSON object at or after ``start``.

        Walks the string respecting string literals and escapes so a ``}``
        inside a quoted value does not close the object early. Returns
        None if no balanced object is found.
        """
        depth = 0
        in_string = False
        escape = False
        open_idx = None
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if open_idx is None:
                    open_idx = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and open_idx is not None:
                    return text[open_idx : i + 1]
                if depth < 0:
                    return None
        return None

    @staticmethod
    def _parse_action(text: str) -> Optional[tuple[str, dict]]:
        action = re.search(r"Action:\s*([\w-]+)", text)
        if not action:
            return None
        inp_marker = re.search(r"Action Input:\s*", text)
        if inp_marker is None:
            return None
        balanced = ReferenceAgent._extract_balanced_json(text, inp_marker.end())
        if balanced is None:
            return action.group(1), _ParseFailure(
                "Action Input did not contain a parseable JSON object."
            )
        try:
            return action.group(1), json.loads(balanced)
        except json.JSONDecodeError as e:
            return action.group(1), _ParseFailure(
                f"Action Input JSON was malformed: {e}. Retry with valid JSON."
            )

    @staticmethod
    def _parse_final_answer(text: str) -> Optional[str]:
        # Stop at the next ReAct marker so commit-then-retry scaffolding
        # does not leak into the scored output.
        m = re.search(
            r"Final Answer:\s*(.*?)(?=\n\s*(?:Thought|Action|Observation)\s*:|\Z)",
            text,
            re.DOTALL,
        )
        return m.group(1).strip() if m else None
