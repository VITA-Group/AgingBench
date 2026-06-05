"""
agingbench/core/full_react_agent.py — Full-capacity ReAct agent for AgingBench.

Unified Tier-1 agent for the recall scenarios (S1/S2/S4/S6). Unlike
``ReferenceAgent`` (which injects ``memory_policy.read()`` into the system
prompt), memory here is reachable ONLY through tools — scenarios register a
``search_memory``-style tool (see ``build_search_memory_tool``); the agent
registers none itself. This forces the ReAct loop to actually retrieve, so the
benchmark measures memory *use*, not context reading.

Behaviors:
- ``<think>`` reasoning traces are preserved and replayed across turns, so
  multi-turn reasoning works for thinking models.
- Per-turn telemetry (``turn_stats``, ``reasoning_content``) is returned, so a
  dead loop is visible without grepping the trace.
- Three termination signals (Final Answer: / <answer> / Answer:) plus a forced
  final on budget exhaustion (tagged ``exhausted=True`` so scoring can tell
  budget-exhaustion from a confidently-wrong commit).
- Tool-call caching: identical ``(tool, args)`` calls reuse the cached result.
- Optional native OpenAI tool-calling path (``native_tools=True``) for
  tool-calling-native models (e.g. gpt-oss) where text-ReAct never terminates.
- Stateless across sessions; the runner owns memory writes.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from .agent import AgentInterface, strip_thinking, _THINK_BLOCK, _ParseFailure
from .llm import BaseLLM, ChatResponse
from .memory.base import MemoryPolicy
from .tools import ToolRegistry


# System prompt — note there is NO {memory} placeholder; memory access is via
# tools only (scenarios register a search_memory tool in the ToolRegistry).

REACT_SYSTEM = """You are an agent that completes tasks step by step.

You have access to the following tools:
{tool_descriptions}

To use a tool, respond with:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON object with tool arguments>

When you have a final answer, respond with:
Thought: I have completed the task.
Final Answer: <your answer>

Notes on memory:
- Your memory of prior interactions is accessible ONLY via the tools above
  (typically `search_memory(query)` or scenario-specific equivalents).
- Memory is NOT preloaded into this prompt. Call a tool when you need
  information from previous sessions.
- If a tool returns nothing useful, try a different query phrasing before
  committing to a Final Answer."""


# Termination signals, in priority order (first match wins). Small models often
# emit Answer:/<answer> instead of the instructed Final Answer:; accept all three.

_FINAL_ANSWER_PATTERNS = [
    re.compile(
        r"Final Answer:\s*(.*?)(?=\n\s*(?:Thought|Action|Observation)\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE),
    re.compile(
        r"(?:^|\n)\s*Answer:\s*(.*?)(?=\n\s*(?:Thought|Action|Observation)\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
]


def _parse_final_answer(text: str) -> Optional[str]:
    """Return the agent's committed answer if any termination pattern matches."""
    for pat in _FINAL_ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            answer = m.group(1).strip()
            if answer:
                return answer
    return None


# Action parsing. The balanced-JSON walker lets Action Input values contain '}'
# inside quoted strings without prematurely closing the object.

def _extract_balanced_json(text: str, start: int) -> Optional[str]:
    depth = 0
    in_string = False
    escape = False
    open_idx: Optional[int] = None
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


def _parse_action(text: str) -> Optional[tuple[str, Any]]:
    action = re.search(r"Action:\s*([\w-]+)", text)
    if not action:
        return None
    inp_marker = re.search(r"Action Input:\s*", text)
    if inp_marker is None:
        return None
    balanced = _extract_balanced_json(text, inp_marker.end())
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


# ---------------------------------------------------------------------------
# Reasoning content preservation (for reasoning-trace models)
# ---------------------------------------------------------------------------

def _split_think(text: str) -> tuple[str, str]:
    """Split a model response into (reasoning, visible_content).

    For reasoning-trace models (Gemma-4, DeepSeek-R1, Sonnet-thinking) the
    response embeds the chain-of-thought inside <think>...</think>. We extract
    it so it can be replayed on subsequent turns rather than discarded.
    Returns ("", text) if no <think> block is present.
    """
    m = _THINK_BLOCK.search(text)
    if not m:
        return "", text.strip()
    reasoning = m.group(0)
    # Strip the closing tag/whitespace prefix; keep the inner content only.
    inner = re.sub(r"</?think>", "", reasoning, flags=re.IGNORECASE).strip()
    visible = _THINK_BLOCK.sub("", text).strip()
    return inner, visible


def _format_assistant_with_reasoning(reasoning: str, visible: str) -> str:
    """Re-format an assistant message with its reasoning trace prepended.

    When the model produced <think>...</think>{visible}, this reconstructs the
    same shape for replay on the next turn. The model sees its own prior
    chain-of-thought and can build on it rather than re-deriving from scratch.
    """
    if not reasoning:
        return visible
    return f"<think>{reasoning}</think>\n{visible}".strip()


# ---------------------------------------------------------------------------
# Token / latency telemetry helpers
# ---------------------------------------------------------------------------

def _safe_count_tokens(llm: BaseLLM, text: str) -> int:
    """Best-effort token count. Returns 0 if the LLM can't count."""
    try:
        return int(llm.count_tokens(text))
    except Exception:
        # Fallback: ~4 chars per token. Crude but stable.
        return max(1, len(text) // 4)


def _messages_to_input_tokens(llm: BaseLLM, messages: list[dict]) -> int:
    """Approximate input-token count of a message list."""
    joined = "\n".join((m.get("content") or "") for m in messages)
    return _safe_count_tokens(llm, joined)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class FullReactAgent(AgentInterface):
    """Full-capacity ReAct agent for AgingBench's Tier-1 unified framework.

    The session contract is unchanged from AgentInterface: one ``run_session``
    call per task or probe, returns ``{output, tool_calls, turns, ...}``.
    The runner is responsible for memory writes between sessions; this agent
    is stateless across sessions.

    Constructor parameters
    ----------------------
    llm           : BaseLLM
    memory_policy : MemoryPolicy
        Held for completeness (scenarios may pass it to tools that need it),
        but NOT consulted directly by the agent. Memory access happens only
        through the tools in ``tools``.
    tools         : ToolRegistry
        Scenario-provided tools. Should include a memory-access tool (e.g.
        ``search_memory``) — without it, the agent has no way to retrieve
        prior-session content.
    max_turns     : int, default 10
        Per-session turn budget. The forced-final-answer fallback fires when
        this is exhausted.
    use_chat_with_usage : bool, default True
        If True, calls ``llm.chat_with_usage`` to capture per-turn token
        counts. Falls back to ``llm.chat`` + estimation if the method is not
        implemented for the given backend.
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory_policy: MemoryPolicy,
        tools: Optional[ToolRegistry] = None,
        max_turns: int = 10,
        use_chat_with_usage: bool = True,
        native_tools: bool = False,
    ):
        self.llm = llm
        self.memory_policy = memory_policy
        self.tools: ToolRegistry = tools or ToolRegistry()
        self.max_turns = max_turns
        self.use_chat_with_usage = use_chat_with_usage
        # When True (and the backend implements chat_with_tools), drive the loop
        # via native OpenAI tool-calling instead of text-parsed ReAct.
        self.native_tools = native_tools
        # Optional override for the system-prompt template (must still contain
        # ``{tool_descriptions}``). Default ``None`` keeps the legacy
        # module-level REACT_SYSTEM behavior bit-for-bit. Scenario runners
        # that want to inject a per-scenario persona (e.g. S6's
        # research-analyst opener loaded from session_tasks.json) set this
        # after construction.
        self.system_template: Optional[str] = None

    # ---------------------------------------------------------------- private

    def _chat_round(
        self,
        messages: list[dict],
        round_idx: int,
    ) -> tuple[str, str, dict]:
        """One LLM call. Returns (visible_content, reasoning, turn_stats_entry)."""
        t0 = time.time()

        # Prefer chat_with_usage when available; fall back to chat() with
        # estimated counts. Scenario YAMLs may select non-LiteLLM backends
        # that haven't implemented chat_with_usage cleanly.
        response_text: str
        input_tokens: int
        output_tokens: int
        adapter_thought: str = ""
        if self.use_chat_with_usage and hasattr(self.llm, "chat_with_usage"):
            try:
                resp: ChatResponse = self.llm.chat_with_usage(messages)
                response_text = resp.text
                input_tokens = int(resp.input_tokens or 0)
                output_tokens = int(resp.output_tokens or 0)
                adapter_thought = (resp.thought or "").strip()
            except (NotImplementedError, AttributeError):
                response_text = self.llm.chat(messages)
                input_tokens = _messages_to_input_tokens(self.llm, messages)
                output_tokens = _safe_count_tokens(self.llm, response_text)
        else:
            response_text = self.llm.chat(messages)
            input_tokens = _messages_to_input_tokens(self.llm, messages)
            output_tokens = _safe_count_tokens(self.llm, response_text)

        latency = round(time.time() - t0, 3)
        # Reasoning comes either pre-split from the adapter (e.g. VLLMAdapter
        # populates ChatResponse.thought) or inline as a <think> block we
        # extract. Prefer the adapter's; fall back to inline parsing.
        if adapter_thought:
            reasoning, visible = adapter_thought, response_text
        else:
            reasoning, visible = _split_think(response_text)
        # Strip any residual <think> marks so the parser sees only visible text.
        visible = strip_thinking(visible, self.llm)

        stats = {
            "round": round_idx,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": latency,
            "had_reasoning": bool(reasoning),
        }
        return visible, reasoning, stats

    @staticmethod
    def _tool_cache_key(name: str, args: dict) -> tuple:
        try:
            return (name, json.dumps(args, sort_keys=True, default=str))
        except Exception:
            return (name, repr(args))

    # ---------------------------------------------------------------- public

    def run_session(self, task: str, session_id: int = 0) -> dict:
        """Execute one session of the ReAct loop and return structured results.

        Returns
        -------
        dict with:
            output             : str   — committed final answer
            tool_calls         : list  — one dict per tool invocation
            turns              : int   — number of LLM rounds used (excl. forced final)
            turn_stats         : list  — per-round telemetry dicts
            reasoning_content  : list[str] — non-empty <think> blocks, in order
            exhausted          : bool  — True if max_turns was reached
        """
        if self.native_tools and hasattr(self.llm, "chat_with_tools"):
            return self._run_session_native(task, session_id)
        _template = self.system_template or REACT_SYSTEM
        system_msg = _template.format(
            tool_descriptions=self.tools.prompt_descriptions(),
        )
        messages: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": task},
        ]

        tool_calls: list[dict] = []
        turn_stats: list[dict] = []
        reasoning_log: list[str] = []
        tool_cache: dict[tuple, Any] = {}

        for turn in range(self.max_turns):
            visible, reasoning, stats = self._chat_round(messages, turn + 1)
            if reasoning:
                reasoning_log.append(reasoning)

            # Replay reasoning on subsequent turns by storing the full
            # <think>{reasoning}</think>\n{visible} shape in history.
            assistant_content = _format_assistant_with_reasoning(reasoning, visible)
            messages.append({"role": "assistant", "content": assistant_content})

            final = _parse_final_answer(visible)
            if final is not None:
                stats["had_tool_call"] = False
                turn_stats.append(stats)
                return {
                    "output": final,
                    "tool_calls": tool_calls,
                    "turns": turn + 1,
                    "turn_stats": turn_stats,
                    "reasoning_content": reasoning_log,
                    "exhausted": False,
                }

            parsed = _parse_action(visible)
            if parsed:
                tool_name, tool_input = parsed
                if isinstance(tool_input, _ParseFailure):
                    messages.append({
                        "role": "user",
                        "content": f"Observation: ERROR: {tool_input.reason}",
                    })
                    stats["had_tool_call"] = False
                    turn_stats.append(stats)
                    continue

                spec = self.tools.get(tool_name)
                if spec:
                    cache_key = self._tool_cache_key(tool_name, tool_input)
                    if cache_key in tool_cache:
                        result = tool_cache[cache_key]
                        obs_text = (
                            f"Observation: {result} "
                            f"[repeated call — same result as before; try a "
                            f"different argument or commit a Final Answer]"
                        )
                    else:
                        try:
                            result = spec.call(tool_input)
                        except Exception as e:
                            result = f"ERROR: {type(e).__name__}: {e}"
                        tool_cache[cache_key] = result
                        obs_text = f"Observation: {result}"

                    tool_calls.append({
                        "tool": tool_name,
                        "version": spec.version,
                        "input": tool_input,
                        "result": result,
                    })
                    messages.append({"role": "user", "content": obs_text})
                    stats["had_tool_call"] = True
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Error: unknown tool '{tool_name}'",
                    })
                    stats["had_tool_call"] = False
            else:
                # No Action, no Final Answer. Nudge the model back into protocol.
                messages.append({
                    "role": "user",
                    "content": "Continue. Use the Action/Final Answer format.",
                })
                stats["had_tool_call"] = False

            turn_stats.append(stats)

        # ---------------------------- max_turns exhausted: force a final answer
        # Same fallback semantics as ReferenceAgent: one final forcing prompt,
        # then mark exhausted=True so scoring can distinguish budget exhaustion
        # from a confidently-wrong commit.
        messages.append({
            "role": "user",
            "content": (
                "You have used all your reasoning turns. Based on what you have "
                "already observed, you MUST now commit to a single best answer. "
                "Reply with exactly one line in the format:\n"
                "Final Answer: <your answer>\n"
                "Do not emit any more Thoughts or Actions."
            ),
        })

        forced_visible, forced_reasoning, forced_stats = self._chat_round(
            messages, self.max_turns + 1,
        )
        if forced_reasoning:
            reasoning_log.append(forced_reasoning)
        forced_stats["had_tool_call"] = False
        forced_stats["forced"] = True
        turn_stats.append(forced_stats)

        final = _parse_final_answer(forced_visible)
        return {
            "output": final if final is not None else forced_visible,
            "tool_calls": tool_calls,
            "turns": self.max_turns,
            "turn_stats": turn_stats,
            "reasoning_content": reasoning_log,
            "exhausted": True,
        }

    # ------------------------------------------------ native tool-calling path

    def _tools_openai_schema(self) -> list[dict]:
        """Convert the ToolRegistry into OpenAI function-tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters or {"type": "object", "properties": {}},
                },
            }
            for spec in self.tools  # ToolRegistry.__iter__ yields ToolSpecs
        ]

    def _run_session_native(self, task: str, session_id: int = 0) -> dict:
        """ReAct loop via native OpenAI tool-calling (for tool-calling-native
        models e.g. gpt-oss). Tool calls arrive structured; a turn with no tool
        call is the final answer. Same return contract as run_session."""
        tool_schemas = self._tools_openai_schema()
        system = "Use tools to retrieve information from prior sessions (memory), then answer."
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        tool_calls: list[dict] = []
        turn_stats: list[dict] = []
        reasoning_log: list[str] = []

        def _result(output, turns, exhausted):
            return {"output": (output or "").strip(), "tool_calls": tool_calls,
                    "turns": turns, "turn_stats": turn_stats,
                    "reasoning_content": reasoning_log, "exhausted": exhausted}

        for turn in range(self.max_turns):
            r = self.llm.chat_with_tools(messages, tool_schemas)
            if r.get("reasoning"):
                reasoning_log.append(r["reasoning"])
            tcs = r.get("tool_calls") or []
            turn_stats.append({"round": turn + 1, "input_tokens": r.get("input_tokens", 0),
                               "output_tokens": r.get("output_tokens", 0),
                               "had_tool_call": bool(tcs),
                               "had_reasoning": bool(r.get("reasoning"))})
            if not tcs:
                return _result(r.get("content"), turn + 1, False)
            # Echo the assistant tool-call message, then append each tool result.
            messages.append({
                "role": "assistant",
                "content": r.get("content") or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in tcs
                ],
            })
            for tc in tcs:
                spec = self.tools.get(tc["name"])
                if spec:
                    try:
                        result = spec.call(tc["arguments"])
                    except Exception as e:  # noqa: BLE001
                        result = f"ERROR: {type(e).__name__}: {e}"
                else:
                    result = f"ERROR: unknown tool '{tc['name']}'"
                tool_calls.append({"tool": tc["name"], "input": tc["arguments"], "result": result})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})

        # Exhausted: ask once for a direct final answer (no tools).
        messages.append({"role": "user",
                         "content": "Stop using tools. Give your single best final answer now."})
        r = self.llm.chat_with_tools(messages, tool_schemas)
        if r.get("reasoning"):
            reasoning_log.append(r["reasoning"])
        # Record the forced call so loop_utilization counts it (parity with the
        # text path's forced_stats; exhausted_count keys off "forced").
        turn_stats.append({"round": self.max_turns + 1,
                           "input_tokens": r.get("input_tokens", 0),
                           "output_tokens": r.get("output_tokens", 0),
                           "had_tool_call": False, "forced": True,
                           "had_reasoning": bool(r.get("reasoning"))})
        return _result(r.get("content"), self.max_turns, True)


# Canonical "memory as a tool" helper for scenario kits. Not used by the agent
# directly — the agent uses whatever ToolRegistry the scenario assembles.

def build_search_memory_tool(memory_policy: MemoryPolicy, top_k: int = 3):
    """Return a ToolSpec wrapping the policy's retrieval for the agent.

    Routing:
      (1) If the policy class overrides ``MemoryPolicy.retrieve`` (e.g.
          AppendOnlyPolicy's cosine retrieval), use it — returns ranked hits.
      (2) Otherwise fall back to paragraph-chunk substring matching over
          ``policy.read()`` — correct for single-blob policies (SummarizeStore,
          GrowingHistory), where the base ``retrieve`` would return the whole
          memory as one chunk and defeat the search.
    """
    from .tools import ToolSpec

    # Class-level override check (handles inherited custom retrieve).
    has_custom_retrieve = (
        type(memory_policy).retrieve is not MemoryPolicy.retrieve
    )

    def _format_hits(hits: list) -> str:
        """Format retrieve() result list as a top-k joined string."""
        formatted: list[str] = []
        for h in hits[:top_k]:
            if isinstance(h, dict):
                formatted.append(str(h.get("text", h)))
            else:
                formatted.append(str(h))
        return "\n---\n".join(formatted)

    def _search_impl(args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "(empty query)"

        # (1) Policy-native ranked retrieval (cosine, BM25, etc.) when the
        # policy provides a real implementation. The base default returns the
        # full memory as a single chunk, which would equate to a memory dump,
        # so we only take this branch when the class explicitly overrides.
        if has_custom_retrieve:
            try:
                hits = memory_policy.retrieve(query, top_k=top_k)
                if hits:
                    return _format_hits(hits)
                return "(no matches)"
            except Exception:
                pass  # fall through to substring match

        # (2) Single-blob fallback: paragraph chunking + substring containment.
        # This is the experimentally meaningful behavior for SummarizeStore /
        # GrowingHistory: we measure whether the literal tokens needed for the
        # query survived compaction. A vector retriever would smooth over the
        # very decay we're trying to quantify.
        full = memory_policy.read() or ""
        if not full:
            return "(memory is empty)"
        chunks = [c.strip() for c in re.split(r"\n\n+", full) if c.strip()]
        q = query.lower()
        hits = [c for c in chunks if q in c.lower()]
        if hits:
            return "\n---\n".join(hits[:top_k])
        # Last-resort: return the most-recent chunks so the agent has SOMETHING.
        # Models the recency-bias aging mode: "couldn't find what I wanted,
        # here's the most recent context, take your best guess."
        return "\n---\n".join(chunks[-top_k:]) if chunks else "(no matches)"

    return ToolSpec(
        name="search_memory",
        version="1.0.0",
        description=(
            "Search your memory of prior sessions for content matching a query. "
            f"Returns up to {top_k} relevant chunks. Use specific keywords."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or short phrase to look up in memory.",
                },
            },
            "required": ["query"],
        },
        fn=_search_impl,
    )
