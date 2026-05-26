"""
agingbench/core/agent_adapter.py — Black-box interface for evaluating any agentic system.

Adapter-driven evaluation sends messages to agents and measures responses.
The agent manages its own tools, memory, and planning. The benchmark only
controls:
  (1) what tasks to send
  (2) when to reset conversation state
  (3) how to score responses

Adapters:
  - ReactFileAdapter: reference ReAct agent with file tools (core/adapters/)
  - ClaudeCodeAgentAdapter: wraps existing ClaudeCodeAdapter (core/adapters/)
  - CursorAgentAdapter: wraps Cursor Agent CLI (core/adapters/)
  - CodexAdapter: stub for OpenAI Codex (core/adapters/)
  - Custom: user implements AgentAdapter for their system
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentResponse:
    """Response from any agent adapter."""

    text: str
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    files_changed: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class AgentAdapter(ABC):
    """Black-box interface for evaluating any agentic system.

    The benchmark sends messages and receives responses.
    The agent manages its own tools, memory, and planning.

    Contract:
      - send_message(): deliver a task, get a response
      - reset_session(): clear ephemeral state (conversation), keep persistent
        state (workspace files, database, .claude/ files)
      - get_workspace_state(): optionally inspect persistent state for metrics
      - get_memory_text(): optionally return readable memory for keyword scoring
    """

    @abstractmethod
    def send_message(self, message: str) -> AgentResponse:
        """Send a task to the agent and receive a response.

        The message is a plain-text task prompt. The adapter may wrap it in
        its native frame (e.g., prepend "Update project notes: " for Claude Code).
        """

    @abstractmethod
    def reset_session(self) -> None:
        """Signal a session boundary.

        Clear ephemeral state (conversation history, in-context messages).
        Keep persistent state (workspace files, databases, .claude/ files).
        The next send_message() starts a fresh conversation but can access
        any files the agent previously created.
        """

    def get_workspace_state(self) -> dict:
        """Optionally inspect the agent's persistent state for metrics.

        Returns:
            dict with optional keys:
                files: list[dict] — [{path, size_bytes, mtime}, ...]
                total_bytes: int — total size of all persistent state
        Default: empty dict (opaque agent — no workspace visibility).

        Scoring impact: AgingBench's probe scorers fall back to response-text
        scoring when this is empty; metrics still emit, but signals tied to
        workspace persistence (S5/S7 file-tracking probes, the lineage-
        continuity component of telemetry-mode interference) will be muted.
        If your agent writes notes / state / memos to a known directory,
        returning those files here meaningfully sharpens the AgingCard.
        """
        return {}

    def get_memory_text(self) -> str:
        """Optionally return readable memory content for keyword scoring.

        For transparent adapters, this returns the concatenated content of
        workspace files. For opaque adapters, returns empty string (benchmark
        relies on response-based scoring via recall probes instead).

        Scoring impact: when empty, keyword-survival probes (S1, S2, S6) and
        the entity-recall component of S3 score only against the agent's
        natural-language reply — they can't credit information that lives
        in the agent's *memory* but wasn't recited in the response. Opaque
        adapters that don't expose memory will read as slightly more aged
        on these scenarios than they otherwise would.
        """
        return ""


def build_custom_adapter(adapter_cfg: dict, workspace_dir):
    """Instantiate an AgentAdapter declared in a SUT YAML's ``adapter:`` block.

    Used by Tier-2 runners (S5 self-planning, S7+ research-notes, S8 SWE-bench)
    to load a user-provided adapter without editing the dispatch chain.

    Expected SUT YAML shape:

        adapter:
          type: custom
          class: my_pkg.my_module:MyAdapter   # importable module:ClassName
          # any further keys are passed as kwargs to MyAdapter(...)
          model: gpt-4o
          max_turns: 30

    The constructor is always passed ``cwd=workspace_dir`` (a string path) and
    any remaining keys from ``adapter:``. Mirrors ``build_memory_policy``'s
    ``type: custom`` hook in ``core/memory/base.py``.
    """
    import importlib

    class_spec = adapter_cfg.get("class", "")
    if ":" not in class_spec:
        raise ValueError(
            "custom adapter requires 'class' in 'module:ClassName' format, "
            f"got {class_spec!r}"
        )
    module_path, class_name = class_spec.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    kwargs = {k: v for k, v in adapter_cfg.items() if k not in ("type", "class")}
    kwargs.setdefault("cwd", str(workspace_dir))
    instance = cls(**kwargs)
    if not isinstance(instance, AgentAdapter):
        raise TypeError(
            f"{class_spec} did not return an AgentAdapter subclass instance"
        )
    return instance
