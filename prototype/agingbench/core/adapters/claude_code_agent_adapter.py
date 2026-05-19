"""
agingbench/core/adapters/claude_code_agent_adapter.py — Wrap ClaudeCodeAdapter for S7.

Thin wrapper that adapts the existing ClaudeCodeAdapter (subprocess/SDK) to the
AgentAdapter interface for Tier-2 (adapter-driven) evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..agent_adapter import AgentAdapter, AgentResponse
from ..claude_code_adapter import ClaudeCodeAdapter


class ClaudeCodeAgentAdapter(AgentAdapter):
    """Wraps existing ClaudeCodeAdapter for S7 Tier-2 evaluation.

    Claude Code manages its own tools and .claude/ memory files.
    This adapter provides the AgentAdapter interface on top.
    """

    def __init__(
        self,
        adapter: Optional[ClaudeCodeAdapter] = None,
        model: str = "claude-sonnet-4-6-20250514",
        cwd: Optional[str] = None,
        max_turns: int = 50,
        **kwargs,
    ):
        cli_path = kwargs.get("cli_path", "claude")
        bare_mode = kwargs.get("bare_mode", False)
        if adapter:
            self.adapter = adapter
        else:
            self.adapter = ClaudeCodeAdapter(
                model=model,
                cwd=cwd or os.getcwd(),
                max_turns=max_turns,
                cli_path=cli_path,
                bare_mode=bare_mode,
            )
        self._cwd = Path(self.adapter.cwd)
        self._prev_workspace_state: dict | None = None  # for .claude/ diff tracking

    # System prompt that instructs Claude Code to persist information to files.
    # This is appended via --append-system-prompt, not --system-prompt, so
    # Claude Code's default coding capabilities remain available.
    MEMORY_SYSTEM_PROMPT = (
        "You are a long-running personal assistant. The user shares facts, preferences, "
        "and constraints across multiple conversations. You MUST save all user information "
        "to files in the notes/ directory of the current workspace (e.g., notes/dining.md, "
        "notes/contacts.md). CRITICAL: Before answering ANY question, you MUST read ALL "
        "files in notes/ using your Read tool. Do not guess or say 'I don't have that info' "
        "without first reading every file in notes/. The answer is almost certainly in one "
        "of your notes files. Always use your file tools to persist and retrieve information."
    )

    def send_message(self, message: str) -> AgentResponse:
        # Resume the session if we have one (multi-turn within a block).
        # First message of a new block has no session_id → starts fresh.
        # Subsequent messages within the same block → resume conversation.
        has_session = self.adapter._session_id is not None
        resp = self.adapter.send_message(
            message,
            resume=has_session,
            system_prompt=self.MEMORY_SYSTEM_PROMPT if not has_session else None,
        )
        return AgentResponse(
            text=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            files_changed=resp.files_changed,
            metadata={
                "session_id": resp.session_id,
                "cost_usd": resp.cost_usd,
                "duration_ms": resp.duration_ms,
                "num_turns": resp.num_turns,
            },
        )

    def reset_session(self) -> None:
        """Start a new Claude Code session (clear conversation, keep .claude/ files).

        Called between blocks to simulate "next day" — conversation history
        is cleared but workspace files persist. Within a block, messages
        use --resume to maintain conversation context (realistic multi-turn).
        """
        self.adapter._session_id = None

    def get_workspace_state(self) -> dict:
        """Read workspace memory directories (.claude/ and notes/) for state.

        Also computes a diff from previous call (tracks .claude/ mutations
        between blocks for reproducibility logging).
        """
        files = []
        total = 0
        current_files: dict[str, int] = {}
        for dirname in (".claude", "notes"):
            mem_dir = self._cwd / dirname
            if not mem_dir.exists():
                continue
            for f in mem_dir.rglob("*"):
                if f.is_file():
                    size = f.stat().st_size
                    rel = str(f.relative_to(self._cwd))
                    files.append({
                        "path": rel,
                        "size_bytes": size,
                        "mtime": f.stat().st_mtime,
                    })
                    total += size
                    current_files[rel] = size

        # Compute diff from previous snapshot
        state_diff = None
        if self._prev_workspace_state is not None:
            prev = self._prev_workspace_state
            added = [p for p in current_files if p not in prev]
            removed = [p for p in prev if p not in current_files]
            modified = [p for p in current_files
                        if p in prev and current_files[p] != prev[p]]
            state_diff = {
                "files_added": added,
                "files_removed": removed,
                "files_modified": modified,
            }
        self._prev_workspace_state = current_files

        result = {"files": files, "total_bytes": total}
        if state_diff:
            result["state_diff"] = state_diff
        # Include CLI version for reproducibility
        if self.adapter._cli_version:
            result["cli_version"] = self.adapter._cli_version
        return result

    def get_memory_text(self) -> str:
        """Read workspace memory files (.claude/ and notes/) as text."""
        parts = []
        for dirname in (".claude", "notes"):
            mem_dir = self._cwd / dirname
            if not mem_dir.exists():
                continue
            for f in sorted(mem_dir.rglob("*")):
                if f.is_file() and f.suffix in (".md", ".txt", ".json"):
                    try:
                        content = f.read_text(errors="replace")
                        parts.append(f"=== {f.relative_to(self._cwd)} ===\n{content}")
                    except Exception:
                        continue
        return "\n\n".join(parts)
