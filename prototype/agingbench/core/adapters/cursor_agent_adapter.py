"""
agingbench/core/adapters/cursor_agent_adapter.py — Wrap CursorAdapter for Tier-2 eval.

Thin wrapper that adapts CursorAdapter (subprocess) to the AgentAdapter interface
for S7/S8 adapter-driven evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..agent_adapter import AgentAdapter, AgentResponse
from ..cursor_adapter import CursorAdapter


class CursorAgentAdapter(AgentAdapter):
    """Wraps CursorAdapter for Tier-2 evaluation.

    Cursor Agent manages its own tools and workspace files. This adapter
    provides the AgentAdapter interface on top.
    """

    MEMORY_SYSTEM_PROMPT = (
        "You are a long-running personal assistant. The user shares facts, preferences, "
        "and constraints across multiple conversations. You MUST save all user information "
        "to files in the notes/ directory of the current workspace (e.g., notes/dining.md, "
        "notes/contacts.md). CRITICAL: Before answering ANY question, you MUST read ALL "
        "files in notes/ using your Read tool. Do not guess or say 'I don't have that info' "
        "without first reading every file in notes/. The answer is almost certainly in one "
        "of your notes files. Always use your file tools to persist and retrieve information."
    )

    def __init__(
        self,
        adapter: Optional[CursorAdapter] = None,
        model: str = "composer-2",
        cwd: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        cli_path = kwargs.get("cli_path", "agent")
        timeout_sec = int(kwargs.get("timeout_sec", kwargs.get("subprocess_timeout", 600)))
        sandbox = kwargs.get("sandbox")
        if adapter:
            self.adapter = adapter
        else:
            self.adapter = CursorAdapter(
                model=model,
                cwd=cwd or os.getcwd(),
                cli_path=cli_path,
                timeout_sec=timeout_sec,
                force=kwargs.get("force", True),
                trust=kwargs.get("trust", True),
                sandbox=sandbox,
            )
        self._cwd = Path(self.adapter.cwd)
        self._system_prompt = system_prompt
        self._max_turns = int(kwargs.get("max_turns", 50))
        self._prev_workspace_state: dict | None = None

    def send_message(self, message: str) -> AgentResponse:
        has_session = self.adapter._session_id is not None
        prompt = self._system_prompt
        if prompt is None and not has_session:
            prompt = self.MEMORY_SYSTEM_PROMPT
        resp = self.adapter.send_message(
            message,
            resume=has_session,
            system_prompt=prompt if not has_session else None,
        )
        return AgentResponse(
            text=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            files_changed=resp.files_changed,
            metadata={
                "session_id": resp.session_id,
                "duration_ms": resp.duration_ms,
                "is_error": resp.is_error,
                "max_turns_setting": self._max_turns,
            },
        )

    def reset_session(self) -> None:
        """Start a new Cursor chat (clear conversation, keep workspace files)."""
        self.adapter.reset_session()

    def get_workspace_state(self) -> dict:
        """Read workspace memory directories for state inspection."""
        files = []
        total = 0
        current_files: dict[str, int] = {}
        for dirname in (".cursor", "notes", ".aging"):
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

        state_diff = None
        if self._prev_workspace_state is not None:
            prev = self._prev_workspace_state
            added = [p for p in current_files if p not in prev]
            removed = [p for p in prev if p not in current_files]
            modified = [
                p for p in current_files
                if p in prev and current_files[p] != prev[p]
            ]
            state_diff = {
                "files_added": added,
                "files_removed": removed,
                "files_modified": modified,
            }
        self._prev_workspace_state = current_files

        result = {"files": files, "total_bytes": total}
        if state_diff:
            result["state_diff"] = state_diff
        if self.adapter._cli_version:
            result["cli_version"] = self.adapter._cli_version
        return result

    def get_memory_text(self) -> str:
        """Read workspace memory files as concatenated text."""
        parts = []
        for dirname in (".cursor", "notes", ".aging"):
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
