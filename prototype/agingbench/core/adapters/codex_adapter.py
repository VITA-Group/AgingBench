"""
agingbench/core/adapters/codex_adapter.py — Adapter for OpenAI Codex CLI agent.

Wraps the Codex CLI (https://github.com/openai/codex) for S7 Tier-2 evaluation.
Codex is invoked via `codex exec` in non-interactive mode with --json for
structured event output.

Requires:
  - npm i -g @openai/codex
  - OPENAI_API_KEY environment variable

Architecture parallel to ClaudeCodeAgentAdapter:
  - codex exec          ↔  claude --print
  - --full-auto         ↔  --dangerously-skip-permissions
  - --json              ↔  --output-format json
  - codex exec resume   ↔  claude --resume <session_id>
  - --cd <path>         ↔  cwd parameter
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..agent_adapter import AgentAdapter, AgentResponse


@dataclass
class CodexResponse:
    """Result of a single Codex CLI invocation."""
    text: str
    session_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    files_changed: list[str] = field(default_factory=list)


class CodexAdapter(AgentAdapter):
    """Adapter for OpenAI Codex CLI agent.

    Invokes `codex exec` in non-interactive mode. The agent manages its own
    tools (file read/write, shell commands) and workspace files. The benchmark
    observes outputs and file changes.

    Parameters
    ----------
    model : str
        Model to use (e.g. "o3", "gpt-4.1", "codex-mini").
    cwd : str or Path
        Working directory (workspace) for Codex.
    max_turns : int
        Maximum agentic turns per invocation.
    cli_path : str
        Path to the codex CLI binary.
    """

    def __init__(
        self,
        model: str = "codex-mini",
        cwd: Optional[str | Path] = None,
        max_turns: int = 25,
        cli_path: str = "codex",
        **kwargs,
    ):
        self.model = model
        self.cwd = str(cwd) if cwd else os.getcwd()
        self.max_turns = max_turns
        self.cli_path = cli_path
        self._session_id: Optional[str] = None

    # System prompt that instructs Codex to persist information to files.
    MEMORY_SYSTEM_PROMPT = (
        "You are a long-running personal assistant. The user shares facts, preferences, "
        "and constraints across multiple conversations. You MUST save all user information "
        "to files in the notes/ directory of the current workspace (e.g., notes/dining.md, "
        "notes/contacts.md). CRITICAL: Before answering ANY question, you MUST read ALL "
        "files in notes/ using your file tools. Do not guess or say 'I don't have that info' "
        "without first reading every file in notes/. The answer is almost certainly in one "
        "of your notes files. Always use your file tools to persist and retrieve information."
    )

    def send_message(self, message: str) -> AgentResponse:
        """Send a task to Codex and receive a response."""
        has_session = self._session_id is not None
        resp = self._send_exec(
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
                "duration_ms": resp.duration_ms,
            },
        )

    def _send_exec(
        self,
        message: str,
        resume: bool = False,
        system_prompt: Optional[str] = None,
    ) -> CodexResponse:
        """Invoke Codex CLI via `codex exec` subprocess.

        Flag compatibility note (v0.117.0+):
          - The `--instructions` flag was removed; system prompts are now
            prepended to the user message as part of the single PROMPT arg.
          - Non-git workspaces require `--skip-git-repo-check`; benchmark
            workspaces are created ad hoc and are almost never git repos.
          - Resume uses positional SESSION_ID followed by PROMPT (flags go
            before the positionals for subcommand options to parse correctly).
        """
        # Merge system prompt into the message since --instructions is gone
        combined = (system_prompt.rstrip() + "\n\n" + message) if system_prompt else message

        if resume and self._session_id:
            # `codex exec resume [OPTIONS] <SESSION_ID> [PROMPT]`
            cmd = [
                self.cli_path, "exec",
                "--json",
                "--full-auto",
                "--skip-git-repo-check",
                "--cd", self.cwd,
                "resume",
                self._session_id,
            ]
            if combined:
                cmd.append(combined)
        else:
            # Start a new session
            cmd = [
                self.cli_path, "exec",
                "--json",
                "--full-auto",
                "--skip-git-repo-check",
                "--model", self.model,
                "--cd", self.cwd,
            ]
            cmd.append(combined)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ},
            )

            if result.returncode != 0:
                return CodexResponse(
                    text=f"[ERROR] Codex exited with code {result.returncode}: "
                         f"{result.stderr[:500]}",
                )

            return self._parse_json_events(result.stdout)

        except subprocess.TimeoutExpired:
            return CodexResponse(text="[ERROR] Codex timed out after 600s")
        except FileNotFoundError:
            return CodexResponse(
                text=f"[ERROR] Codex CLI not found at '{self.cli_path}'. "
                     f"Install with: npm i -g @openai/codex"
            )

    def _parse_json_events(self, stdout: str) -> CodexResponse:
        """Parse Codex's JSONL event stream from `codex exec --json`.

        Event types:
          - thread.started: contains session/thread ID
          - turn.completed: contains agent messages
          - item.*: file changes, command executions, agent messages
          - turn.failed / error: failures
        """
        text_parts = []
        session_id = None
        files_changed = []
        input_tokens = 0
        output_tokens = 0

        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Not JSON — might be plain text progress from stderr leak
                continue

            etype = event.get("type", "")

            # Extract session/thread ID
            if etype == "thread.started":
                session_id = event.get("thread_id") or event.get("session_id")

            # Extract agent text messages
            if etype in ("item.message", "item.agent_message"):
                content = event.get("content") or event.get("text") or ""
                if isinstance(content, list):
                    # Content blocks: [{"type": "text", "text": "..."}]
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                elif isinstance(content, str):
                    text_parts.append(content)

            # Extract file changes
            if etype in ("item.file_change", "item.file_edit"):
                path = event.get("path") or event.get("file") or ""
                if path:
                    files_changed.append(path)

            # Extract command execution results as fallback text
            if etype == "item.command_execution":
                output = event.get("output", "")
                if output:
                    text_parts.append(f"[exec] {output[:200]}")

            # Extract token usage from turn.completed
            if etype == "turn.completed":
                usage = event.get("usage", {})
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                # Also capture final message if present
                msg = event.get("message") or event.get("content") or ""
                if isinstance(msg, str) and msg:
                    text_parts.append(msg)

            # Handle errors
            if etype in ("turn.failed", "error"):
                err = event.get("error") or event.get("message") or str(event)
                text_parts.append(f"[ERROR] {err}")

        if session_id:
            self._session_id = session_id

        return CodexResponse(
            text="\n".join(text_parts) if text_parts else "(no output)",
            session_id=session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            files_changed=files_changed,
        )

    def reset_session(self) -> None:
        """Clear session state for a fresh conversation.

        Clears the session ID so the next send_message() starts a new
        Codex exec session. Workspace files persist on disk.
        """
        self._session_id = None

    def get_workspace_state(self) -> dict:
        """Read workspace memory directories (notes/) for state inspection."""
        files = []
        total = 0
        workspace = Path(self.cwd)
        for dirname in ("notes",):
            mem_dir = workspace / dirname
            if not mem_dir.exists():
                continue
            for f in mem_dir.rglob("*"):
                if f.is_file():
                    size = f.stat().st_size
                    files.append({
                        "path": str(f.relative_to(workspace)),
                        "size_bytes": size,
                        "mtime": f.stat().st_mtime,
                    })
                    total += size
        return {"files": files, "total_bytes": total}

    def get_memory_text(self) -> str:
        """Read workspace memory files (notes/) as concatenated text."""
        parts = []
        workspace = Path(self.cwd)
        notes_dir = workspace / "notes"
        if not notes_dir.exists():
            return ""
        for f in sorted(notes_dir.rglob("*")):
            if f.is_file() and f.suffix in (".md", ".txt", ".json"):
                try:
                    content = f.read_text(errors="replace")
                    parts.append(f"=== {f.relative_to(workspace)} ===\n{content}")
                except Exception:
                    continue
        return "\n\n".join(parts)
