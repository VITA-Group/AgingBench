"""
agingbench/baselines/claude_code_adapter.py — Adapter for invoking Claude Code.

Two modes:
  1. SDK mode (preferred): Uses claude-code-sdk Python package for programmatic control.
  2. Subprocess mode (fallback): Invokes `claude` CLI with --print flag.

Unlike BaseLLM, this adapter wraps an *agentic system* — Claude Code manages
its own tools (Read, Edit, Bash, etc.) and memory (.claude/ files). The
benchmark observes outputs and file changes rather than controlling internal
state.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ClaudeCodeError(RuntimeError):
    """Fatal Claude Code CLI failure (usage limit, auth, model access, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: int | None = None,
        api_status: int | None = None,
        raw: str | None = None,
    ):
        super().__init__(message)
        self.exit_code = exit_code
        self.api_status = api_status
        self.raw = raw


_FATAL_API_STATUSES = frozenset({401, 402, 403, 429, 529})
_FATAL_MESSAGE_MARKERS = (
    "rate limit",
    "usage limit",
    "quota",
    "billing",
    "overloaded",
    "too many requests",
    "credit",
    "subscription",
    "authentication",
    "not logged in",
    "out of extra usage",
    "exceeds",
)


def _is_fatal_api_error(message: str, api_status: int | None) -> bool:
    if api_status in _FATAL_API_STATUSES:
        return True
    low = message.lower()
    return any(marker in low for marker in _FATAL_MESSAGE_MARKERS)


def _raise_if_fatal_cli_error(
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> None:
    """Parse Claude Code JSON/text output and raise on unrecoverable failures."""
    message = stderr.strip()
    api_status: int | None = None
    is_error = False

    if stdout.strip():
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            if not message:
                message = stdout.strip()[:500]
        else:
            if isinstance(data, dict):
                is_error = bool(data.get("is_error"))
                api_status = data.get("api_error_status")
                message = str(
                    data.get("result")
                    or data.get("error")
                    or data.get("message")
                    or message
                )

    if exit_code != 0 or is_error:
        if not message:
            message = f"Claude Code exited with code {exit_code}"
        if exit_code != 0 or is_error or _is_fatal_api_error(message, api_status):
            raise ClaudeCodeError(
                message,
                exit_code=exit_code,
                api_status=api_status,
                raw=(stdout or stderr)[:1000] or None,
            )


@dataclass
class ClaudeCodeResponse:
    """Result of a single Claude Code invocation."""
    text: str
    session_id: Optional[str] = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    files_changed: list[str] = field(default_factory=list)
    num_turns: int = 0  # agent turns the CLI consumed; proxy for tool-use intensity


class ClaudeCodeAdapter:
    """
    Adapter for invoking Claude Code as a subprocess or via SDK.

    Parameters
    ----------
    model : str
        Model to use (e.g. "claude-sonnet-4-6-20250514").
    cwd : str or Path
        Working directory for Claude Code (the project it operates on).
    mode : str
        "subprocess" (default) or "sdk".
    max_turns : int
        Maximum agentic turns per invocation.
    cli_path : str
        Path to the claude CLI binary.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6-20250514",
        cwd: Optional[str | Path] = None,
        mode: str = "subprocess",
        max_turns: int = 50,
        cli_path: str = "claude",
        bare_mode: bool = False,
    ):
        self.model = model
        self.cwd = str(cwd) if cwd else os.getcwd()
        self.mode = mode
        self.max_turns = max_turns
        self.cli_path = cli_path
        self.bare_mode = bare_mode
        self._session_id: Optional[str] = None
        self._cli_version: Optional[str] = self._detect_version()

    def _detect_version(self) -> Optional[str]:
        """Detect the Claude CLI version at init time for reproducibility logging."""
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def send_message(
        self,
        message: str,
        resume: bool = False,
        system_prompt: Optional[str] = None,
    ) -> ClaudeCodeResponse:
        """
        Send a message to Claude Code and return the response.

        Parameters
        ----------
        message : str
            The user message / task for Claude Code.
        resume : bool
            If True, resume the previous session (preserving context).
        system_prompt : str, optional
            Custom system prompt to prepend.
        """
        if self.mode == "sdk":
            return self._send_sdk(message, resume, system_prompt)
        return self._send_subprocess(message, resume, system_prompt)

    def _send_subprocess(
        self,
        message: str,
        resume: bool,
        system_prompt: Optional[str],
    ) -> ClaudeCodeResponse:
        """Invoke Claude Code via CLI subprocess."""
        cmd = [
            self.cli_path,
            "--print",  # non-interactive, print output
            "--output-format", "json",
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            # Grant file tools scoped to cwd (workspace) only.
            # No unrestricted Bash — only read-only ls/cat/mkdir for safety.
            "--allowedTools", "Edit,Write,Read,Bash(ls*),Bash(cat*),Bash(mkdir*),Bash(pip*),Glob",
            # Bypass interactive permission prompts in non-interactive mode.
            "--dangerously-skip-permissions",
        ]

        if self.bare_mode:
            # Disable auto-memory, hooks, plugin sync, keychain reads
            # for reproducible benchmarking (P5: CLI agent reproducibility)
            cmd.append("--bare")

        if resume and self._session_id:
            cmd.extend(["--resume", self._session_id])

        if system_prompt:
            # Use --append-system-prompt to ADD instructions while keeping
            # Claude Code's default capabilities (file tools, coding, etc.)
            cmd.extend(["--append-system-prompt", system_prompt])

        # Pass the message via stdin to avoid shell escaping issues
        full_message = message

        try:
            result = subprocess.run(
                cmd,
                input=full_message,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                timeout=600,  # 10 minute timeout per invocation
                env={**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
            )

            _raise_if_fatal_cli_error(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

            # Parse JSON output
            return self._parse_json_output(result.stdout)

        except ClaudeCodeError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeError("Claude Code timed out after 600s") from exc
        except FileNotFoundError as exc:
            raise ClaudeCodeError(
                f"Claude Code CLI not found at '{self.cli_path}'. "
                f"Install with: npm install -g @anthropic-ai/claude-code"
            ) from exc

    def _send_sdk(
        self,
        message: str,
        resume: bool,
        system_prompt: Optional[str],
    ) -> ClaudeCodeResponse:
        """Invoke Claude Code via the Python SDK (claude-code-sdk)."""
        try:
            import asyncio
            from claude_code_sdk import ClaudeCode, ClaudeCodeOptions

            options = ClaudeCodeOptions(
                model=self.model,
                cwd=self.cwd,
                max_turns=self.max_turns,
            )
            if system_prompt:
                options.system_prompt = system_prompt
            if resume and self._session_id:
                options.resume = self._session_id

            # Run the async SDK call
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self._run_sdk_async(message, options)
                )
            finally:
                loop.close()

            return result

        except ImportError:
            print("[warn] claude-code-sdk not installed, falling back to subprocess")
            self.mode = "subprocess"
            return self._send_subprocess(message, resume, system_prompt)

    async def _run_sdk_async(self, message: str, options) -> ClaudeCodeResponse:
        """Async helper for SDK mode."""
        from claude_code_sdk import ClaudeCode

        text_parts = []
        async for event in ClaudeCode.create(prompt=message, options=options):
            if hasattr(event, "content"):
                text_parts.append(str(event.content))
            elif hasattr(event, "text"):
                text_parts.append(event.text)

        return ClaudeCodeResponse(
            text="\n".join(text_parts) if text_parts else "(no output)",
        )

    def _parse_json_output(self, stdout: str) -> ClaudeCodeResponse:
        """Parse Claude Code's JSON output format."""
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Fallback: treat as plain text
            return ClaudeCodeResponse(text=stdout.strip())

        # Claude Code JSON output varies by version; handle common formats
        if isinstance(data, dict):
            if data.get("is_error"):
                message = str(
                    data.get("result")
                    or data.get("error")
                    or data.get("message")
                    or "Claude Code returned is_error=true"
                )
                raise ClaudeCodeError(
                    message,
                    api_status=data.get("api_error_status"),
                    raw=stdout[:1000],
                )
            text = data.get("result", data.get("text", data.get("content", "")))
            session_id = data.get("session_id")
            if session_id:
                self._session_id = session_id
            return ClaudeCodeResponse(
                text=str(text),
                session_id=session_id,
                cost_usd=data.get("cost_usd", 0.0),
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                duration_ms=data.get("duration_ms", 0),
                num_turns=int(data.get("num_turns", 0)),
            )
        elif isinstance(data, list):
            # Array of message blocks
            texts = []
            for item in data:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif item.get("type") == "result":
                        texts.append(item.get("result", ""))
            return ClaudeCodeResponse(text="\n".join(texts) if texts else str(data))

        return ClaudeCodeResponse(text=str(data))

    def reset_session(self) -> None:
        """Clear session state for a fresh conversation."""
        self._session_id = None

    def get_memory_files(self) -> dict[str, str]:
        """
        Read Claude Code's native memory files from the project directory.

        Returns a dict mapping file paths to their contents.
        """
        memory_dir = Path(self.cwd) / ".claude"
        files = {}

        if not memory_dir.exists():
            return files

        # Read all markdown files in .claude/
        for md_file in memory_dir.rglob("*.md"):
            try:
                files[str(md_file.relative_to(self.cwd))] = md_file.read_text()
            except Exception:
                pass

        # Also check for CLAUDE.md at project root
        claude_md = Path(self.cwd) / "CLAUDE.md"
        if claude_md.exists():
            files["CLAUDE.md"] = claude_md.read_text()

        return files

    def snapshot_project(self) -> dict[str, str]:
        """
        Snapshot all Python files in the project directory.

        Returns a dict mapping relative paths to file contents.
        Used by the validator to check constraint adherence in generated code.
        """
        project_dir = Path(self.cwd)
        snapshot = {}

        for py_file in project_dir.rglob("*.py"):
            rel = str(py_file.relative_to(project_dir))
            # Skip __pycache__ and hidden dirs
            if "__pycache__" in rel or rel.startswith("."):
                continue
            try:
                snapshot[rel] = py_file.read_text()
            except Exception:
                pass

        return snapshot
