"""
agingbench/core/cursor_adapter.py — Adapter for invoking Cursor Agent CLI.

Invokes the Cursor Agent via `agent -p` (headless print mode) with JSON output.
Unlike BaseLLM, this adapter wraps an agentic system — Cursor manages its own
tools and workspace files. The benchmark observes outputs rather than controlling
internal state.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CursorResponse:
    """Result of a single Cursor Agent invocation."""

    text: str
    session_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    files_changed: list[str] = field(default_factory=list)
    is_error: bool = False


class CursorAdapter:
    """Adapter for invoking Cursor Agent as a subprocess.

    Parameters
    ----------
    model : str
        Model id (e.g. "composer-2", "gpt-5.2-codex").
    cwd : str or Path
        Working directory / workspace for the agent.
    cli_path : str
        Path to the agent binary (default: "agent").
    timeout_sec : int
        Subprocess timeout per invocation.
    force : bool
        Pass --force to allow shell/file tools without prompts.
    trust : bool
        Pass --trust for headless workspace trust.
    sandbox : str or None
        Optional sandbox mode override ("enabled" or "disabled").
    """

    def __init__(
        self,
        model: str = "composer-2",
        cwd: Optional[str | Path] = None,
        cli_path: str = "agent",
        timeout_sec: int = 600,
        force: bool = True,
        trust: bool = True,
        sandbox: Optional[str] = None,
    ):
        self.model = model
        self.cwd = str(cwd) if cwd else os.getcwd()
        self.cli_path = cli_path
        self.timeout_sec = timeout_sec
        self.force = force
        self.trust = trust
        self.sandbox = sandbox
        self._session_id: Optional[str] = None
        self._cli_version: Optional[str] = self._detect_version()

    def _detect_version(self) -> Optional[str]:
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def send_message(
        self,
        message: str,
        resume: bool = False,
        system_prompt: Optional[str] = None,
    ) -> CursorResponse:
        combined = message
        if system_prompt:
            combined = system_prompt.rstrip() + "\n\n" + message

        cmd = [
            self.cli_path,
            "-p",
            "--output-format", "json",
            "--model", self.model,
            "--workspace", self.cwd,
        ]
        if self.force:
            cmd.append("--force")
        if self.trust:
            cmd.append("--trust")
        if self.sandbox in ("enabled", "disabled"):
            cmd.extend(["--sandbox", self.sandbox])
        if resume and self._session_id:
            cmd.extend(["--resume", self._session_id])
        cmd.append(combined)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return CursorResponse(text=f"[ERROR] Cursor Agent timed out after {self.timeout_sec}s")
        except FileNotFoundError:
            return CursorResponse(
                text=(
                    f"[ERROR] Cursor Agent CLI not found at '{self.cli_path}'. "
                    "Install with: curl -fsSL https://cursor.com/install | bash"
                ),
                is_error=True,
            )

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            return CursorResponse(
                text=f"[ERROR] Cursor Agent exited with code {result.returncode}: {err[:500]}",
                is_error=True,
            )

        stdout = result.stdout.strip()
        if not stdout:
            err = (result.stderr or "").strip()
            return CursorResponse(
                text=f"[ERROR] Cursor Agent returned no output{(': ' + err[:500]) if err else ''}",
                is_error=True,
            )
        return self._parse_json_output(stdout)

    def _parse_json_output(self, stdout: str) -> CursorResponse:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return CursorResponse(text=stdout)

        if not isinstance(data, dict):
            return CursorResponse(text=str(data))

        session_id = data.get("session_id")
        if session_id:
            self._session_id = session_id

        usage = data.get("usage") or {}
        input_tokens = int(
            usage.get("inputTokens")
            or usage.get("input_tokens")
            or 0
        )
        output_tokens = int(
            usage.get("outputTokens")
            or usage.get("output_tokens")
            or 0
        )

        return CursorResponse(
            text=str(data.get("result", data.get("text", data.get("content", "")))),
            session_id=session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=int(data.get("duration_ms") or data.get("duration_api_ms") or 0),
            is_error=bool(data.get("is_error")),
        )

    def reset_session(self) -> None:
        """Clear session state for a fresh conversation."""
        self._session_id = None
