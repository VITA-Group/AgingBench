"""
agingbench/core/adapters/openhands_adapter.py — OpenHands Tier-2 agent adapter.

Wraps the OpenHands SDK (installed in a separate conda env) via a subprocess
bridge so that agingbench's environment does not need the ~100-package
openhands-ai dependency set.

Pattern mirrors ClaudeCodeAgentAdapter:
  - send_message() delegates to openhands_bridge.py via subprocess
  - reset_session() clears the conversation_id (fresh conversation next turn)
  - get_workspace_state() / get_memory_text() inspect the persistent workspace
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..agent_adapter import AgentAdapter, AgentResponse

_BRIDGE_PY = Path(__file__).parent / "openhands_bridge.py"

# Default resolution order for the Python interpreter that runs the
# OpenHands SDK in its isolated env:
#   1. SUT YAML `adapter.bridge_python` (passed to __init__)
#   2. OPENHANDS_BRIDGE_PYTHON env var
#   3. Auto-detect an `openhands` conda env on PATH (e.g. via `conda env list`)
#   4. None → raise a clear error explaining the install path
#
# Previously this was hardcoded to a developer-machine path, which fails
# silently on every other machine with a missing-file subprocess error.
_DEFAULT_BRIDGE_PYTHON: Optional[str] = None


def _autodetect_openhands_python() -> Optional[str]:
    """Best-effort: look for an OpenHands-compatible python on this machine.

    Checks:
      1. ${CONDA_PREFIX}/../openhands/bin/python  (sibling conda env)
      2. ~/anaconda3/envs/openhands/bin/python
      3. ~/miniconda3/envs/openhands/bin/python
      4. The first `python` on PATH that can `import openhands.sdk`

    Returns the absolute path or None.
    """
    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix).parent / "openhands" / "bin" / "python")
    home = Path.home()
    candidates.extend([
        home / "anaconda3" / "envs" / "openhands" / "bin" / "python",
        home / "miniconda3" / "envs" / "openhands" / "bin" / "python",
        home / ".pyenv" / "versions" / "openhands" / "bin" / "python",
    ])
    for c in candidates:
        if c.is_file() and os.access(str(c), os.X_OK):
            return str(c)
    return None

_DEFAULT_SYSTEM_PROMPT = (
    "You are a long-running personal assistant. The user shares facts, preferences, "
    "and constraints across multiple conversations. You MUST save all user information "
    "to files in the notes/ directory of the current workspace (e.g., notes/dining.md, "
    "notes/contacts.md). Before answering ANY question, read every file in notes/ first. "
    "Always persist new information and retrieve prior information through the file system."
)


class OpenHandsAdapter(AgentAdapter):
    """Runs an OpenHands agent via subprocess bridge to the isolated env.

    Configuration (SUT YAML adapter block):
        adapter:
          type: openhands
          model: gpt-4o-mini           # litellm model string
          cwd: /path/to/workspace      # persistent workspace
          max_turns: 30
          bridge_python: /path/to/python   # optional; defaults to openhands env
          system_prompt: "..."         # optional; overrides default notes-policy prompt
          api_key_env: OPENAI_API_KEY  # optional; which env var holds the key
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        cwd: Optional[str] = None,
        max_turns: int = 30,
        bridge_python: Optional[str] = None,
        system_prompt: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: Optional[str] = None,  # "low" | "medium" | "high"
        preset: Optional[str] = None,             # "default" | "gpt5" | None (auto-detect)
        subprocess_timeout: int = 1800,           # per-turn wall-clock cap (s)
        **kwargs,
    ):
        self._cwd = Path(cwd or os.getcwd()).resolve()
        self._cwd.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._max_turns = max_turns
        # Resolution: explicit > env var > autodetect > clear error
        self._bridge_python = (
            bridge_python
            or os.environ.get("OPENHANDS_BRIDGE_PYTHON")
            or _autodetect_openhands_python()
        )
        if self._bridge_python is None:
            raise RuntimeError(
                "OpenHandsAdapter: cannot locate a Python interpreter with the "
                "OpenHands SDK installed. Set OPENHANDS_BRIDGE_PYTHON to the "
                "absolute path of the python in your openhands env, or pass "
                "`bridge_python: /path/to/python` in your SUT YAML's "
                "adapter block. Install OpenHands with: "
                "`conda create -n openhands python=3.11 && conda activate openhands && pip install openhands-sdk openhands-tools`."
            )
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._api_key_env = api_key_env
        self._reasoning_effort = reasoning_effort
        self._preset = preset
        self._subprocess_timeout = subprocess_timeout

        self._persistence_dir = self._cwd / ".openhands_persist"
        self._persistence_dir.mkdir(parents=True, exist_ok=True)
        self._workspace_dir = self._cwd
        self._conversation_id: Optional[str] = None
        self._prev_workspace_state: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # AgentAdapter interface
    # ------------------------------------------------------------------

    def send_message(self, message: str) -> AgentResponse:
        api_key = os.environ.get(self._api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"OpenHandsAdapter: env var {self._api_key_env} not set "
                "(load .env before invoking agingbench)."
            )

        req = {
            "task": message,
            "workspace_dir": str(self._workspace_dir),
            "persistence_dir": str(self._persistence_dir),
            "model": self._model,
            "api_key": api_key,
            "conversation_id": self._conversation_id,
            "system_prompt": self._system_prompt if self._conversation_id is None else None,
            "max_iterations": self._max_turns,
            "reasoning_effort": self._reasoning_effort,
            "preset": self._preset,
        }

        t0 = time.time()
        proc = subprocess.run(
            [self._bridge_python, str(_BRIDGE_PY)],
            input=json.dumps(req),
            capture_output=True,
            text=True,
            timeout=self._subprocess_timeout,
        )
        duration_ms = int((time.time() - t0) * 1000)

        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenHands bridge exited {proc.returncode}\n"
                f"stdout: {proc.stdout[:500]}\nstderr: {proc.stderr[:500]}"
            )

        try:
            resp = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"OpenHands bridge returned invalid JSON: {e}\n"
                f"stdout: {proc.stdout[:500]}"
            )

        if resp.get("error"):
            raise RuntimeError(f"OpenHands bridge error: {resp['error']}")

        # Store conversation_id for next turn (resumption within a block)
        self._conversation_id = resp.get("conversation_id") or self._conversation_id

        return AgentResponse(
            text=resp.get("text", ""),
            input_tokens=resp.get("input_tokens", 0),
            output_tokens=resp.get("output_tokens", 0),
            files_changed=resp.get("files_changed", []),
            metadata={
                "conversation_id": self._conversation_id,
                "cost_usd": resp.get("cost_usd", 0.0),
                "duration_ms": duration_ms,
                "iterations": resp.get("iterations", 0),
                "num_turns": resp.get("iterations", 0),  # alias for cross-framework consistency
                "model": self._model,
            },
        )

    def reset_session(self) -> None:
        """Start a fresh OpenHands conversation on the next send_message.

        Workspace files (notes/, etc.) persist across blocks; only the
        conversation history is cleared.
        """
        self._conversation_id = None

    # ------------------------------------------------------------------
    # Workspace inspection (mirrors ClaudeCodeAgentAdapter pattern)
    # ------------------------------------------------------------------

    def _memory_dirs(self) -> list[Path]:
        return [self._cwd / "notes", self._cwd / ".openhands_memory"]

    def get_workspace_state(self) -> dict:
        files = []
        total = 0
        current: dict[str, int] = {}
        for mem_dir in self._memory_dirs():
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
                    current[rel] = size

        state_diff = None
        if self._prev_workspace_state is not None:
            prev = self._prev_workspace_state
            added = [p for p in current if p not in prev]
            removed = [p for p in prev if p not in current]
            modified = [p for p in current
                        if p in prev and current[p] != prev[p]]
            state_diff = {
                "files_added": added,
                "files_removed": removed,
                "files_modified": modified,
            }
        self._prev_workspace_state = current

        result = {"files": files, "total_bytes": total, "model": self._model}
        if state_diff:
            result["state_diff"] = state_diff
        return result

    def get_memory_text(self) -> str:
        parts = []
        for mem_dir in self._memory_dirs():
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
