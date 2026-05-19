"""
agingbench/core/memory/workspace.py — Workspace-backed memory policy (used by S5 self-planning).

Unlike other MemoryPolicy implementations (summarize_store, append_only), this
policy does NOT actively manage memory. Instead, it passively observes a workspace
directory that the agent manages via file tools.

read()  → concatenates all workspace files (sorted by mtime, most recent first)
write() → NO-OP (the agent writes files directly via tools)
reset() → clears all files in the workspace

Use with build_memory_policy():
    memory_policy:
      type: custom
      class: "agingbench.core.memory.workspace:WorkspaceMemoryPolicy"
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from .base import MemoryPolicy


class WorkspaceMemoryPolicy(MemoryPolicy):
    """Memory policy backed by a filesystem workspace directory.

    The agent manages files via tools (write_file, read_file, list_files).
    This policy is a passive observer — it reads what the agent wrote.
    """

    def __init__(self, workspace_dir: str = "", **kwargs):
        """
        Args:
            workspace_dir: absolute path to workspace. If empty, a temp dir
                           is created (caller should set it before use).
        """
        if workspace_dir:
            self._dir = Path(workspace_dir)
            self._dir.mkdir(parents=True, exist_ok=True)
        else:
            self._dir = None  # caller must set via set_workspace_dir()

    def set_workspace_dir(self, path: str) -> None:
        """Set workspace directory (used when runner creates a temp dir)."""
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def workspace_dir(self) -> Path:
        if self._dir is None:
            raise RuntimeError("workspace_dir not set — call set_workspace_dir() first")
        return self._dir

    def read(self, query: Optional[str] = None) -> str:
        """Return concatenated content of all workspace files.

        Files are sorted by modification time (most recent first) so the
        agent's latest notes appear first in context.
        """
        if self._dir is None or not self._dir.exists():
            return ""

        files = []
        for f in self._dir.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                try:
                    files.append((f, f.stat().st_mtime))
                except OSError:
                    continue

        if not files:
            return ""

        # Sort by mtime descending (most recent first)
        files.sort(key=lambda x: x[1], reverse=True)

        parts = []
        for f, _ in files:
            rel = f.relative_to(self._dir)
            try:
                content = f.read_text(errors="replace")
            except Exception:
                continue
            parts.append(f"=== {rel} ===\n{content}")

        return "\n\n".join(parts)

    def write(self, new_content: str, llm=None) -> None:
        """No-op. The agent manages files directly via tools."""
        pass

    def reset(self) -> None:
        """Delete all files in workspace (but keep the directory)."""
        if self._dir and self._dir.exists():
            for item in self._dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

    def snapshot(self) -> str:
        """Return full file listing + contents for debugging."""
        return self.read()

    def file_listing(self) -> list[dict]:
        """Return [{path, size_bytes, mtime}] for workspace audit metrics."""
        if self._dir is None or not self._dir.exists():
            return []

        result = []
        for f in self._dir.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                try:
                    stat = f.stat()
                    result.append({
                        "path": str(f.relative_to(self._dir)),
                        "size_bytes": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                except OSError:
                    continue
        return sorted(result, key=lambda x: x["mtime"], reverse=True)
