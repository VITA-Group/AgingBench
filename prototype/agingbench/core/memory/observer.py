"""
agingbench/baselines/memory/observer.py — Read-only observer of Claude Code's
native memory files.

Unlike other MemoryPolicy implementations, ObserverPolicy does NOT manage
memory — Claude Code manages its own .claude/ files and CLAUDE.md. This
policy only *observes* the current state for scoring and tracing.

write() is a no-op. read() returns the concatenated content of all memory
files that Claude Code has created.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import MemoryPolicy


class ObserverPolicy(MemoryPolicy):
    """
    Read-only observer of Claude Code's native memory system.

    Reads from:
      - {project_dir}/CLAUDE.md
      - {project_dir}/.claude/MEMORY.md
      - {project_dir}/.claude/projects/*/CLAUDE.md
      - {project_dir}/.claude/projects/*/*.md (individual memory files)

    write() is a no-op — Claude Code decides what to persist.
    reset() clears the .claude/ directory for a fresh run.
    """

    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir)
        self.n_writes = 0  # Track for compatibility with trace logger

    def read(self, query: Optional[str] = None) -> str:
        """
        Return the concatenated content of all Claude Code memory files.

        This represents M_t — what Claude Code has chosen to persist.
        """
        parts = []

        # 1. Project-root CLAUDE.md
        claude_md = self.project_dir / "CLAUDE.md"
        if claude_md.exists():
            parts.append(f"=== CLAUDE.md ===\n{claude_md.read_text()}")

        # 2. .claude/MEMORY.md (the memory index)
        memory_md = self.project_dir / ".claude" / "MEMORY.md"
        if memory_md.exists():
            parts.append(f"=== .claude/MEMORY.md ===\n{memory_md.read_text()}")

        # 3. Individual memory files in .claude/ subdirectories
        claude_dir = self.project_dir / ".claude"
        if claude_dir.exists():
            for md_file in sorted(claude_dir.rglob("*.md")):
                rel = md_file.relative_to(self.project_dir)
                if str(rel) in ("CLAUDE.md", ".claude/MEMORY.md"):
                    continue  # Already included above
                try:
                    content = md_file.read_text().strip()
                    if content:
                        parts.append(f"=== {rel} ===\n{content}")
                except Exception:
                    pass

        return "\n\n".join(parts) if parts else ""

    def write(self, new_content: str, llm=None) -> None:
        """
        No-op. Claude Code manages its own memory.

        We still increment n_writes for trace compatibility.
        """
        self.n_writes += 1

    def reset(self) -> None:
        """
        Clear Claude Code's memory files for a fresh run.

        Removes .claude/ directory contents and CLAUDE.md.
        """
        import shutil

        claude_dir = self.project_dir / ".claude"
        if claude_dir.exists():
            shutil.rmtree(claude_dir)

        claude_md = self.project_dir / "CLAUDE.md"
        if claude_md.exists():
            claude_md.unlink()

        self.n_writes = 0

    def snapshot(self) -> str:
        """Return raw memory content for scoring."""
        return self.read()

    def memory_file_count(self) -> int:
        """Count the number of memory files Claude Code has created."""
        count = 0
        claude_dir = self.project_dir / ".claude"
        if claude_dir.exists():
            count += len(list(claude_dir.rglob("*.md")))
        if (self.project_dir / "CLAUDE.md").exists():
            count += 1
        return count

    def memory_token_estimate(self) -> int:
        """Rough token estimate of all memory content."""
        content = self.read()
        return len(content) // 4 if content else 0
