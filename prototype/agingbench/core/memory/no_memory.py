"""
agingbench/baselines/memory/no_memory.py — Control-group policy.

The agent receives no persistent memory between sessions.
Establishes m(0) performance ceiling in the experimental matrix.
"""

from typing import Optional
from .base import MemoryPolicy


class NoMemoryPolicy(MemoryPolicy):
    """Discard all writes; always return empty string on read."""

    def read(self, query: Optional[str] = None) -> str:
        return ""

    def write(self, new_content: str, llm=None) -> None:
        pass  # intentionally a no-op

    def reset(self) -> None:
        pass
