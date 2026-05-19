"""
agingbench/baselines/memory/summarize_store.py — Lossy consolidation policy.

Primary failure driver for S1 (cross-session drift) and S2 (recursive compression).
After each write(), the current memory is replaced by a compressed version
produced by the LLM. Each compaction pass is a potential lossy step — the
information that survives is exactly what the LLM chose to preserve.

The compaction prompt is configurable (pass prompt_template at construction).
Default uses MEDIUM aggressiveness: instructs the model to preserve named
values, but forces a rewrite (not verbatim copy), inducing gradual paraphrasing.
"""

from typing import Optional
from .base import MemoryPolicy

COMPACT_MEDIUM = """You are a project knowledge manager. Below is a project specification document.
Rewrite it as a concise summary. You MUST preserve ALL of the following verbatim:
- Every specific budget figure (exact dollar amounts with the $ sign)
- Every deadline (exact dates including month and day)
- Every named person and their assigned role
- Every technical constraint (specific version numbers and technology names)
Do not omit any named constraint. Use clear, direct language. Be concise but complete.

DOCUMENT:
{text}

SUMMARY:"""


class SummarizeStorePolicy(MemoryPolicy):
    def __init__(self, prompt_template: str = COMPACT_MEDIUM, word_budget: Optional[int] = None):
        self._memory: str = ""
        self.prompt_template = prompt_template
        self.word_budget = word_budget
        self.n_writes: int = 0
        # Token usage from the most recent write(); read by runner for trace logging.
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    def read(self, query: Optional[str] = None) -> str:
        return self._memory

    def write(self, new_content: str, llm=None) -> None:
        """
        Combine existing memory with new_content, then compress.
        Uses chat_with_usage() when the LLM supports it to capture token counts.
        Falls back to plain concatenation when llm is None (unit tests).
        """
        combined = (self._memory + "\n\n" + new_content).strip() if self._memory else new_content

        if llm is not None:
            messages = [{"role": "user", "content": self.prompt_template.format(text=combined)}]
            if hasattr(llm, "chat_with_usage"):
                resp = llm.chat_with_usage(messages)
                self._memory = resp.text.strip()
                self.last_input_tokens = resp.input_tokens
                self.last_output_tokens = resp.output_tokens
            else:
                self._memory = llm.chat(messages).strip()
                self.last_input_tokens = self.last_output_tokens = 0
        else:
            self._memory = combined
            self.last_input_tokens = self.last_output_tokens = 0

        # Hard truncation to word_budget (if set) — guarantees compression budget
        # even when the model ignores the prompt-level "at most N words" target.
        if self.word_budget is not None:
            words = self._memory.split()
            if len(words) > self.word_budget:
                self._memory = " ".join(words[: self.word_budget])

        self.n_writes += 1

    def reset(self) -> None:
        self._memory = ""
        self.n_writes = 0
        self.last_input_tokens = self.last_output_tokens = 0

    def snapshot(self) -> str:
        return self._memory

    # ---- Maintenance event methods (called by MaintenanceEvent.apply) ----

    def maintenance_flush_history(self) -> None:
        """Drop the stored summary entirely — simulates a log-rotation /
        context-window flush event. Next read() returns empty string until
        the next write() rebuilds memory from the new session only."""
        self._memory = ""

    def maintenance_recompact(self, llm=None) -> None:
        """Re-apply the compaction prompt to the existing summary without
        any new content. Simulates a checkpoint / re-summarization event
        that can lose information by running through another lossy pass.

        Acts as a no-op when memory is empty or when no LLM is available.
        """
        if not self._memory or llm is None:
            return
        messages = [{"role": "user", "content": self.prompt_template.format(text=self._memory)}]
        if hasattr(llm, "chat_with_usage"):
            resp = llm.chat_with_usage(messages)
            self._memory = resp.text.strip()
        else:
            self._memory = llm.chat(messages).strip()

    def maintenance_partial_reset(self, ratio: float = 0.5) -> None:
        """Drop the trailing `ratio` fraction of the stored summary text.
        Approximates partial memory loss for an opaque single-summary store.
        """
        if not self._memory:
            return
        keep_chars = max(1, int(len(self._memory) * (1 - ratio)))
        self._memory = self._memory[:keep_chars]
