"""
agingbench/core/memory/growing_history.py — Growing history compaction.

The most realistic memory aging policy: maintains a raw log of ALL past
interactions. Each write appends to the raw log, then compresses the
ENTIRE log (profile + all history) into a fixed word budget.

Session 0:  compress(profile + history_0)           ≈ 1,300 tok → 300 words
Session 5:  compress(profile + history_0..5)         ≈ 4,000 tok → 300 words
Session 10: compress(profile + history_0..10)        ≈ 7,000 tok → 300 words
Session 14: compress(profile + history_0..14)        ≈ 10,000 tok → 300 words

The compression ratio grows from ~4:1 to ~33:1, naturally producing smooth
decay as more interaction history competes with constraint information for
the fixed budget.
"""

from typing import Optional
from .base import MemoryPolicy


class GrowingHistoryStorePolicy(MemoryPolicy):
    def __init__(
        self,
        prompt_template: str,
        word_budget: int = 300,
    ):
        self._compressed: str = ""       # The compressed memory M_t
        self._raw_history: list[str] = []  # Full raw interaction log
        self._initial_profile: str = ""    # Original profile text
        self.prompt_template = prompt_template
        self.word_budget = word_budget
        self.n_writes: int = 0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_raw_size: int = 0       # For tracing: raw input size

    def read(self, query: Optional[str] = None) -> str:
        return self._compressed

    def write(self, new_content: str, llm=None) -> None:
        # First write: new_content is the profile text
        if self.n_writes == 0:
            self._initial_profile = new_content

        # Append the new content as raw history (skip the first write which is just profile init)
        if self.n_writes > 0:
            self._raw_history.append(new_content)

        # Build full document: profile + ALL raw history
        full_document = self._initial_profile
        if self._raw_history:
            full_document += "\n\n" + "\n\n".join(self._raw_history)

        self.last_raw_size = len(full_document.split())

        if llm is not None:
            prompt = self.prompt_template.format(
                text=full_document,
                word_limit=self.word_budget,
            )
            messages = [{"role": "user", "content": prompt}]
            if hasattr(llm, "chat_with_usage"):
                resp = llm.chat_with_usage(messages)
                self._compressed = resp.text.strip()
                self.last_input_tokens = resp.input_tokens
                self.last_output_tokens = resp.output_tokens
            else:
                self._compressed = llm.chat(messages).strip()
                self.last_input_tokens = self.last_output_tokens = 0
        else:
            # No LLM: just truncate (for unit tests)
            self._compressed = full_document
            self.last_input_tokens = self.last_output_tokens = 0

        self.n_writes += 1

    def reset(self) -> None:
        self._compressed = ""
        self._raw_history = []
        self._initial_profile = ""
        self.n_writes = 0
        self.last_input_tokens = self.last_output_tokens = 0
        self.last_raw_size = 0

    def snapshot(self) -> str:
        return self._compressed

    # ---- Maintenance event methods (duck-typed, called by MaintenanceEvent.apply) ----

    def maintenance_partial_reset(self, ratio: float = 0.5) -> None:
        """Drop oldest `ratio` fraction of raw history entries."""
        n_drop = int(len(self._raw_history) * ratio)
        if n_drop > 0:
            self._raw_history = self._raw_history[n_drop:]

    def maintenance_reindex(self) -> None:
        """Shuffle the order of raw history entries (disrupts temporal ordering)."""
        import random
        random.shuffle(self._raw_history)

    def maintenance_reduce_budget(self, new_budget: int = 150) -> None:
        """Permanently reduce the word budget for compaction."""
        self.word_budget = min(self.word_budget, new_budget)

    def maintenance_memory_compaction(self, truncate_to: int = 500) -> None:
        """Force-truncate compressed memory to a character limit."""
        if len(self._compressed) > truncate_to:
            self._compressed = self._compressed[:truncate_to] + " [COMPACTED]"

    def maintenance_flush_history(self) -> None:
        """Clear the raw history log (simulates log rotation / storage cleanup).

        The compressed summary survives, but future write() calls can only
        build from profile + new session content — all prior sessions are
        gone from the compression input. This is the most realistic
        maintenance shock: raw logs get cleaned up, summary is the only
        surviving record.
        """
        self._raw_history = []

    def maintenance_recompact(self) -> None:
        """Replace raw history with the current compressed text, then on the
        next write() the system re-compresses from this checkpoint.

        Simulates a system that periodically checkpoints by replacing raw
        logs with the compressed summary. The next compression pass works
        from an already-lossy input, causing double-compression information
        loss — facts that survived the first pass may not survive the second.
        """
        if self._compressed:
            self._raw_history = [self._compressed]
