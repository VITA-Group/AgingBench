"""
agingbench/baselines/memory/chain_compress.py — Telephone-game compaction.

Each write compresses (previous_compressed_memory + full_new_history) into
a fixed word budget. Critically, the original profile is NEVER re-read —
only the compressed output from the previous step is available.

This creates compounding errors:
  M_0 = compress(profile)                    # some detail lost
  M_1 = compress(M_0 + history_1)            # M_0 already lossy → more lost
  M_2 = compress(M_1 + history_2)            # errors compound
  ...
  M_14 = compress(M_13 + history_14)         # 14 rounds of re-summarization

If a constraint is lost at step t, it can NEVER be recovered at step t+1
because the original profile is not re-injected. Each re-summarization also
gradually paraphrases precise values (dollar amounts, times, names),
introducing drift even for "preserved" constraints.

This models real systems where memory is updated in-place (like ChatGPT
Memory or Claude's MEMORY.md) rather than re-derived from raw logs.
"""

from typing import Optional
from .base import MemoryPolicy


class ChainCompressPolicy(MemoryPolicy):
    def __init__(
        self,
        prompt_template: str,
        word_budget: int = 200,
    ):
        self._memory: str = ""
        self.prompt_template = prompt_template
        self.word_budget = word_budget
        self.n_writes: int = 0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_input_words: int = 0

    def read(self, query: Optional[str] = None) -> str:
        return self._memory

    def write(self, new_content: str, llm=None) -> None:
        # Chain: previous compressed memory + new content
        if self._memory:
            combined = self._memory + "\n\n" + new_content
        else:
            combined = new_content

        self.last_input_words = len(combined.split())

        if llm is not None:
            prompt = self.prompt_template.format(
                text=combined,
                word_limit=self.word_budget,
            )
            messages = [{"role": "user", "content": prompt}]
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

        self.n_writes += 1

    def reset(self) -> None:
        self._memory = ""
        self.n_writes = 0
        self.last_input_tokens = self.last_output_tokens = 0
        self.last_input_words = 0

    def snapshot(self) -> str:
        return self._memory
