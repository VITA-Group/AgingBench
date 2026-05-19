"""
agingbench/baselines/memory/chain_growing.py — Chain compress with growing input.

Combines chain compression (no re-reading original profile) with growing
interaction history, producing a true decay curve:

  M_0 = compress(profile, budget=B)              → fits well, CVR ≈ 0
  M_1 = compress(M_0 + H_0 + H_1, budget=B)     → slightly more input
  M_2 = compress(M_1 + H_0 + H_1 + H_2, budget=B) → growing pressure
  ...

Key insight: the compressed memory M_t never re-reads the original profile,
so lost info is permanent. But the RAW interaction history grows each session,
creating increasing pressure. As more history competes for the fixed budget,
constraints get progressively squeezed out.

Start with a generous budget (e.g., 400 words) so the initial profile fits,
then let history growth create the pressure.
"""

from typing import Optional
from .base import MemoryPolicy


class ChainGrowingPolicy(MemoryPolicy):
    def __init__(
        self,
        prompt_template: str,
        word_budget: int = 400,
    ):
        self._memory: str = ""
        self._raw_history: list[str] = []
        self.prompt_template = prompt_template
        self.word_budget = word_budget
        self.n_writes: int = 0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_input_words: int = 0

    def read(self, query: Optional[str] = None) -> str:
        return self._memory

    def write(self, new_content: str, llm=None) -> None:
        if self.n_writes > 0:
            self._raw_history.append(new_content)

        # Build input: previous compressed memory + ALL raw history
        # This creates growing pressure while preventing profile re-read
        if self._memory:
            combined = self._memory + "\n\n" + "\n\n".join(self._raw_history)
        else:
            # First write: just the profile
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
        self._raw_history = []
        self.n_writes = 0
        self.last_input_tokens = self.last_output_tokens = 0
        self.last_input_words = 0

    def snapshot(self) -> str:
        return self._memory
