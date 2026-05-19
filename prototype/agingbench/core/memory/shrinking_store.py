"""
agingbench/baselines/memory/shrinking_store.py — Shrinking-budget compaction policy.

Like SummarizeStorePolicy but the word limit decreases each write cycle,
simulating growing context pressure. This produces a smooth, monotonically
increasing CVR curve because the amount of information that can be preserved
shrinks predictably over time.

The word budget follows a linear schedule:
    budget(t) = start_budget - t * decay_per_step

Once budget hits min_budget, it stays there.
"""

from typing import Optional
from .base import MemoryPolicy


class ShrinkingStorePolicy(MemoryPolicy):
    def __init__(
        self,
        prompt_template: str,
        start_budget: int = 400,
        min_budget: int = 120,
        decay_per_step: int = 20,
    ):
        self._memory: str = ""
        self.prompt_template = prompt_template
        self.start_budget = start_budget
        self.min_budget = min_budget
        self.decay_per_step = decay_per_step
        self.n_writes: int = 0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    @property
    def current_budget(self) -> int:
        budget = self.start_budget - self.n_writes * self.decay_per_step
        return max(budget, self.min_budget)

    def read(self, query: Optional[str] = None) -> str:
        return self._memory

    def write(self, new_content: str, llm=None) -> None:
        combined = (self._memory + "\n\n" + new_content).strip() if self._memory else new_content

        budget = self.current_budget

        if llm is not None:
            prompt = self.prompt_template.format(
                text=combined,
                word_limit=budget,
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

    def snapshot(self) -> str:
        return self._memory
