"""
agingbench/core/memory/lossy_episodic.py — Pure R-lesion baseline.

Semantics: lossy W per session + full R at read time.

Each write() compresses the current session's content into a lossy per-session
summary using the configured compaction prompt, then APPENDS it to a list.
read() returns ALL per-session summaries concatenated, with no filtering or
ranking. This isolates the retrieval component from the write component:

  - summarize_store (lossy):   single rolling summary, re-compressed each write
                               (W lossy + R trivial / single-chunk)
  - growing_history (lossy):   compresses accumulated raw history to a
                               word_budget cap each write (W lossy on full log)
  - append_only:               verbatim chunks + top-k cosine retrieval
                               (W lossless + R filtered)
  - lossy_episodic (this):     per-session lossy chunks + full read
                               (W lossy + R idealized)

Comparing summarize_store-lossy vs lossy_episodic with the same compaction
prompt isolates the effect of rolling re-compression vs preserving discrete
per-session lossy artifacts — i.e. the R-side "filtering quality" degree of
freedom, holding W-side per-session aggressiveness constant.
"""

from typing import Optional
from .base import MemoryPolicy


class LossyEpisodicPolicy(MemoryPolicy):
    """Lossy per-session compression; read returns all summaries joined."""

    def __init__(self, prompt_template: str):
        self._summaries: list[str] = []
        self.prompt_template = prompt_template
        self.n_writes: int = 0
        # For tracing parity with summarize_store / growing_history.
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    def read(self, query: Optional[str] = None) -> str:
        if not self._summaries:
            return ""
        return "\n\n---\n\n".join(self._summaries)

    def write(self, new_content: str, llm=None) -> None:
        if llm is not None:
            prompt = self.prompt_template.format(text=new_content)
            messages = [{"role": "user", "content": prompt}]
            if hasattr(llm, "chat_with_usage"):
                resp = llm.chat_with_usage(messages)
                summary = resp.text.strip()
                self.last_input_tokens = resp.input_tokens
                self.last_output_tokens = resp.output_tokens
            else:
                summary = llm.chat(messages).strip()
                self.last_input_tokens = self.last_output_tokens = 0
        else:
            summary = new_content
            self.last_input_tokens = self.last_output_tokens = 0

        self._summaries.append(summary)
        self.n_writes += 1

    def reset(self) -> None:
        self._summaries = []
        self.n_writes = 0
        self.last_input_tokens = self.last_output_tokens = 0

    def snapshot(self) -> str:
        return self.read()

    def entry_count(self) -> int:
        return len(self._summaries)
