"""byo_memory_minimal.py — Bring-Your-Own-Memory template for Tier-1 scenarios.

AgingBench's Tier-1 scenarios (S1-S4, S6) decouple the model from the memory
backbone: you tell the SUT YAML which LLM provider to use, AND which
`MemoryPolicy` to use. The runner calls your policy's `write()` at the end of
each session (the U operator from §2.3 of the paper) and `read()` at the start
of the next (the R operator). The four built-in mechanisms — compression,
interference, revision, maintenance — emerge from how your policy chooses
*what to keep, what to drop, and how to retrieve*.

This file is a runnable, copy-paste-able template. It shows:

  1. The minimum MemoryPolicy contract (read + write + reset).
  2. How to point a SUT YAML at this class via `memory_policy: { type: custom, ... }`.
  3. A trivial RAG-style implementation you can replace with your own backbone
     (vector DB, knowledge graph, episodic store, learned compressor, ...).

For real policies, see:
  agingbench/core/memory/append_only.py        — vector retrieval + top-k
  agingbench/core/memory/summarize_store.py    — LLM-driven compaction
  agingbench/core/memory/lossy_episodic.py     — drop-old-entries baseline
  agingbench/core/memory/observer.py           — code-modification observer
  agingbench/core/memory/typed_state.py        — typed key-value store

To run S1 against this stub:

    pip install "git+https://github.com/VITA-Group/AgingBench.git@v0.3.0#subdirectory=prototype"
    cp examples/byo_memory_minimal.py /path/to/your_pkg/my_memory.py
    cp examples/sut_byo_memory.yaml \\
       agingbench/registry/suts/byo/sut_byo_memory.yaml   # see below
    agingbench run --scenario s1_research_literature \\
                   --sut agingbench/registry/suts/byo/sut_byo_memory.yaml \\
                   --output-dir runs/byo_mem_demo

Companion SUT YAML:

    sut_id: byo_memory_demo
    description: "Demo: minimal MemoryPolicy via type:custom"

    model:
      provider: litellm
      model: claude-haiku-4-5
      max_tokens: 2048

    memory_policy:
      type: custom
      class: your_pkg.my_memory:MyMemory       # importable on PYTHONPATH
      keep_last_k: 5                           # any extra keys forwarded as kwargs

    seed: 42

After the run completes, the canonical AgingCard is written to
`runs/byo_mem_demo/aging_card.json` — same schema as every other Tier-1 run.
"""
from __future__ import annotations

from typing import Optional

from agingbench.core.memory.base import MemoryPolicy


class MyMemory(MemoryPolicy):
    """Replace this stub with your real memory backbone.

    The runner contract:
      * read(query=...)  is called at the START of every session. Whatever you
                         return is injected verbatim into the agent's context.
      * write(new_content, llm=...) is called at the END of every session
                         with the agent's *output* (notes, decisions, etc.).
                         `llm` is the runner's LocalLLM instance — use it for
                         summarize/compress steps if your policy needs it.
      * reset() must restore a fresh empty state (between independent runs).
    """

    def __init__(self, keep_last_k: int = 5, **kwargs):
        self.keep_last_k = keep_last_k
        self._entries: list[str] = []

    # ---- required ---------------------------------------------------------

    def read(self, query: Optional[str] = None) -> str:
        """Retrieve what to inject into the next session's context.

        Trivial stub: return the last K entries joined by separators. A real
        implementation would do vector retrieval / structured query / typed
        lookup against `query` here.
        """
        if not self._entries:
            return ""
        recent = self._entries[-self.keep_last_k:]
        return "\n---\n".join(recent)

    def write(self, new_content: str, llm=None) -> None:
        """Update memory with the session's output.

        Trivial stub: append the raw content. A real policy might:
          - chunk + embed (RAG)
          - summarize with `llm` (compression)
          - extract typed facts into a KV store (typed state)
          - merge with prior entries (consolidation)
        """
        if new_content and new_content.strip():
            self._entries.append(new_content.strip())

    def reset(self) -> None:
        self._entries = []

    # ---- optional but recommended -----------------------------------------

    def snapshot(self) -> str:
        """Raw dump used by oracle ablations / debugging."""
        return "\n---\n".join(self._entries)

    def dump_store(self) -> str:
        """ALL stored content, bypassing read()'s top-k / ranking.

        Used by AgingBench's P2 (oracle retrieval) diagnostic to separate
        write losses from read losses. Override if read() does retrieval.
        """
        return self.snapshot()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Ranked retrieval, used by retrieval-quality metrics (G3-M4/M5).

        Trivial stub returns the last `top_k` entries with score 1.0. Real
        retrieval-based policies (vector / BM25 / hybrid) should return
        actual relevance scores so AgingBench can report precision/recall.
        """
        return [
            {"text": entry, "score": 1.0, "id": f"entry_{i}"}
            for i, entry in enumerate(self._entries[-top_k:])
        ]

    def entry_count(self) -> int:
        """For bloat-tracking; helps diagnose compression failures."""
        return len(self._entries)


# ---- sanity check (run as a script) --------------------------------------

if __name__ == "__main__":
    m = MyMemory(keep_last_k=2)
    assert m.read() == ""
    m.write("session 1: user is allergic to peanuts")
    m.write("session 2: meeting moved to 3pm")
    m.write("session 3: prefers metric units")
    # keep_last_k=2 → read returns sessions 2+3
    assert "session 1" not in m.read()
    assert "session 2" in m.read() and "session 3" in m.read()
    assert m.entry_count() == 3
    # but dump_store() bypasses top-k → all three visible
    assert "session 1" in m.dump_store()
    m.reset()
    assert m.entry_count() == 0
    print("OK — MemoryPolicy contract holds.")
