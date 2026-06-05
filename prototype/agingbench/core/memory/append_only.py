"""
agingbench/core/memory/append_only.py — Naive append-only episodic store.

Each session's output is appended as a new entry. On read, the top-k most
relevant entries are returned via cosine similarity (or the last-k if no
embedding model is configured).

Primary failure driver for S4 (stale cached tool call patterns).
"""

from typing import Optional
from .base import MemoryPolicy
from ..retriever import SQLiteRetriever


class AppendOnlyPolicy(MemoryPolicy):
    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_model: Optional[str] = "all-MiniLM-L6-v2",
        top_k: int = 5,
        max_input_tokens: int = 200_000,
    ):
        self.retriever = SQLiteRetriever(db_path=db_path, embedding_model=embedding_model)
        self.top_k = top_k
        # Truncation cap: keep the most recent ~max_input_tokens worth of memory.
        # Set close to model context limit but leave room for system+task+output.
        self.max_input_tokens = max_input_tokens
        self._counter = 0

    @staticmethod
    def _approx_tokens(text: str) -> int:
        # Rough heuristic: 1 token ≈ 4 chars (English). Good enough for budget guard.
        return len(text) // 4

    def _truncate_to_budget(self, text: str) -> str:
        n_tok = self._approx_tokens(text)
        if n_tok <= self.max_input_tokens:
            return text
        # Keep the tail (most recent content); prefix with marker
        chars_to_keep = self.max_input_tokens * 4
        return "[...older entries truncated to fit context budget...]\n" + text[-chars_to_keep:]

    def read(self, query: Optional[str] = None) -> str:
        all_texts = self.retriever.get_all_texts()
        if not all_texts:
            return ""
        if query and self.retriever._embedding_model:
            results = self.retriever.cosine_search(query, top_k=self.top_k)
            joined = "\n---\n".join(text for _, text in results)
        else:
            # fallback: return last top_k entries
            joined = "\n---\n".join(all_texts[-self.top_k :])
        return self._truncate_to_budget(joined)

    def write(self, new_content: str, llm=None) -> None:
        self._counter += 1
        self.retriever.add_entry(
            entry_id=f"entry_{self._counter:04d}",
            text=new_content,
            cycle=self._counter,
        )

    def reset(self) -> None:
        self.retriever.reset()
        self._counter = 0

    def snapshot(self) -> str:
        return "\n---\n".join(self.retriever.get_all_texts())

    def dump_store(self) -> str:
        """Return ALL stored entries, bypassing cosine top-k retrieval."""
        all_texts = self.retriever.get_all_texts()
        return "\n---\n".join(all_texts) if all_texts else ""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Ranked retrieval via cosine similarity over stored entries."""
        if self.retriever._embedding_model:
            results = self.retriever.cosine_search(query, top_k=top_k)
            return [
                {"text": text, "score": score, "id": f"cosine_{i}"}
                for i, (score, text) in enumerate(results)
            ]
        # Fallback: return all entries (no ranking)
        return [
            {"text": t, "score": 1.0, "id": f"entry_{i}"}
            for i, t in enumerate(self.retriever.get_all_texts()[-top_k:])
        ]

    def entry_count(self) -> int:
        return self.retriever.count()

    # ---- Maintenance event methods (duck-typed, called by MaintenanceEvent.apply) ----

    def maintenance_partial_reset(self, ratio: float = 0.5) -> None:
        """Remove oldest `ratio` fraction of entries, keeping the most recent."""
        all_texts = self.retriever.get_all_texts()
        n_keep = max(1, int(len(all_texts) * (1 - ratio)))
        texts_to_keep = all_texts[-n_keep:]
        self.retriever.reset()
        self._counter = 0
        for text in texts_to_keep:
            self._counter += 1
            self.retriever.add_entry(f"entry_{self._counter:04d}", text, self._counter)

    def maintenance_flush_history(self) -> None:
        """Drop every stored entry — equivalent to a complete log rotation.
        Equivalent to `maintenance_partial_reset(ratio=1.0)` but stated
        explicitly so SUT YAMLs can refer to the canonical shock name."""
        self.retriever.reset()
        self._counter = 0

    def maintenance_recompact(self, llm=None) -> None:
        """Merge all stored entries into a single most-recent entry (bounded
        by top_k) — simulates a checkpoint / re-indexing event. Does not
        use an LLM; just rewrites the store so later reads return the
        concatenated recent context in one shot instead of across many
        shards. This is the append_only analogue of summarize_store's
        recompact, minus the LLM-based compression step."""
        all_texts = self.retriever.get_all_texts()
        if not all_texts:
            return
        merged = "\n---\n".join(all_texts[-self.top_k:])
        merged = self._truncate_to_budget(merged)
        self.retriever.reset()
        self._counter = 1
        self.retriever.add_entry("entry_0001", merged, 1)
