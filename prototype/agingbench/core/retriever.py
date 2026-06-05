"""
agingbench/core/retriever.py — SQLite-backed retriever with cosine similarity.

Used by append_only and (in P3) for KB-growth retrieval. For P2, only
get_by_id / overwrite are used; cosine_search is exercised in P3.
"""

import time
import sqlite3
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseRetriever(ABC):
    """
    Abstract retriever interface for memory stores.

    Implementations provide ranked retrieval over a growing set of text entries.
    This enables swapping retriever backends (SQLite, FAISS, Chroma, etc.)
    without changing the memory policy or scenario logic.
    """

    @abstractmethod
    def add_entry(self, entry_id: str, text: str, cycle: int = 0) -> None:
        """Add a text entry to the store."""

    @abstractmethod
    def cosine_search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """Return top-k (score, text) by relevance. Score in [0, 1]."""

    @abstractmethod
    def get_all_texts(self) -> list[str]:
        """Return all stored texts in insertion order."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of entries."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all entries."""


class SQLiteRetriever(BaseRetriever):
    """
    Persistent retriever backed by SQLite.

    Table: memory_store(entry_id TEXT PK, cycle INT, text TEXT, embedding BLOB, created_at REAL)

    For P2:  write/overwrite/get_by_id (embedding column left NULL)
    For P3:  add_entry with embedding; cosine_search
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS memory_store (
        entry_id   TEXT PRIMARY KEY,
        cycle      INTEGER,
        text       TEXT NOT NULL,
        embedding  BLOB,
        created_at REAL
    );
    """

    def __init__(self, db_path: str = ":memory:", embedding_model: Optional[str] = None):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(self.SCHEMA)
        self.conn.commit()
        self._encoder = None
        self._embedding_model = embedding_model

    def _get_encoder(self):
        if self._encoder is None and self._embedding_model:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self._embedding_model)
        return self._encoder

    # ------------------------------------------------------------------ writes

    def overwrite(self, entry_id: str, text: str, cycle: int = 0) -> None:
        """Atomic upsert — always exactly one row per entry_id."""
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_store (entry_id, cycle, text, embedding, created_at) "
            "VALUES (?, ?, ?, NULL, ?)",
            (entry_id, cycle, text, time.time()),
        )
        self.conn.commit()

    def add_entry(self, entry_id: str, text: str, cycle: int = 0) -> None:
        """Insert with embedding (used in P3 / append_only). Computes embedding if encoder set."""
        emb_bytes = None
        enc = self._get_encoder()
        if enc is not None:
            emb = enc.encode(text, normalize_embeddings=True).astype(np.float32)
            emb_bytes = emb.tobytes()
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_store (entry_id, cycle, text, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry_id, cycle, text, emb_bytes, time.time()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ reads

    def get_by_id(self, entry_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT text FROM memory_store WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        return row[0] if row else None

    def get_all_texts(self) -> list[str]:
        rows = self.conn.execute("SELECT text FROM memory_store ORDER BY created_at").fetchall()
        return [r[0] for r in rows]

    def cosine_search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """Return top-k (score, text) by cosine similarity. Requires embedding_model."""
        enc = self._get_encoder()
        if enc is None:
            raise RuntimeError("cosine_search requires embedding_model to be set")
        q_emb = enc.encode(query, normalize_embeddings=True).astype(np.float32)
        rows = self.conn.execute(
            "SELECT text, embedding FROM memory_store WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return []
        scored = []
        for text, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            score = float(np.dot(q_emb, emb))
            scored.append((score, text))
        scored.sort(reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memory_store").fetchone()[0]

    def reset(self) -> None:
        self.conn.execute("DELETE FROM memory_store")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
