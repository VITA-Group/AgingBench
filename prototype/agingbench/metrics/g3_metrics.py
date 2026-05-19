"""
agingbench/metrics/g3_metrics.py — G3: Memory Quality Evaluation.

Metrics:
  summarization_fidelity(t) Fraction of gold source facts surviving in M_t.
  memory_bloat(t)           Token count of M_t at cycle t.
  contradiction_rate(t)     Fraction of M_t entries contradicting the gold timeline.
  retrieval_precision(t)    Precision of retrieved chunks vs. gold-relevant set.
  retrieval_recall(t)       Recall of retrieved chunks vs. gold-relevant set.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# summarization_fidelity — full-document coverage fraction
# ---------------------------------------------------------------------------

def compute_summarization_fidelity(
    memory_text: str,
    gold_facts: list[str],
    case_sensitive: bool = False,
) -> float:
    """
    Fraction of gold facts that can be found (substring match) in M_t.

    Parameters
    ----------
    memory_text : str
        The full content of M_t (the current memory store).
    gold_facts : list[str]
        Exhaustive list of atomic facts that should be present.
        Each fact is a short string (e.g., "budget: $50K", "signed by Lin Chen",
        "deadline: 2024-10-15"). A fact is considered surviving if it appears
        as a substring of memory_text.
    case_sensitive : bool
        Whether matching should be case-sensitive. Default: case-insensitive.

    Returns
    -------
    float  Fidelity score in [0, 1].  1.0 = all gold facts survive.
    """
    if not gold_facts:
        return 1.0

    if not case_sensitive:
        memory_text = memory_text.lower()

    hits = 0
    for fact in gold_facts:
        target = fact if case_sensitive else fact.lower()
        if target in memory_text:
            hits += 1

    return hits / len(gold_facts)


# ---------------------------------------------------------------------------
# memory_bloat — token count of M_t
# ---------------------------------------------------------------------------

def compute_memory_bloat(
    memory_text: str,
    tokenizer_fn: callable | None = None,
) -> int:
    """
    Token count of M_t at a given cycle.

    Parameters
    ----------
    memory_text : str
        The full content of M_t.
    tokenizer_fn : callable, optional
        A function str -> int that returns the token count. If None, falls
        back to whitespace splitting (approximate).

    Returns
    -------
    int  Token count of the memory store.
    """
    if tokenizer_fn is not None:
        return tokenizer_fn(memory_text)
    # fallback: approximate with whitespace split
    return len(memory_text.split())


# ---------------------------------------------------------------------------
# contradiction_rate — fraction of entries contradicting gold
# ---------------------------------------------------------------------------

def compute_contradiction_rate(
    memory_entries: list[dict],
    gold_timeline: list[dict],
    contradiction_checker: callable | None = None,
) -> float:
    """
    Fraction of M_t entries that contradict the gold timeline.

    Parameters
    ----------
    memory_entries : list[dict]
        Parsed entries from M_t. Each dict should have at least:
          - "fact_id" or "decision_id": str  (links to gold)
          - "content": str  (the stored claim)
    gold_timeline : list[dict]
        Ground-truth decisions. Each dict should have:
          - "fact_id" or "decision_id": str
          - "content": str  (canonical claim)
    contradiction_checker : callable, optional
        A function (memory_content: str, gold_content: str) -> bool that
        returns True if the two claims contradict each other. If None,
        uses a simple heuristic: entries that share a fact_id but have
        different content are flagged as contradictions.

    Returns
    -------
    float  Contradiction rate in [0, 1].
    """
    if not memory_entries:
        return 0.0

    gold_map = {}
    for g in gold_timeline:
        fid = g.get("fact_id") or g.get("decision_id", "")
        gold_map[fid] = g.get("content", "")

    contradictions = 0
    matched = 0

    for entry in memory_entries:
        fid = entry.get("fact_id") or entry.get("decision_id", "")
        if fid not in gold_map:
            continue
        matched += 1

        if contradiction_checker is not None:
            if contradiction_checker(entry.get("content", ""), gold_map[fid]):
                contradictions += 1
        else:
            # simple heuristic: content mismatch on matched fact_id
            mem_content = entry.get("content", "").strip().lower()
            gold_content = gold_map[fid].strip().lower()
            if mem_content and gold_content and mem_content != gold_content:
                contradictions += 1

    return contradictions / matched if matched > 0 else 0.0


# ---------------------------------------------------------------------------
# retrieval_precision / retrieval_recall
# ---------------------------------------------------------------------------

def compute_retrieval_precision(
    retrieved_ids: list[str],
    gold_relevant_ids: set[str],
) -> float:
    """
    Precision of retrieved chunks vs. gold-relevant set.

    Parameters
    ----------
    retrieved_ids : list[str]
        IDs of chunks returned by the retriever for a given query.
    gold_relevant_ids : set[str]
        IDs of chunks that are actually relevant (ground truth).

    Returns
    -------
    float  Precision in [0, 1].  |retrieved ∩ relevant| / |retrieved|.
    """
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in gold_relevant_ids)
    return hits / len(retrieved_ids)


def compute_retrieval_recall(
    retrieved_ids: list[str],
    gold_relevant_ids: set[str],
) -> float:
    """
    Recall of retrieved chunks vs. gold-relevant set.

    Parameters
    ----------
    retrieved_ids : list[str]
        IDs of chunks returned by the retriever for a given query.
    gold_relevant_ids : set[str]
        IDs of chunks that are actually relevant (ground truth).

    Returns
    -------
    float  Recall in [0, 1].  |retrieved ∩ relevant| / |relevant|.
    """
    if not gold_relevant_ids:
        return 1.0
    hits = sum(1 for rid in retrieved_ids if rid in gold_relevant_ids)
    return hits / len(gold_relevant_ids)


# ---------------------------------------------------------------------------
# Convenience: compute all G3 metrics for one session
# ---------------------------------------------------------------------------

def score_session_g3(
    memory_text: str,
    gold_facts: list[str],
    memory_entries: list[dict] | None = None,
    gold_timeline: list[dict] | None = None,
    retrieved_ids: list[str] | None = None,
    gold_relevant_ids: set[str] | None = None,
    tokenizer_fn: callable | None = None,
    contradiction_checker: callable | None = None,
) -> dict[str, float | int]:
    """
    Compute all applicable G3 metrics for one session checkpoint.

    Returns a dict with keys matching the metric spec table IDs.
    Metrics are omitted if the required inputs are not provided.
    """
    result: dict[str, float | int] = {}

    result["summarization_fidelity"] = compute_summarization_fidelity(
        memory_text, gold_facts
    )
    result["memory_bloat"] = compute_memory_bloat(memory_text, tokenizer_fn)

    if memory_entries is not None and gold_timeline is not None:
        result["contradiction_rate"] = compute_contradiction_rate(
            memory_entries, gold_timeline, contradiction_checker
        )

    if retrieved_ids is not None and gold_relevant_ids is not None:
        result["retrieval_precision"] = compute_retrieval_precision(
            retrieved_ids, gold_relevant_ids
        )
        result["retrieval_recall"] = compute_retrieval_recall(
            retrieved_ids, gold_relevant_ids
        )

    return result
