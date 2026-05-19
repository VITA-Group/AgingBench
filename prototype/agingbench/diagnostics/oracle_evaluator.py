"""
agingbench/diagnostics/oracle_evaluator.py — P2 and P3 single-call evaluators.

P2 (Oracle Retrieval): LLM receives the FULL contents of the agent's actual
    memory store — everything that survived W — plus the probe question.
    Bypasses the agent's retrieval algorithm R entirely.

P3 (Oracle Context): LLM receives ground-truth facts directly — perfect
    context.  Establishes the absolute reasoning ceiling of the LLM (U).

Both are single LLM calls (no agent loop, no tools, no multi-turn ReAct).
"""

from __future__ import annotations

from ..core.agent import strip_thinking


def evaluate_p2(
    llm,
    probe_question: str,
    store_text: str,
) -> str:
    """P2: Oracle Retrieval from the agent's actual memory store.

    The LLM receives the FULL contents of the agent's memory store
    (everything that physically survived the write process W) plus the
    probe question.  This bypasses the agent's retrieval algorithm R
    entirely.

    Parameters
    ----------
    llm : BaseLLM
        The same LLM used by the agent.
    probe_question : str
        The recall probe question.
    store_text : str
        Full dump of the agent's memory store (from ``policy.dump_store()``).

    Returns
    -------
    str — the LLM's answer.
    """
    if not store_text or not store_text.strip():
        store_block = "(empty — no information stored)"
    else:
        store_block = store_text

    prompt = (
        "You are answering a factual question using information from your memory.\n\n"
        "=== YOUR MEMORY (complete contents) ===\n"
        f"{store_block}\n"
        "=== END MEMORY ===\n\n"
        f"Question: {probe_question}\n\n"
        "Answer concisely based ONLY on the information in your memory above. "
        "If the information is not in your memory, say 'I don't have that information.'"
    )
    response = llm.chat([{"role": "user", "content": prompt}])
    return strip_thinking(response, llm)


def evaluate_p3(
    llm,
    probe_question: str,
    gold_facts: str,
) -> str:
    """P3: Oracle Context (ground truth injected directly).

    The LLM receives the ground-truth facts directly — perfect context.
    This establishes the absolute reasoning ceiling of the LLM (U).

    Parameters
    ----------
    llm : BaseLLM
        The same LLM used by the agent.
    probe_question : str
        The recall probe question.
    gold_facts : str
        Ground-truth facts relevant to the probe.

    Returns
    -------
    str — the LLM's answer.
    """
    prompt = (
        "You are answering a factual question. Here are the relevant facts:\n\n"
        "=== FACTS ===\n"
        f"{gold_facts}\n"
        "=== END FACTS ===\n\n"
        f"Question: {probe_question}\n\n"
        "Answer concisely based on the facts above."
    )
    response = llm.chat([{"role": "user", "content": prompt}])
    return strip_thinking(response, llm)
