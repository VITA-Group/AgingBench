"""Pressure-derived awareness blocks for scenario-aware system prompts.

Layer 2 framing: tells the agent about the *categories* of pressure it will
face (revisions, confusables, lossy memory), without leaking the *schedule* of
specific probes or lifecycle events. Builds a few sentences derived from the
active ``PressureConfig`` + memory_policy type so the same code works across
SUTs without per-SUT prompt tuning.

Reveal level by design:
  - Tell: category of pressure ("decisions may be revised")
  - Don't tell: specific schedule ("revision at session 7"), specific values,
    or probe IDs.
"""

from __future__ import annotations

from typing import Optional


def build_awareness_block(
    pressure=None,
    memory_policy_type: str = "",
    variant: str = "standard",
) -> str:
    """Build a Layer-2 awareness block from PressureConfig + memory policy type.

    ``variant``:
      - ``"standard"`` (default): includes all applicable awareness bullets.
      - ``"lean"``: drops the memory-policy bullet (compaction / partial-
        retrieval warnings) to test whether shorter awareness improves
        downstream metrics on small models.

    Returns an empty string when nothing relevant is active so the system
    prompt stays compact under low-pressure SUTs.
    """
    notes: list[str] = []

    n_pairs = int(getattr(pressure, "n_confusable_pairs", 0) or 0)
    similar = bool(getattr(pressure, "confusable_similar_names", False))
    high_sim = bool(getattr(pressure, "confusable_high_similarity", False))

    if n_pairs > 0 and similar:
        notes.append(
            "Multiple people or entities may have near-identical names "
            "(e.g., 'John Smith' vs 'John Smyth'). Match full names exactly; "
            "different spellings refer to different people."
        )
    elif n_pairs > 0 and high_sim:
        notes.append(
            "Similar-named entities with close numeric values may appear. "
            "Distinguish them by their qualifier (region, quarter, phase)."
        )
    elif n_pairs > 0:
        notes.append(
            "Similar-named entities may appear across domains. Distinguish "
            "them by their domain context."
        )

    update_rate = float(getattr(pressure, "update_rate", 0.0) or 0.0)
    if update_rate > 0:
        notes.append(
            "Decisions may be revised across sessions. If the same decision "
            "appears with different values in your memory, the LATEST one is "
            "authoritative; do not cite the earlier value."
        )

    forget_rate = float(getattr(pressure, "forget_rate", 0.0) or 0.0)
    if forget_rate > 0:
        notes.append(
            "Some prior decisions may be formally retracted in later meetings. "
            "If your memory contains an invalidation notice for a value, do "
            "NOT cite that value."
        )

    if variant != "lean":
        if memory_policy_type == "summarize_store":
            notes.append(
                "Your memory between sessions is periodically compacted. "
                "Specific values you record may be summarized away; when answering, "
                "quote critical numbers verbatim."
            )
        elif memory_policy_type == "append_only":
            # No compaction at write-time, but read-time top_k retrieval may still
            # hide stale facts. The category is the same for the agent.
            notes.append(
                "Your memory accumulates verbatim across sessions but may be "
                "retrieved partially. If you cannot find a specific value in the "
                "memory shown, it may exist outside the current retrieval window."
            )

    if not notes:
        return ""

    return "AWARENESS\n" + "\n".join(f"• {n}" for n in notes) + "\n\n"
