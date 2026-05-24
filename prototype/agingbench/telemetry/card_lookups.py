"""Static lookups for the Agent Lifespan Card.

Maps the four aging mechanisms to (a) memory-pipeline stages (W/R/U/S)
and (b) recommended repair recipes. Frozen for v1; verbatim from the
design memo at /ssd1/jianing/second_commend.md.

Stage taxonomy follows the AgingBench paper's memory pipeline:
  W  write       — what the agent commits to memory
  R  retrieval   — what the agent pulls back at probe time
  U  utilization — how the agent uses what it retrieved
  S  store       — lifecycle events (model swap, ctx flush, etc.)
"""

MECHANISM_TO_STAGE = {
    "compression":  "W",
    "interference": "R",
    "revision":     "U",
    "maintenance":  "S",
}

STAGE_LABELS = {
    "W": "write",
    "R": "retrieval",
    "U": "utilization",
    "S": "store",
}

MECHANISM_TO_REPAIR = {
    "compression":  "value-preserving compaction prompt + write-time keyword preservation",
    "interference": "retrieval discipline; force re-reads; tighter context budget",
    "revision":     "typed state for derived values (see paper §5.2 / app:typed-state)",
    "maintenance":  "regression checks after lifecycle events; flush-aware compaction",
}


def diagnostic_signature(mechanism: str) -> str:
    """Render the card's diagnostic-signature line.

    Example: diagnostic_signature("revision") -> "utilization-dominant (U-stage)"
    """
    stage = MECHANISM_TO_STAGE[mechanism]
    return f"{STAGE_LABELS[stage]}-dominant ({stage}-stage)"


def recommended_repair(mechanism: str) -> str:
    """Look up the recommended repair recipe for a dominant mechanism."""
    return MECHANISM_TO_REPAIR[mechanism]
