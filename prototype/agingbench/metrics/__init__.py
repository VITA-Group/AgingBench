"""
agingbench.metrics — Metric computation for all four evaluation groups.

Public API:
  aging   — AgingCurve dataclass + curve summary statistics (§4)
  g2      — CVR, tool_usage_shift, instruction_decay (G2)
  g3      — summarization_fidelity, memory_bloat, contradiction_rate,
            retrieval_precision, retrieval_recall (G3)
  g4      — FASR, RR, CFR, LA, shock, recovery (G4)
"""

from agingbench.metrics.aging import (  # noqa: F401
    AgingCurve,
    compute_decay_slope,
    compute_half_life,
    compute_hazard_proxy,
    load_curve_from_trace,
    summarize,
)
from agingbench.metrics.g2_metrics import (  # noqa: F401
    compute_cvr,
    compute_tool_usage_shift,
)
from agingbench.metrics.g3_metrics import (  # noqa: F401
    compute_contradiction_rate,
    compute_memory_bloat,
    compute_retrieval_precision,
    compute_retrieval_recall,
    compute_summarization_fidelity,
    score_session_g3,
)
from agingbench.metrics.g4_metrics import (  # noqa: F401
    compute_cfr,
    compute_fasr,
    compute_la,
    compute_recovery,
    compute_rr,
    compute_shock,
    score_life_event,
    score_session_g4,
)
from agingbench.metrics.aging_card import (  # noqa: F401
    AGING_CARD_SCHEMA_VERSION,
    build_aging_card,
    build_and_write_aging_card,
    write_aging_card,
)
