"""
inference/_selector.py — Dominant-mechanism arbitration.

Replaces the naive argmax over per-mechanism severity scores with
**independent-evidence gating + separation margin**. The naive argmax
overclaims because several signals (notably lineage continuity) are
compatible with multiple mechanisms; only mechanisms with an independent
prerequisite present should be eligible for dominant attribution.

Signal classification:

  | Signal                              | Independent for | Shared with                          |
  |-------------------------------------|-----------------|--------------------------------------|
  | saturation                          | compression     | —                                    |
  | arg_specificity_decline (P3)        | compression     | —                                    |
  | tool_kl_drift                       | interference    | —                                    |
  | anchor_drift (P2)                   | interference    | —                                    |
  | value_supersession (P1 / args-only) | revision        | —                                    |
  | lifecycle_event + pre/post delta    | maintenance     | —                                    |
  | lineage_continuity_drop (P4)        | (shared)        | compression, interference, revision  |

Rule:
  1. A mechanism gets credited *only* if at least one of its independent
     signals fires (`gate=True`). Shared signals (`lineage_continuity_drop`)
     add weight on top of independent evidence but cannot stand alone.
  2. After gating, apply argmax over surviving mechanisms. Report dominant
     only if its credited score is ≥ MARGIN × runner-up.
  3. If no mechanism passes the gate → return
     {"dominant": None, "reason": "no_independent_evidence",
      "compatible": [...]}.
  4. Multiple pass + separation below margin → return
     {"dominant": None, "reason": "co_dominant", "co_dominant": [a, b]}.
"""
from __future__ import annotations

from typing import Optional

from ._verdict import is_degrading

# Separation margin: dominant must be ≥ this × runner-up to be reported alone.
SEPARATION_MARGIN = 1.5

INDEPENDENT_SIGNALS = {
    "compression":  ["saturation", "arg_specificity"],
    "interference": ["tool_kl", "anchor_drift"],
    "revision":     ["value_supersession"],
    "maintenance":  ["lifecycle_event"],
}

SHARED_SIGNALS = ["lineage_continuity"]


def pick_dominant(trace_audit: dict) -> dict:
    """Pick the dominant mechanism from a trace_audit dict.

    Returns:
        {
            "dominant":  Optional[str],
            "reason":    str,                       # status: "argmax_with_margin"|"co_dominant"|"no_independent_evidence"|"no_signal"
            "scores":    dict[str, float],          # credited severity per surviving mechanism
            "evidence":  dict[str, list[str]],      # per-mechanism list of fired signal names
            "co_dominant": Optional[list[str]],     # populated when separation < margin
            "compatible": Optional[list[str]],      # populated when no gate passes
        }
    """
    evidence: dict[str, list[str]] = {m: [] for m in INDEPENDENT_SIGNALS}
    scores:   dict[str, float] = {}

    # 1. Score each mechanism by summing severities of fired signals.
    for mech in INDEPENDENT_SIGNALS:
        block = trace_audit.get(mech) or {}
        ind_fired, ind_severity = _independent_score(mech, block)
        evidence[mech].extend(ind_fired)

        # Shared signals (lineage) only add weight if independent fired.
        shared_severity = 0.0
        if ind_fired:
            shared_fired, shared_severity = _shared_score(block)
            evidence[mech].extend(shared_fired)

        total = ind_severity + shared_severity
        if total > 0 and ind_fired:
            scores[mech] = round(total, 4)

    # 2. Gate: surviving mechanisms with at least one independent signal.
    if not scores:
        # No mechanism passed the gate. Report any mechanisms with shared-only
        # evidence as "compatible" (the data is consistent with them but we
        # don't have independent attribution).
        compatible = [
            m for m in INDEPENDENT_SIGNALS
            if (trace_audit.get(m) or {}).get("lineage_continuity_verdict") == "rising_degradation"
            or (trace_audit.get(m) or {}).get("lineage_continuity_verdict") == "falling_degradation"
            or (trace_audit.get(m) or {}).get("lineage_continuity_verdict") == "floor_degradation"
        ]
        return {
            "dominant":   None,
            "reason":     "no_independent_evidence" if compatible else "no_signal",
            "scores":     {},
            "evidence":   evidence,
            "co_dominant": None,
            "compatible": compatible or None,
        }

    # 3. Apply argmax with separation margin.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_mech, top_score = ranked[0]
    if len(ranked) == 1 or ranked[1][1] == 0:
        return {
            "dominant":    top_mech,
            "reason":      "argmax_with_margin",
            "scores":      scores,
            "evidence":    evidence,
            "co_dominant": None,
            "compatible":  None,
        }

    runner_up_mech, runner_up_score = ranked[1]
    if top_score >= SEPARATION_MARGIN * runner_up_score:
        return {
            "dominant":    top_mech,
            "reason":      "argmax_with_margin",
            "scores":      scores,
            "evidence":    evidence,
            "co_dominant": None,
            "compatible":  None,
        }
    # Co-dominant: top and runner-up within the margin.
    return {
        "dominant":    None,
        "reason":      "co_dominant",
        "scores":      scores,
        "evidence":    evidence,
        "co_dominant": [top_mech, runner_up_mech],
        "compatible":  None,
    }


def _independent_score(mechanism: str, block: dict) -> tuple[list[str], float]:
    """Return (list of fired signal names, summed severity weight) for the
    independent signals of a mechanism. Severity uses verdict + slope.
    """
    fired: list[str] = []
    sev = 0.0

    if mechanism == "compression":
        if (block.get("saturation_session_rate") or 0.0) > 0.5:
            fired.append("saturation")
            sev += float(block.get("saturation_session_rate") or 0.0)
        v = block.get("tool_argument_specificity_verdict")
        if v and is_degrading(v):
            fired.append("arg_specificity")
            sev += 0.5
    elif mechanism == "interference":
        kl = block.get("tool_kl_mean_post_baseline")
        if kl is not None and kl > 0.05:
            fired.append("tool_kl")
            sev += min(1.0, float(kl))
        v = block.get("goal_anchor_drift_verdict")
        if v and is_degrading(v):
            fired.append("anchor_drift")
            sev += 0.5
    elif mechanism == "revision":
        n_stale = block.get("n_stale_propagations", 0) or 0
        if n_stale > 0:
            fired.append("value_supersession")
            sev += min(1.0, n_stale / 5.0)
    elif mechanism == "maintenance":
        deltas = block.get("m_maintenance_delta") or {}
        for shock_name, info in deltas.items() if isinstance(deltas, dict) else []:
            if isinstance(info, dict):
                d = info.get("delta")
                if d is not None and d < -0.05:
                    fired.append(f"lifecycle_event:{shock_name}")
                    sev += min(1.0, abs(d))
        # Also accept a top-level intervention rising verdict (older shape)
        v = block.get("intervention_rate_verdict")
        if v and is_degrading(v):
            fired.append("lifecycle_event:intervention_rate")
            sev += 0.5
    return fired, sev


def _shared_score(block: dict) -> tuple[list[str], float]:
    """Score from shared signals — only invoked when independent evidence
    is already present (so we don't credit a mechanism on a shared signal alone).
    """
    fired: list[str] = []
    sev = 0.0
    v = block.get("lineage_continuity_verdict")
    if v and is_degrading(v):
        fired.append("lineage_continuity")
        sev += 0.3
    return fired, sev
