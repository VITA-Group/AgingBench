"""
Render the Agent Lifespan Card from a `trace_audit` dict.

Produces an ASCII card matching atlas's design (headline, trace regime
disclosure, per-mechanism evidence with a 5-star strength meter,
dominant-mechanism arbitration, signature, repair, share actions).
Pure formatting; no inference.
"""
from __future__ import annotations

from typing import Optional

from .inference._verdict import is_degrading


# Star characters — uses the BLACK STAR (★) and WHITE STAR (☆) glyphs so
# the meter renders uniformly across mechanisms regardless of strength.
_FILLED = "★"
_EMPTY  = "☆"
_MAX_STARS = 5


# ─── Top-level renderer ─────────────────────────────────────────────────────

def render_card_ascii(
    trace_audit: dict,
    *,
    title_suffix: Optional[str] = None,
    width: int = 60,
) -> str:
    """Format a trace_audit dict as the multi-line Lifespan Card.

    Parameters
    ----------
    trace_audit : the full `card.trace_audit` dict produced by
                  `trace_to_card_v11`.
    title_suffix : optional string appended to the header
                   (e.g. "(real Claude Code trace)").
    width : separator-line width.
    """
    lines: list[str] = []

    # ── Header
    header = "🕰️  Agent Lifespan Report"
    if title_suffix:
        header += f" {title_suffix}"
    lines.append(header)
    lines.append("─" * width)

    # ── Headline
    hb = trace_audit.get("headline") or {}
    headline_label = hb.get("label") or "Aging not measurable on this trace"
    lines.append(headline_label)

    # ── Trace regime disclosure
    regime = trace_audit.get("trace_regime") or {}
    if regime:
        parts = []
        if regime.get("tool_using") is not None:
            parts.append("tool-using" if regime["tool_using"] else "chat-only")
        if regime.get("n_sessions") is not None:
            parts.append(f"{regime['n_sessions']} sessions")
        if regime.get("outcomes"):
            parts.append(f"outcomes:{regime['outcomes']}")
        if regime.get("adapter"):
            parts.append(f"adapter:{regime['adapter']}")
        lines.append(f"Trace regime:          {' · '.join(parts)}")

    lines.append("")

    # ── Per-mechanism evidence
    lines.append("Mechanism evidence:")
    dominant = (trace_audit.get("dominant_mechanism") or {}).get("dominant")
    co_dominant = (trace_audit.get("dominant_mechanism") or {}).get("co_dominant") or []

    for mech in ("compression", "interference", "revision", "maintenance"):
        block = trace_audit.get(mech) or {}
        n_stars = _mechanism_strength(mech, block, dominant, co_dominant)
        stars = _render_stars(n_stars)
        summary, extra = _mechanism_evidence(mech, block)
        # Fixed-width column layout: 2 indent + 14 name + 5 stars + 2 gap = 23
        prefix = f"  {mech:<14}{stars}"
        lines.append(f"{prefix}  {summary}")
        # Detail line (trajectory) — render only for dominant to avoid clutter
        if extra and mech == dominant:
            lines.append(f"  {'':<14}{' ' * _MAX_STARS}  {extra}")

    lines.append("")

    # ── Dominant mechanism arbitration result
    dm = trace_audit.get("dominant_mechanism") or {}
    dm_line = _dominant_line(dm)
    if dm_line:
        lines.append(dm_line)

    # ── Signature + repair (only when a single mechanism dominates)
    sig = trace_audit.get("signature")
    rep = trace_audit.get("repair")
    if sig:
        lines.append(f"Diagnostic signature:  {sig}")
    if rep:
        lines.append(f"Recommended repair:    {rep}")

    lines.append("")
    lines.append("[ Share on X ]   [ Open AgingCard JSON ]")

    return "\n".join(lines)


# ─── Per-mechanism strength meter (0-5 stars) ───────────────────────────────

def _render_stars(n: int) -> str:
    """Render n filled stars out of _MAX_STARS, padded with empty stars."""
    n = max(0, min(_MAX_STARS, n))
    return _FILLED * n + _EMPTY * (_MAX_STARS - n)


def _mechanism_strength(
    mechanism: str,
    block: dict,
    dominant: Optional[str],
    co_dominant: list[str],
) -> int:
    """Map a mechanism block's evidence to 0..5 stars.

    Driven by the mechanism's own independent signals (per-block evidence),
    NOT the selector's gated score — so each mechanism reports its own
    strength even when another wins the dominance argmax.

    Dominance gives a minimum floor (so the headline winner never looks
    weaker than its runners-up in the meter), but does not artificially
    inflate beyond what the evidence shows.
    """
    base = _per_mechanism_severity(mechanism, block)
    if mechanism == dominant:
        base = max(base, 4.0)
    elif mechanism in (co_dominant or []):
        base = max(base, 3.0)
    return int(round(max(0.0, min(5.0, base))))


def _per_mechanism_severity(mechanism: str, block: dict) -> float:
    """Per-mechanism heuristic severity in [0, 5]. Higher = stronger evidence."""
    if mechanism == "compression":
        return _compression_severity(block)
    if mechanism == "interference":
        return _interference_severity(block)
    if mechanism == "revision":
        return _revision_severity(block)
    if mechanism == "maintenance":
        return _maintenance_severity(block)
    if mechanism == "consistency":
        return _consistency_severity(block)
    return 0.0


def _consistency_severity(block: dict) -> float:
    """Consistency block isn't a mechanism but it carries the aging-happened
    signal. Score by behavior_drift_at_repeat magnitude so the 5th sparkline
    on the website gets a meaningful strength meter rather than always 0.
    """
    drift = float(block.get("behavior_drift_at_repeat") or 0.0)
    if drift > 0.5:
        return 5.0
    if drift > 0.3:
        return 4.0
    if drift > 0.15:
        return 3.0
    if drift > 0.05:
        return 2.0
    if drift > 0.0:
        return 1.0
    return 0.0


def _compression_severity(block: dict) -> float:
    score = 0.0
    sat = block.get("saturation_session_rate") or 0.0
    if sat > 0.7:
        score += 2.5
    elif sat > 0.3:
        score += 1.5
    elif sat > 0.05:
        score += 0.5
    cn_v = block.get("context_noise_verdict") or ""
    if is_degrading(cn_v):
        score += 1.0
    spec_v = block.get("tool_argument_specificity_verdict") or ""
    if is_degrading(spec_v):
        score += 1.5
    return score


def _interference_severity(block: dict) -> float:
    score = 0.0
    kl = block.get("tool_kl_mean_post_baseline") or 0.0
    if kl > 0.2:
        score += 2.5
    elif kl > 0.1:
        score += 1.5
    elif kl > 0.05:
        score += 0.5
    anchor_v = block.get("goal_anchor_drift_verdict") or ""
    if is_degrading(anchor_v):
        score += 1.0
    lineage_v = block.get("lineage_continuity_verdict") or ""
    if is_degrading(lineage_v):
        score += 1.0
    return score


def _revision_severity(block: dict) -> float:
    score = 0.0
    n_stale = block.get("n_stale_propagations") or 0
    if n_stale > 50:
        score += 3.0
    elif n_stale > 20:
        score += 2.5
    elif n_stale > 5:
        score += 1.5
    elif n_stale > 0:
        score += 0.5
    v = (block.get("value_supersession_verdict")
         or block.get("violation_trajectory_verdict")
         or "")
    if is_degrading(v):
        score += 1.0
    n_ent = block.get("n_entities_tracked") or 0
    if n_ent >= 5:
        score += 0.5
    return score


def _maintenance_severity(block: dict) -> float:
    score = 0.0
    delta = block.get("median_outcome_rate_delta")
    if delta is not None:
        if delta < -0.15:
            score += 3.0
        elif delta < -0.05:
            score += 1.5
        elif delta < -0.01:
            score += 0.5
    intervention_v = block.get("intervention_rate_verdict") or ""
    if is_degrading(intervention_v):
        score += 1.5
    return score


def _mechanism_evidence(mechanism: str, block: dict) -> tuple[str, Optional[str]]:
    """Return (summary, optional extra detail line) for a mechanism block."""
    if mechanism == "compression":
        return _compression_evidence(block), None
    if mechanism == "interference":
        return _interference_evidence(block), None
    if mechanism == "revision":
        return _revision_evidence(block)
    if mechanism == "maintenance":
        return _maintenance_evidence(block), None
    return "", None


def _compression_evidence(block: dict) -> str:
    parts: list[str] = []
    sat_rate = block.get("saturation_session_rate")
    if sat_rate is not None and sat_rate > 0.5:
        parts.append(f"saturation {sat_rate:.2f}")
    cn_verdict = block.get("context_noise_verdict")
    if cn_verdict and cn_verdict not in ("flat", "no_signal"):
        parts.append(f"context_noise {cn_verdict.replace('_', ' ')}")
    spec_verdict = block.get("tool_argument_specificity_verdict")
    if spec_verdict and spec_verdict not in ("flat", "no_signal"):
        parts.append(f"arg_specificity {spec_verdict.replace('_', ' ')}")
    return "; ".join(parts) if parts else "no compression signal"


def _interference_evidence(block: dict) -> str:
    parts: list[str] = []
    kl = block.get("tool_kl_mean_post_baseline")
    if kl is not None:
        parts.append(f"KL={kl:.2f}")
    anchor_verdict = block.get("goal_anchor_drift_verdict")
    if anchor_verdict and anchor_verdict not in ("flat", "no_signal"):
        parts.append(f"anchor drift {anchor_verdict.replace('_', ' ').replace('degradation', '').strip()}")
    n_tools = block.get("n_distinct_tools")
    if n_tools:
        parts.append(f"n_distinct_tools={n_tools}")
    lineage_verdict = block.get("lineage_continuity_verdict")
    if lineage_verdict and lineage_verdict not in ("flat", "no_signal"):
        parts.append(f"lineage {lineage_verdict.replace('_', ' ')}")
    return ", ".join(parts) if parts else "no interference signal"


def _revision_evidence(block: dict) -> tuple[str, Optional[str]]:
    n_stale = block.get("n_stale_propagations") or 0
    n_ent = block.get("n_entities_tracked") or 0
    if n_stale == 0:
        return "no stale-value citations detected", None
    summary = (
        f"{n_stale} stale propagation{'s' if n_stale != 1 else ''} "
        f"across {n_ent} tracked entit{'ies' if n_ent != 1 else 'y'}"
    )
    traj = block.get("value_supersession_trajectory") or \
           block.get("per_session_violation_trajectory") or []
    verdict = block.get("value_supersession_verdict") or \
              block.get("violation_trajectory_verdict") or "no_signal"
    if traj:
        extra = f"trajectory: {list(traj)} ({verdict.replace('_', ' ')})"
        return summary, extra
    return summary, None


def _maintenance_evidence(block: dict) -> str:
    n_shocks = block.get("n_shocks") or 0
    if n_shocks == 0:
        return "no lifecycle shock detected (clean)"
    delta_outcome = block.get("median_outcome_rate_delta")
    if delta_outcome is None or abs(delta_outcome) < 0.05:
        return f"{n_shocks} shocks detected, but no post-shock damage"
    direction = "drop" if delta_outcome < 0 else "rise"
    return (
        f"{n_shocks} shocks detected; outcome rate {direction} "
        f"{abs(delta_outcome):.2f} post-shock"
    )


# ─── Dominant-mechanism arbitration line ────────────────────────────────────

def _dominant_line(dm: dict) -> str:
    dominant = dm.get("dominant")
    reason = dm.get("reason")
    scores = dm.get("scores") or {}

    if dominant is None:
        if reason == "co_dominant":
            return f"Dominant mechanism:    co-dominant ({' + '.join(dm.get('co_dominant') or [])})"
        if reason == "no_independent_evidence":
            compat = dm.get("compatible") or []
            return f"Dominant mechanism:    no independent evidence; compatible: {', '.join(compat)}"
        return "Dominant mechanism:    no signal"

    # Margin description
    top_score = scores.get(dominant, 0.0)
    other_scores = sorted(
        (v for k, v in scores.items() if k != dominant),
        reverse=True,
    )
    if other_scores and other_scores[0] > 0:
        runner_up_score = other_scores[0]
        runner_up_mech = next(
            (k for k, v in scores.items() if k != dominant and v == runner_up_score),
            None,
        )
        margin = top_score / runner_up_score if runner_up_score else float("inf")
        return (
            f"Dominant mechanism:    {dominant} "
            f"(score {top_score:.2f} vs runner-up {runner_up_mech} "
            f"{runner_up_score:.2f}, margin {margin:.2f})"
        )
    return f"Dominant mechanism:    {dominant} (score {top_score:.2f}, no runner-up)"
