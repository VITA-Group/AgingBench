"""
inference/interference.py — Tool-distribution drift over sessions.

Primary signal: KL divergence of per-session tool-call distribution
relative to a baseline (early-session) distribution. Drift > 0 means
the agent is using tools differently than at start = behavioral drift
proxy for interference.

This is NOT scenario interference_resistance (which is the score on
constructed confusable pairs). The block flags it as
`derived_from: 'tool_distribution_drift'`.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Optional

from ..schema import TelemetryRecord, CoverageReport


def infer_interference(
    sessions: list[list[TelemetryRecord]],
    baseline_window_n: int = 3,
    epsilon: float = 0.001,
) -> dict:
    if len(sessions) < baseline_window_n + 1:
        return _empty(len(sessions))

    # Per-session tool-name distribution
    dists: list[dict[str, float]] = []
    all_tools: set[str] = set()
    for s in sessions:
        names = [tc.name for r in s for tc in r.tool_calls]
        if not names:
            dists.append({})
            continue
        c = Counter(names)
        total = sum(c.values())
        dists.append({k: v / total for k, v in c.items()})
        all_tools.update(c.keys())

    n_with_tools = sum(1 for d in dists if d)
    if n_with_tools == 0 or not all_tools:
        return _empty(len(sessions))

    # Smooth + project all distributions onto the union of tools
    smoothed = []
    for d in dists:
        denom = sum(d.values()) + epsilon * len(all_tools)
        smoothed.append({t: (d.get(t, 0) + epsilon) / denom for t in all_tools})

    baseline = _avg_distributions(smoothed[:baseline_window_n])
    kl_traj = [
        round(_kl(d, baseline), 4)
        for d in smoothed
    ]

    post_baseline = kl_traj[baseline_window_n:]
    kl_mean = round(statistics.mean(post_baseline), 4) if post_baseline else None
    kl_slope = _ols_slope(post_baseline) if len(post_baseline) >= 3 else None

    # NEW (long-horizon trajectory): semantic goal-anchor drift (P2).
    # Encode session-0 user prompt; for each later session compute cosine
    # similarity between the anchor embedding and the agent-text
    # embedding. Declining = the agent's output has drifted from the
    # original goal. Robust to paraphrasing (unlike the previous Jaccard
    # implementation which over-reported drift on synonyms).
    goal_drift_traj = _goal_anchor_drift_trajectory(sessions)
    goal_drift_slope = _ols_skipping_none(goal_drift_traj) if len(goal_drift_traj) >= 3 else None

    # NEW (P4): tool-result lineage continuity. Detects interference as
    # "agent stops referencing previously-emitted entity IDs." Build a
    # forward lineage graph: each tool result that emits an entity ID is
    # a node; later tool-call args referencing that ID are edges. Per
    # session: continuity rate = |reused| / |introduced|. Declining =
    # the agent is forgetting / overwriting prior entities.
    lineage_traj = _tool_lineage_continuity_trajectory(sessions)
    lineage_slope = _ols_skipping_none(lineage_traj) if len(lineage_traj) >= 3 else None

    coverage = _interference_coverage(sessions, n_with_tools)

    # goal_anchor_drift is bounded [0, 1]; falling = degradation.
    # Floor at 0.02 means the agent has effectively lost touch with the
    # session-0 vocabulary regardless of any tiny remaining slope.
    from ._verdict import degradation_verdict
    goal_anchor_verdict = degradation_verdict(
        goal_drift_traj, goal_drift_slope,
        rising_is_bad=False, floor_threshold=0.02,
        ceiling_threshold=None, slope_eps=0.005,
    )
    # lineage continuity bounded [0, 1]; falling = degradation.
    lineage_verdict = degradation_verdict(
        lineage_traj, lineage_slope,
        rising_is_bad=False, floor_threshold=0.05, slope_eps=0.005,
    )

    return {
        "tool_kl_trajectory":             kl_traj,
        "tool_kl_mean_post_baseline":     kl_mean,
        "tool_kl_slope":                  (round(kl_slope, 6) if kl_slope is not None else None),
        "baseline_window_size":           baseline_window_n,
        "n_distinct_tools":               len(all_tools),
        "goal_anchor_drift_trajectory":   [round(x, 4) if x is not None else None for x in goal_drift_traj],
        "goal_anchor_drift_slope":        (round(goal_drift_slope, 4) if goal_drift_slope is not None else None),
        "goal_anchor_drift_verdict":      goal_anchor_verdict,
        "lineage_continuity_trajectory":  [round(x, 4) if x is not None else None for x in lineage_traj],
        "lineage_continuity_slope":       (round(lineage_slope, 6) if lineage_slope is not None else None),
        "lineage_continuity_verdict":     lineage_verdict,
        "coverage":                       coverage.as_dict(),
        "derived_from":                   "tool_distribution_drift",
    }


def _tool_lineage_continuity_trajectory(sessions: list) -> list:
    """Per session: |entities reused from prior sessions| / |entities introduced
    so far|. Lower = the agent has lost touch with previously-emitted entities
    (interference-style forgetting). Returns None per session that has no
    introduced entities yet (avoid /0).

    Entities are sourced from:
      • Tool result_summary text (extract_capitalised_entities + UUID/path patterns)
      • Tool args (large IDs, UUIDs, paths via is_specific_value)
    Tool-call result IDs are the canonical "lineage". Reuse is "this entity
    ID appears in a later agent tool_call arg or response_preview."
    """
    from ._text_utils import extract_capitalised_entities, is_specific_value

    introduced_set: set[str] = set()
    out = []
    for sidx, s in enumerate(sessions):
        # New entities introduced in this session (from tool results)
        new_in_session: set[str] = set()
        # Reused entities in this session (in agent args/text)
        used_in_session: set[str] = set()

        for r in s:
            for tc in r.tool_calls or []:
                # Result-side: anything from tool result_summary is an emitted entity.
                if tc.result_summary:
                    ents = extract_capitalised_entities(tc.result_summary)
                    new_in_session.update(ents)
                    for v in _walk_values(_safe_dict(tc.result_summary)):
                        if isinstance(v, str) and is_specific_value(v):
                            new_in_session.add(v.lower())
                # Args-side: large IDs / UUIDs / paths used by the agent.
                if r.role == "agent" and tc.args:
                    for v in _walk_values(tc.args):
                        if isinstance(v, str) and is_specific_value(v):
                            used_in_session.add(v.lower())
            # Response-side: agent referencing entity by capitalisation
            if r.role == "agent" and r.response_preview:
                used_in_session.update(extract_capitalised_entities(r.response_preview))

        # Continuity = fraction of currently-used entities that were
        # already in the introduced set BEFORE this session started.
        prior_introduced = set(introduced_set)
        introduced_set.update(new_in_session)
        if not used_in_session or not prior_introduced:
            out.append(None)
        else:
            reused = used_in_session & prior_introduced
            out.append(len(reused) / len(used_in_session))
    return out


def _safe_dict(text: str) -> dict:
    """Try to JSON-parse result_summary; return empty dict on failure."""
    import json
    if not text:
        return {}
    s = text.strip()
    if not (s.startswith("{") or s.startswith("[")):
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"_": obj}
    except Exception:
        return {}


def _walk_values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_values(v)
    else:
        yield obj


def _goal_anchor_drift_trajectory(sessions: list) -> list:
    """Per session: cosine similarity of agent-text embedding vs the
    session-0 anchor embedding. Declining trajectory = the agent has
    drifted semantically from the original task framing.

    Uses sentence-transformers for semantic similarity (replaces the
    pre-P2 Jaccard implementation, which over-reported drift on
    paraphrases). Falls back to Jaccard if the encoder is unavailable
    so the install-time-no-NLP path still produces a signal.

    Field name `goal_anchor_drift_trajectory` is preserved for backward
    compatibility (website sparkline + existing tests). New
    `derived_from: "semantic_anchor_drift"` is exposed at the block level.

    Returns None per session that has no agent output to compare.
    """
    if not sessions:
        return []

    first_user = " ".join(
        r.prompt_preview or "" for r in sessions[0] if r.role == "user"
    )
    if not first_user.strip():
        return [None] * len(sessions)

    # Try the encoder path; fall back to Jaccard if unavailable.
    try:
        from ...metrics.semantic_scorer import _get_model, cosine_similarity
        model = _get_model()
    except ImportError:
        model = None

    if model is not None:
        anchor_emb = model.encode([first_user[:512]])[0]
        out = []
        agent_texts = []
        idx_map = []
        for i, s in enumerate(sessions):
            agent_text = " ".join(r.response_preview or "" for r in s if r.role == "agent")
            if not agent_text.strip():
                out.append(None)
            else:
                agent_texts.append(agent_text[:512])
                idx_map.append(i)
                out.append(None)  # placeholder
        if agent_texts:
            embs = model.encode(agent_texts)
            for emb, i in zip(embs, idx_map):
                out[i] = float(cosine_similarity(anchor_emb, emb))
        return out

    # Fallback: Jaccard on significant terms (preserves old behaviour for
    # environments without sentence-transformers).
    from ._text_utils import jaccard, significant_terms

    anchor = significant_terms(first_user)
    if not anchor:
        return [None] * len(sessions)
    out = []
    for s in sessions:
        agent_text = " ".join(r.response_preview or "" for r in s if r.role == "agent")
        if not agent_text:
            out.append(None)
            continue
        out.append(jaccard(anchor, significant_terms(agent_text)))
    return out


def _ols_skipping_none(ys):
    pairs = [(i, y) for i, y in enumerate(ys) if y is not None]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    den = sum((p[0] - mx) ** 2 for p in pairs)
    return num / den if den else None


def _empty(n_sess: int) -> dict:
    return {
        "tool_kl_trajectory":             [],
        "tool_kl_mean_post_baseline":     None,
        "tool_kl_slope":                  None,
        "baseline_window_size":           0,
        "n_distinct_tools":               0,
        "lineage_continuity_trajectory":  [],
        "lineage_continuity_slope":       None,
        "lineage_continuity_verdict":     "no_signal",
        "coverage":                       CoverageReport(n_sess, 0.0, "no_test_fired").as_dict(),
        "derived_from":                   "tool_distribution_drift",
    }


def _avg_distributions(dists: list[dict[str, float]]) -> dict[str, float]:
    if not dists:
        return {}
    keys = set().union(*[d.keys() for d in dists])
    return {k: sum(d.get(k, 0) for d in dists) / len(dists) for k in keys}


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    s = 0.0
    for k, pv in p.items():
        qv = q.get(k, 1e-12)
        if pv > 0 and qv > 0:
            s += pv * math.log(pv / qv)
    return s


def _ols_slope(ys: list[float]) -> Optional[float]:
    if len(ys) < 2:
        return None
    n = len(ys)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None


def _interference_coverage(sessions, n_with_tools: int) -> CoverageReport:
    n_sess = len(sessions)
    if n_sess == 0:
        return CoverageReport(0, 0.0, "no_test_fired")
    cov = n_with_tools / n_sess
    if cov > 0.5:
        verdict = "strong"
    elif cov > 0.2:
        verdict = "adequate"
    elif cov > 0:
        verdict = "weak"
    else:
        verdict = "no_test_fired"
    return CoverageReport(n_with_tools, round(cov, 3), verdict)
