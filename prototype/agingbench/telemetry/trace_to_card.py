"""
agingbench/telemetry/trace_to_card.py — Map a production trace into an AgingCard.

Two callable surfaces:

  trace_to_card(...)        — v1.0 STUB API. Backward-compatible.
                              Returns a partial card with cost-block aggregation.
                              Kept verbatim for existing callers.

  trace_to_card_v11(...)    — v1.1 full pipeline.
                              Adapter → privacy scrub → session detect → memory
                              reconstruct → 4-mechanism inference → AgingCard
                              with `trace_audit` block.

The v1.1 surface is opt-in (separate function name) so consumers of the
v1.0 stub aren't surprised by output schema changes. v1.0 callers can
migrate by switching to `trace_to_card_v11` whenever they're ready.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# Trace formats the v1 stub knows about. NOTE: `langsmith` is included
# (silently routes through the `generic` adapter) for backward compat,
# but is intentionally NOT advertised as a first-class format in the
# README — it ships without a dedicated fixture/test. Users with
# LangSmith run JSON should prefer `trace_format="generic"` going
# forward; a dedicated langsmith adapter + fixture is on the roadmap.
SUPPORTED_TRACE_FORMATS = (
    "langfuse", "langsmith", "otlp", "generic", "claude_code",
    "openai_assistants", "openhands",
)


@dataclass
class TraceToCardResult:
    """Result of running `trace_to_card` (v1.0 stub).

    Attributes
    ----------
    card : dict
        The (partial) AgingCard dict.
    derived_fields : list[str]
        Field names that were successfully derived from the trace.
    missing_fields : list[str]
        Field names that could NOT be derived.
    n_calls : int
        Number of trace events ingested.
    """

    card: dict
    derived_fields: list[str]
    missing_fields: list[str]
    n_calls: int


def trace_to_card(trace_jsonl: Path,
                  scenario_hint: Optional[str] = None,
                  sut_hint: Optional[dict] = None,
                  trace_format: str = "generic") -> TraceToCardResult:
    """Translate a JSONL trace file into a partial AgingCard. (v1.0 stub.)

    Backward-compatible signature; behavior unchanged from prior release.
    For the full v1.1 pipeline (mechanism inference + trace_audit block),
    call `trace_to_card_v11(...)` instead.
    """
    trace_jsonl = Path(trace_jsonl)
    if trace_format not in SUPPORTED_TRACE_FORMATS:
        raise ValueError(
            f"Unsupported trace_format {trace_format!r}; "
            f"must be one of {SUPPORTED_TRACE_FORMATS}"
        )

    events: list[dict] = []
    if trace_jsonl.is_file():
        with trace_jsonl.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    n_calls = len(events)
    in_toks, out_toks = _sum_tokens(events)

    derived: list[str] = ["card_envelope", "n_calls", "cost.total_tokens"]
    missing: list[str] = [
        "checkpoints",
        "headline.m_final",
        "mechanism_metrics.*",
        "pressure",
        "seed",
    ]

    from agingbench.metrics.aging_card import build_aging_card

    fake_metrics = {
        "scenario": scenario_hint or "telemetry_unknown",
        "sut_id": (sut_hint or {}).get("sut_id", "telemetry_unknown"),
        "metric_group": "telemetry",
        "m0": None, "m_final": None, "half_life": None,
        "decay_slope": None, "hazard_proxy": None,
        "n_checkpoints": 0, "n_sessions": 0,
        "headline_metric": "telemetry_partial",
        "aging_detected": None,
        "checkpoints": [], "session_results": [],
    }
    card = build_aging_card(
        metrics=fake_metrics, sut_cfg=sut_hint or {},
        dependency_metrics=None, warnings=["telemetry_partial"],
        extra_provenance={
            "trace_format": trace_format,
            "trace_file": str(trace_jsonl),
            "telemetry_stub_version": "v1.0.0-stub",
        },
    )
    card["cost_and_efficiency"] = {
        **(card.get("cost_and_efficiency") or {}),
        "total_input_tokens": in_toks or None,
        "total_output_tokens": out_toks or None,
        "total_calls": n_calls,
    }
    return TraceToCardResult(card=card, derived_fields=derived,
                             missing_fields=missing, n_calls=n_calls)


def _sum_tokens(events: list[dict]) -> tuple[int, int]:
    """Best-effort token-count aggregation across heterogeneous trace formats."""
    in_toks = 0
    out_toks = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        for k in ("input_tokens", "prompt_tokens", "in_tokens"):
            if k in ev and isinstance(ev[k], (int, float)):
                in_toks += int(ev[k])
                break
        for k in ("output_tokens", "completion_tokens", "out_tokens"):
            if k in ev and isinstance(ev[k], (int, float)):
                out_toks += int(ev[k])
                break
        usage = ev.get("usage") if isinstance(ev.get("usage"), dict) else None
        if usage:
            for k in ("input_tokens", "prompt_tokens"):
                if k in usage and isinstance(usage[k], (int, float)):
                    in_toks += int(usage[k])
                    break
            for k in ("output_tokens", "completion_tokens"):
                if k in usage and isinstance(usage[k], (int, float)):
                    out_toks += int(usage[k])
                    break
    return in_toks, out_toks


# ---------------------------------------------------------------------------
# v1.1 full pipeline — opt-in via separate function
# ---------------------------------------------------------------------------

@dataclass
class TraceToCardV11Result:
    card: dict                          # full AgingCard with trace_audit block
    n_records: int                      # records ingested (post-adapter)
    n_sessions: int
    n_outcome_events: int
    session_detection_mode: str
    profile_used: str
    outcome_rules_hash: str


def trace_to_card_v11(
    trace_jsonl: Path,
    trace_format: str = "generic",
    profile: str = "generic",
    overrides: Optional[dict] = None,
    sut_hint: Optional[dict] = None,
    outcomes_jsonl: Optional[Path] = None,
    extract_outcomes: Optional[list[str]] = None,
    synthetic_probe_cards: Optional[list[Path]] = None,
    scrub_pii: bool = True,
    idle_gap_minutes: float = 30.0,
) -> TraceToCardV11Result:
    """Full v1.1 telemetry pipeline.

    Pipeline:
      1. Read raw events from `trace_jsonl`
      2. Normalize via adapter for `trace_format`
      3. PII-scrub (default on)
      4. Detect sessions (explicit_id → idle_gap fallback)
      5. Detect memory shock events
      6. Run 4 mechanism inferences
      7. Optionally attach OutcomeEvents from a separate jsonl
      8. Optionally merge synthetic-probe AgingCards into the result
      9. Emit one combined AgingCard with `trace_audit` block

    Parameters
    ----------
    trace_jsonl : Path
        Path to the production trace.
    trace_format : str
        One of SUPPORTED_TRACE_FORMATS.
    profile : str
        Deployment-profile name (`generic`, `code_assistant`, ...).
    overrides : dict, optional
        Per-call overrides applied to the loaded profile (e.g.
        `{"outcome_rules": {"ticket_escalated": "success"}}`).
    sut_hint : dict, optional
        SUT metadata (sut_id, model_id, memory_policy_type) for the card's
        `sut` block.
    outcomes_jsonl : Path, optional
        Separate JSONL of OutcomeEvent records to attach.
    extract_outcomes : list[str], optional
        Names of built-in OutcomeEvent extractors to run against the
        records. Each entry can be a bare name ("claude_session_flags",
        "record_patterns") or a name:arg form ("git_log:./my-repo[:since_days=30]").
        See agingbench.telemetry.outcome_extractors.list_extractors().
    synthetic_probe_cards : list[Path], optional
        Paths to AgingCards produced by `agingbench run --scenario ... --card`
        runs against the user's deployed agent. These get merged into the
        result under `synthetic_probes`.
    scrub_pii : bool
        Run PII redaction on prompt/response previews. Default True.
    idle_gap_minutes : float
        Session-detection idle threshold for traces without explicit IDs.
    """
    from .adapters import get_adapter
    from .card_lookups import diagnostic_signature, recommended_repair
    from .inference import (
        infer_compression, infer_interference, infer_revision, infer_maintenance,
    )
    from .inference._selector import pick_dominant
    from .inference.consistency import infer_consistency
    from .memory_reconstruction import detect_shocks
    from .privacy_scrubber import scrub_records
    from .profiles import load_profile, merge_overrides, outcome_rules_hash
    from .schema import OutcomeEvent
    from .session_detection import detect_sessions
    from .synthetic_probe import load_probe_result, merge_probe_into_card

    trace_path = Path(trace_jsonl)
    adapter = get_adapter(trace_format)
    prof = merge_overrides(load_profile(profile), overrides)

    # 1+2. Read + normalise
    records = []
    if trace_path.is_file():
        with trace_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = adapter(ev)
                if rec is not None:
                    records.append(rec)

    # 3. Privacy scrub
    if scrub_pii:
        records = scrub_records(records)

    # 4. Session detection
    sessions, mode = detect_sessions(records, idle_gap_minutes=idle_gap_minutes)

    # 5. Memory shock detection
    shocks = detect_shocks(sessions)

    # 6. Outcome event attachment. Hash outcome session_ids the same way
    # records were hashed during scrubbing so the join key matches.
    outcomes: list[OutcomeEvent] = []
    if outcomes_jsonl and Path(outcomes_jsonl).is_file():
        outcomes.extend(_load_outcome_events(Path(outcomes_jsonl),
                                              hash_session_ids=scrub_pii))
    # Run built-in extractors against the records.
    if extract_outcomes:
        from .outcome_extractors import run_extractor
        for spec in extract_outcomes:
            outcomes.extend(run_extractor(spec, records))

    # 7. Mechanism inference
    compression_b  = infer_compression(sessions)
    interference_b = infer_interference(sessions)
    revision_b     = infer_revision(sessions)
    maintenance_b  = infer_maintenance(sessions, shocks, outcomes)
    # P5: cross-session task-consistency probe (load-bearing for the card
    # headline when outcomes are absent).
    consistency_b  = infer_consistency(sessions)

    # 8. Headline curve from outcomes (if any). Falls back to P5
    # behavior_drift_at_repeat, then to aggregate mechanism trend.
    headline_block = _headline_from_outcomes(sessions, outcomes)
    headline_block = _augment_headline_with_fallbacks(
        headline_block, consistency_b,
        compression_b, interference_b, revision_b, maintenance_b,
        outcomes_present=bool(outcomes),
    )

    # 9. Build AgingCard envelope
    from agingbench.metrics.aging_card import build_aging_card

    cost_block = _aggregate_cost_block(records, n_sessions=len(sessions))

    fake_metrics = {
        "scenario": f"telemetry::{prof.deployment_type}",
        "sut_id":   (sut_hint or {}).get("sut_id", "telemetry_unknown"),
        "metric_group": "telemetry",
        "m0":          headline_block.get("m0"),
        "m_final":     headline_block.get("m_final"),
        "half_life":   headline_block.get("half_life"),
        "decay_slope": headline_block.get("decay_slope"),
        "hazard_proxy": None,
        "n_checkpoints": len(headline_block.get("checkpoints", [])),
        "n_sessions": len(sessions),
        "headline_metric": "telemetry_outcome_rate" if outcomes else "telemetry_partial",
        "aging_detected": _aging_detected_v12(headline_block),
        "checkpoints": headline_block.get("checkpoints", []),
        "session_results": [],
        # Surface aggregated cost so build_aging_card._build_cost_block picks
        # them up via its top-level priority.
        **{k: v for k, v in cost_block.items() if v is not None},
    }

    warnings = []
    if not outcomes:
        warnings.append("telemetry_no_outcome_events")
        warnings.append("headline_metric_not_derivable_without_outcomes")
    if not records:
        warnings.append("telemetry_empty_trace")

    card = build_aging_card(
        metrics=fake_metrics,
        sut_cfg=sut_hint or {},
        dependency_metrics=None,
        warnings=warnings,
        extra_provenance={
            "trace_format":   trace_format,
            "trace_file":     str(trace_path),
            "deployment_type": prof.deployment_type,
            "telemetry_version": "v1.1",
        },
    )

    # Attach the trace_audit block (parallel to mechanism_metrics)
    rules_hash = outcome_rules_hash(prof)
    trace_audit = {
        "derived_from":           "telemetry",
        "deployment_type":        prof.deployment_type,
        "n_sessions_detected":    len(sessions),
        "n_outcome_events":       len(outcomes),
        "session_detection_mode": mode,
        "outcome_rules_hash":     rules_hash,
        "compression":            compression_b,
        "interference":           interference_b,
        "revision":               revision_b,
        "maintenance":            maintenance_b,
        "consistency":            consistency_b,
        "headline":               headline_block,
        "trace_regime":           _build_trace_regime(
            sessions, records, outcomes, trace_format
        ),
    }

    # Dominant-mechanism selector (independent-evidence gated + margin).
    selector_result = pick_dominant(trace_audit)
    trace_audit["dominant_mechanism"] = selector_result
    # Atlas's card surface: signature + repair from static lookups when a
    # single mechanism dominates with adequate separation.
    if selector_result["dominant"] is not None:
        trace_audit["signature"] = diagnostic_signature(selector_result["dominant"])
        trace_audit["repair"]    = recommended_repair(selector_result["dominant"])
    else:
        trace_audit["signature"] = None
        trace_audit["repair"]    = None

    card["trace_audit"] = trace_audit

    # 10. Merge synthetic probes if any
    if synthetic_probe_cards:
        for path in synthetic_probe_cards:
            if Path(path).is_file():
                probe = load_probe_result(Path(path))
                merge_probe_into_card(card, probe)

    return TraceToCardV11Result(
        card=card,
        n_records=len(records),
        n_sessions=len(sessions),
        n_outcome_events=len(outcomes),
        session_detection_mode=mode,
        profile_used=prof.deployment_type,
        outcome_rules_hash=rules_hash,
    )


def _load_outcome_events(path: Path, hash_session_ids: bool = True) -> list[OutcomeEvent]:
    from .privacy_scrubber import hash_session_id
    from .schema import OutcomeEvent
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") == "outcome" or "outcome" in d:
                sid = d.get("session_id", "_unknown")
                if hash_session_ids and sid != "_unknown":
                    sid = hash_session_id(sid)
                out.append(OutcomeEvent(
                    session_id=sid,
                    task_id=d.get("task_id", "_unknown"),
                    outcome=d.get("outcome", "abandoned"),
                    user_signal=d.get("user_signal"),
                    correction_of=d.get("correction_of"),
                    source=d.get("source", "user"),
                ))
    return out


def _aggregate_cost_block(records, n_sessions: int) -> dict:
    """Aggregate per-record token / latency / cost into the cost_and_efficiency
    fields. Returns a dict ready to merge into `metrics` for build_aging_card.
    Any field with no signal is left as None so the downstream `pick()` logic
    in aging_card.py treats it as missing.
    """
    import statistics

    if not records:
        return {"total_input_tokens": None, "total_output_tokens": None,
                "total_calls": None, "tokens_per_session_mean": None,
                "total_cost_usd": None, "latency_ms_p50": None,
                "latency_ms_p95": None}

    in_toks = sum(r.input_tokens for r in records)
    out_toks = sum(r.output_tokens for r in records)
    n_calls = sum(1 for r in records if r.role == "agent")
    latencies = [r.duration_ms for r in records if r.duration_ms is not None]
    costs = [r.cost_usd for r in records if r.cost_usd is not None]

    def _percentile(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        idx = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
        return float(s[idx])

    return {
        "total_input_tokens":      in_toks if in_toks > 0 else None,
        "total_output_tokens":     out_toks if out_toks > 0 else None,
        "total_calls":             n_calls if n_calls > 0 else None,
        "tokens_per_session_mean": (
            (in_toks + out_toks) / n_sessions if n_sessions > 0 and (in_toks + out_toks) > 0 else None
        ),
        "total_cost_usd":          (sum(costs) if costs else None),
        "latency_ms_p50":          _percentile(latencies, 50),
        "latency_ms_p95":          _percentile(latencies, 95),
    }


def _headline_from_outcomes(sessions, outcomes) -> dict:
    """Compute aging-curve checkpoints from outcome events.

    For each session, compute success rate from any outcome events whose
    session_id matches a record in that session.
    """
    if not outcomes or not sessions:
        return {}

    # session_idx → list[OutcomeEvent]
    by_session: dict[int, list] = {}
    for s_idx, sess in enumerate(sessions):
        sids = {r.session_id for r in sess if r.session_id}
        for o in outcomes:
            if o.session_id in sids:
                by_session.setdefault(s_idx, []).append(o)

    if not by_session:
        return {}

    checkpoints = []
    for s_idx in sorted(by_session.keys()):
        os = by_session[s_idx]
        if not os:
            continue
        rate = sum(1 for o in os if o.outcome == "success") / len(os)
        checkpoints.append([s_idx, round(rate, 4)])

    if not checkpoints:
        return {}

    m0 = checkpoints[0][1]
    m_final = checkpoints[-1][1]
    # OLS slope
    xs = [c[0] for c in checkpoints]
    ys = [c[1] for c in checkpoints]
    n = len(xs)
    if n >= 2:
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = round(num / den, 6) if den else None
    else:
        slope = None

    # Half-life: first session where score <= m0/2
    half_life = None
    for s_idx, score in checkpoints:
        if score <= m0 / 2:
            half_life = s_idx
            break

    return {
        "checkpoints":  checkpoints,
        "m0":           round(m0, 4),
        "m_final":      round(m_final, 4),
        "decay_slope":  slope,
        "half_life":    half_life,
    }


def _aging_detected_v12(headline_block: Optional[dict]) -> Optional[bool]:
    """Return True/False/None for whether aging is detected.

    Widened from the pre-v1.2 check (which required outcome-derived
    decay_slope to fire) so traces without OutcomeEvents still surface
    aging when v1.2 signals (behavior_drift on repeat tasks or rising
    aggregate mechanism trend) say so. Returns None only when the
    headline block is itself empty.
    """
    if not headline_block:
        return None

    # Tier 1 (outcome-derived): decay_slope < -0.01 or ≥10% drop from m0
    decay = headline_block.get("decay_slope")
    if decay is not None and decay < -0.01:
        return True
    m0 = headline_block.get("m0")
    m_final = headline_block.get("m_final")
    if (m0 is not None and m_final is not None and m0 > 0
            and (m0 - m_final) / m0 >= 0.10):
        return True

    source = headline_block.get("source") or ""

    # Tier 2: behavior_drift on repeat tasks ≥ 10% counts as detected
    if source == "behavior_drift_at_repeat":
        drift = headline_block.get("behavior_drift_at_repeat") or 0.0
        if drift > 0.10:
            return True

    # Tier 3: aggregate mechanism severity rising
    if source == "aging_trend":
        slope = headline_block.get("aging_trend_slope") or 0.0
        if slope > 0.01:
            return True

    # Tier 4 (additive — fires regardless of headline source): maintenance
    # shock_damage trajectory rising on a trace with enough shock events.
    # The aggregate_mechanism_trend that feeds Tier 3 does NOT include the
    # maintenance block, so without this clause a trace whose only aging
    # signal is repeated shock-damage accumulation would be flagged as
    # "no aging" even though the dominant-mechanism selector correctly
    # identifies it as maintenance-dominant.
    if (headline_block.get("maintenance_shock_damage_verdict") == "rising_degradation"
            and (headline_block.get("maintenance_n_shocks") or 0) >= 3):
        return True

    return False


def _augment_headline_with_fallbacks(
    headline: dict,
    consistency: dict,
    compression: dict, interference: dict, revision: dict, maintenance: dict,
    *,
    outcomes_present: bool,
) -> dict:
    """Implement the 4-tier headline policy described in the plan.

    Tier 1: outcomes present                          -> half_life (already computed)
    Tier 2: outcomes absent, P5 clusters present      -> behavior_drift_at_repeat
    Tier 3: neither, but mechanism trend rises        -> aging_trend slope
    Tier 4: nothing                                    -> not_measurable

    Always returns a dict with at least: source, label. Preserves tier-1
    fields (checkpoints, m0, m_final, half_life, decay_slope) when present.

    Also stashes maintenance shock-damage info (`maintenance_shock_damage_verdict`,
    `maintenance_n_shocks`) on the headline dict regardless of which tier
    fires, so `_aging_detected_v12` can read it without an extra parameter.
    """
    out = dict(headline or {})

    # Stash maintenance shock-damage signal on the headline (additive — does
    # not affect tier selection, only consumed by the aging-detection flag).
    if maintenance:
        out["maintenance_shock_damage_verdict"] = maintenance.get("shock_damage_verdict")
        out["maintenance_n_shocks"] = maintenance.get("n_shocks") or 0

    if outcomes_present and out.get("half_life") is not None:
        out["source"] = "outcome_half_life"
        out["label"]  = f"Half-life: {out['half_life']} sessions"
        return out

    # Tier 2: P5-derived behavior_drift. Fires on ≥ 1 repeat-task cluster
    # (each cluster already ≥ 2 occurrences by `min_cluster_size=2`). The
    # `underpowered` boolean flag stays on the headline dict so a UI can
    # apply muted styling if it wants, but the label itself stays clean.
    n_clusters = (consistency or {}).get("n_repeated_tasks_detected", 0)
    if n_clusters >= 1:
        drift = consistency.get("behavior_drift_at_repeat") or 0.0
        n_sessions = consistency.get("consistency_drop_trajectory") or []
        cluster_sizes = consistency.get("cluster_sizes") or []
        n_occurrences = sum(cluster_sizes)
        out["source"] = "behavior_drift_at_repeat"
        out["label"]  = (
            f"Behavior drift: {round(drift * 100)}% on repeat tasks "
            f"({len(n_sessions)} sessions, {n_clusters} cluster"
            f"{'s' if n_clusters != 1 else ''}, {n_occurrences} occurrences)"
        )
        out["behavior_drift_at_repeat"] = drift
        out["n_clusters"] = n_clusters
        out["underpowered"] = n_clusters == 1
        return out

    # Tier 3: aggregate mechanism severity rising over time.
    trend_slope, n_blocks_rising = _aggregate_mechanism_trend(
        compression, interference, revision, maintenance,
    )
    if trend_slope is not None and trend_slope > 0.01 and n_blocks_rising >= 1:
        out["source"] = "aging_trend"
        out["label"]  = f"Aging trend: rising (slope {trend_slope:+.3f}/session)"
        out["aging_trend_slope"] = round(trend_slope, 4)
        return out

    # Tier 3.5: maintenance shock-damage (cumulative form). Catches traces
    # where shocks fired enough to accumulate meaningful damage even when
    # the per-session-delta aggregate (Tier 3) doesn't qualify — e.g.,
    # front-loaded shocks that taper later in deployment. Symmetric with
    # the Tier-4 clause in `_aging_detected_v12` so the headline label
    # matches the aging flag.
    if (maintenance and maintenance.get("shock_damage_verdict") == "rising_degradation"
            and (maintenance.get("n_shocks") or 0) >= 3):
        n_shocks = maintenance.get("n_shocks") or 0
        cum = (maintenance.get("shock_damage_trajectory") or [0])[-1] if maintenance.get("shock_damage_trajectory") else 0.0
        out["source"] = "maintenance_shock_damage"
        out["label"]  = f"Maintenance damage: {n_shocks} shocks, cumulative {cum:.1f}"
        out["maintenance_cum_damage"] = round(cum, 2)
        return out

    # Tier 4: not measurable.
    out["source"] = "not_measurable"
    out["label"]  = "Aging not measurable on this trace"
    out["disclosure"] = (
        "Enable an outcome extractor or run a longer trace with repeat "
        "tasks to surface a headline metric."
    )
    return out


def _aggregate_mechanism_trend(
    compression: dict, interference: dict, revision: dict, maintenance: dict,
) -> tuple[Optional[float], int]:
    """Sum per-session severity scores across the four mechanism blocks and
    fit a slope. Returns (slope, n_blocks_with_rising_signal)."""
    series: list[Optional[float]] = []
    n_rising = 0
    # Compression: saturation trajectory (already (i, v) tuples)
    sat = (compression or {}).get("saturation_trajectory") or []
    sat_vals = [v for _, v in sat]
    if sat_vals:
        series.append(sat_vals)
        if compression.get("saturation_slope") and compression["saturation_slope"] > 0.005:
            n_rising += 1
    # Interference: KL trajectory
    kl = (interference or {}).get("tool_kl_trajectory") or []
    if kl:
        series.append(list(kl))
        if interference.get("tool_kl_slope") and interference["tool_kl_slope"] > 0.005:
            n_rising += 1
    # Revision: violation trajectory (already per-session counts)
    rev = (revision or {}).get("value_supersession_trajectory") or (revision or {}).get("per_session_violation_trajectory") or []
    if rev:
        series.append(list(rev))
        if revision.get("value_supersession_slope") and revision["value_supersession_slope"] > 0.005:
            n_rising += 1
    # Maintenance: shock_damage_trajectory is cumulative — convert to
    # per-session deltas so it composes with the other per-session signals
    # without dominating the mean. A rising per-session damage series means
    # later shocks bite harder than earlier ones; that's the maintenance
    # contribution to aggregate aging.
    maint_cum = (maintenance or {}).get("shock_damage_trajectory") or []
    if maint_cum:
        maint_per_session = [
            (maint_cum[i] - maint_cum[i-1]) if i > 0 else maint_cum[0]
            for i in range(len(maint_cum))
        ]
        series.append(maint_per_session)
        sd_slope = maintenance.get("shock_damage_slope")
        if sd_slope is not None and sd_slope > 0.05:
            n_rising += 1
    if not series:
        return None, 0
    # Aggregate: per-session mean of available signals.
    max_len = max(len(s) for s in series)
    agg = []
    for i in range(max_len):
        vals = [s[i] for s in series if i < len(s) and s[i] is not None]
        if vals:
            agg.append(sum(vals) / len(vals))
        else:
            agg.append(None)
    nn = [v for v in agg if v is not None]
    if len(nn) < 3:
        return None, n_rising
    # OLS slope
    n = len(nn)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(nn) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, nn))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else None
    return slope, n_rising


def _build_trace_regime(sessions, records, outcomes, trace_format: str) -> dict:
    """Disclosure record: what kind of trace did the card run on?

    Used by the card UI to caveat low-coverage outputs and by the website
    sparkline to render appropriate verbosity.
    """
    n_tool_calls = 0
    for r in records:
        n_tool_calls += len(r.tool_calls or [])
    return {
        "tool_using":   n_tool_calls > 0,
        "multi_session": len(sessions) >= 2,
        "outcomes":      "linked" if outcomes else "absent",
        "n_sessions":    len(sessions),
        "n_records":     len(records),
        "n_tool_calls":  n_tool_calls,
        "adapter":       trace_format,
    }
