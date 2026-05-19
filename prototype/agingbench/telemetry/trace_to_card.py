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


# Trace formats the v1 stub knows about.
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
    from .inference import (
        infer_compression, infer_interference, infer_revision, infer_maintenance,
    )
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

    # 8. Headline curve from outcomes (if any)
    headline_block = _headline_from_outcomes(sessions, outcomes)

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
        "aging_detected": (
            headline_block.get("decay_slope") is not None
            and headline_block.get("decay_slope") < -0.01
        ) if headline_block else None,
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
    card["trace_audit"] = {
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
        "headline":               headline_block,
    }

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
