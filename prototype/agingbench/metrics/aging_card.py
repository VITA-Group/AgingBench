"""
agingbench/metrics/aging_card.py — AgingCard JSON consolidator.

Produces a flat, downstream-friendly JSON summary of an AgingBench run by
reading the existing per-run output files (metrics.json,
dependency_metrics.json) and packing them into a versioned envelope. The
schema is documented in `aging_card_schema.json`.

Design constraints:
- Pure post-processor. NEVER modifies `metrics.json` or
  `dependency_metrics.json`; only reads them.
- Opt-in: callers must pass the `--card` CLI flag for the file to be
  emitted.
- Additive only: produces a new output file `aging_card.json`; never
  changes an existing output path or payload.

Schema versioning: emits `schema_version` per SemVer. Major bumps add a
migration in `aging_card_migrate.py`. Current schema is 1.0.0.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

# Bump major for breaking changes; minor for additive optional fields; patch for fixes.
AGING_CARD_SCHEMA_VERSION = "1.0.0"
AGING_CARD_CARD_TYPE = "agingbench.AgingCard"


def build_aging_card(metrics: dict,
                     sut_cfg: Optional[dict] = None,
                     dependency_metrics: Optional[dict] = None,
                     pressure: Optional[Any] = None,
                     suite_id: Optional[str] = None,
                     seed: Optional[int] = None,
                     scenario_version: str = "1.0",
                     warnings: Optional[list[str]] = None,
                     extra_links: Optional[dict[str, str]] = None,
                     extra_provenance: Optional[dict[str, Any]] = None,
                     run_id: Optional[str] = None,
                     trace_path: Optional[Path] = None) -> dict:
    """
    Construct a v1.0.0 AgingCard dict from existing per-run outputs.

    Parameters
    ----------
    metrics : dict
        The deserialized `metrics.json` for this run. NOT mutated.
    sut_cfg : dict, optional
        The SUT YAML loaded as a dict. NOT mutated.
    dependency_metrics : dict, optional
        The deserialized `dependency_metrics.json`, if produced. NOT mutated.
    pressure : PressureConfig or dict, optional
        The PressureConfig used for this run; if a PressureConfig instance,
        `to_dict()` is called.
    suite_id : str, optional
        Suite identifier (e.g., "lite", "full", "adhoc").
    seed : int, optional
        Random seed for the run. Falls back to sut_cfg.get("seed").
    scenario_version : str
        Scenario version tag; defaults to "1.0".
    warnings : list[str], optional
        Caller-supplied warnings (e.g., "floor_saturation", "telemetry_partial").
    extra_links : dict, optional
        Additional file paths to surface in the "links" block.
    extra_provenance : dict, optional
        Additional provenance fields (git_sha, compute_environment, etc.).
    run_id : str, optional
        Caller-supplied run id; auto-generated UUID4 if omitted.
    trace_path : Path, optional
        Path to `trace.jsonl` for the run. When provided, the cost block
        aggregates tokens, calls, latency, and cost from per-call llm_call
        events; this is the only source for those fields when metrics.json
        lacks an aggregated cost block.

    Returns
    -------
    dict
        AgingCard payload conforming to v1.0.0. The caller is responsible for
        writing this to `aging_card.json`.
    """
    sut_cfg = sut_cfg or {}
    dependency_metrics = dependency_metrics or {}
    warnings = list(warnings or [])
    extra_links = dict(extra_links or {})
    extra_provenance = dict(extra_provenance or {})

    # Build the mechanism block up-front so we can flag mechanisms that the run
    # never exercised. The block is always shape-complete (all four keys
    # present, fields None when unmeasured) for cross-scenario consistency —
    # but an all-None block reads as "measured and came out empty" to anyone
    # skimming the card. Emit an explicit, machine-filterable transparency
    # warning instead, so a vacuous maintenance/interference block is never
    # mistaken for a real zero. (e.g. an S6 run with no lifecycle shocks, or
    # S1 which by design injects no interference content.)
    mechanism_block = _build_mechanism_block(metrics, dependency_metrics)
    for _m in _unexercised_mechanisms(mechanism_block):
        _w = f"mechanism_not_exercised:{_m}"
        if _w not in warnings:
            warnings.append(_w)

    card = {
        "schema_version": AGING_CARD_SCHEMA_VERSION,
        "card_type": AGING_CARD_CARD_TYPE,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id or str(uuid.uuid4()),

        "scenario": metrics.get("scenario", "unknown"),
        "scenario_version": scenario_version,
        "suite_id": suite_id or "adhoc",

        "sut": _build_sut_block(sut_cfg, metrics),

        "seed": _coerce_int(seed if seed is not None else sut_cfg.get("seed"), default=0),
        "n_sessions": _true_n_sessions(metrics),
        "pressure": _build_pressure_block(pressure),

        "headline": _build_headline_block(metrics),
        "mechanism_metrics": mechanism_block,
        "cost_and_efficiency": _build_cost_block(metrics, trace_path=trace_path),

        "checkpoints": list(metrics.get("checkpoints") or []),

        "provenance": _build_provenance_block(extra_provenance, sut_cfg),

        "warnings": warnings,
        "links": _build_links_block(extra_links),
    }
    return card


def write_aging_card(card: dict, output_dir: Path) -> Path:
    """Write the card JSON to `aging_card.json` in `output_dir`. Returns the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "aging_card.json"
    out_path.write_text(json.dumps(card, indent=2, sort_keys=True, default=_json_default))
    return out_path


def build_and_write_aging_card(output_dir: Path,
                               metrics: Optional[dict] = None,
                               sut_cfg: Optional[dict] = None,
                               dependency_metrics: Optional[dict] = None,
                               **kwargs) -> Optional[Path]:
    """
    Convenience: load metrics.json / dependency_metrics.json from `output_dir`
    if not passed, build a card, and write it. Returns the written path, or
    None if the required `metrics.json` is missing.

    Per non-interference guarantee: this function NEVER writes anywhere except
    `output_dir / 'aging_card.json'`. It only READS metrics.json and
    dependency_metrics.json.
    """
    output_dir = Path(output_dir)
    if metrics is None:
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.is_file():
            return None
        with metrics_path.open("r") as f:
            metrics = json.load(f)
    if dependency_metrics is None:
        dep_path = output_dir / "dependency_metrics.json"
        if dep_path.is_file():
            with dep_path.open("r") as f:
                dependency_metrics = json.load(f)
    # Auto-discover trace.jsonl in output_dir so the cost block aggregates
    # tokens/calls/latency from per-call llm_call events. Caller can override
    # by passing trace_path explicitly via kwargs.
    if "trace_path" not in kwargs:
        candidate = output_dir / "trace.jsonl"
        if candidate.is_file():
            kwargs["trace_path"] = candidate
    card = build_aging_card(
        metrics=metrics,
        sut_cfg=sut_cfg,
        dependency_metrics=dependency_metrics,
        **kwargs,
    )
    return write_aging_card(card, output_dir)


# ---------- block builders ----------

def _build_sut_block(sut_cfg: dict, metrics: dict) -> dict:
    model = sut_cfg.get("model") or {}
    memory_policy = sut_cfg.get("memory_policy") or {}
    return {
        "sut_id": sut_cfg.get("sut_id") or metrics.get("sut_id", "unknown"),
        "model_provider": (model.get("provider") if isinstance(model, dict) else None),
        "model_id": (model.get("model") if isinstance(model, dict) else None),
        "memory_policy_type": (memory_policy.get("type") if isinstance(memory_policy, dict) else None),
        "config_yaml_path": sut_cfg.get("_config_path"),
        "config_hash_sha256": sut_cfg.get("_config_hash_sha256"),
    }


def _build_pressure_block(pressure) -> dict:
    if pressure is None:
        return {}
    if hasattr(pressure, "to_dict"):
        d = dict(pressure.to_dict())
    elif isinstance(pressure, dict):
        d = dict(pressure)
    else:
        return {}
    return d


def _build_headline_block(metrics: dict) -> dict:
    return {
        "metric_name": metrics.get("headline_metric") or metrics.get("metric_group") or "primary",
        "m0": metrics.get("m0"),
        "m_final": metrics.get("m_final"),
        "half_life": metrics.get("half_life"),
        "decay_slope": metrics.get("decay_slope"),
        "hazard_proxy": metrics.get("hazard_proxy"),
        "aging_detected": _infer_aging_detected(metrics),
    }


def _infer_aging_detected(metrics: dict) -> Optional[bool]:
    """Return True iff this run shows a meaningful aging signal.

    Runners that pre-compute their own boolean win (explicit > inferred).
    Otherwise: True when decay_slope is negative beyond a small threshold
    OR m_final is materially below m0. Returns None only when there is no
    headline signal at all (no slope and no m0/m_final pair) — that case
    distinguishes "no measurement" from "measured no aging".
    """
    explicit = metrics.get("aging_detected")
    if isinstance(explicit, bool):
        return explicit
    slope = metrics.get("decay_slope")
    m0 = metrics.get("m0")
    m_final = metrics.get("m_final")
    if slope is None and m0 is None and m_final is None:
        return None
    # Slope-based criterion (preferred). The 0.01 threshold matches the
    # paper's "aging" cutoff used for the binary classification in §5.
    if slope is not None and slope < -0.01:
        return True
    # Magnitude-based fallback for runs with too few cycles to fit a slope.
    if m0 is not None and m_final is not None:
        try:
            if float(m0) > 0 and (float(m0) - float(m_final)) / float(m0) >= 0.10:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _true_n_sessions(metrics: dict) -> int:
    """Number of sessions actually RUN — one ``session_results`` entry per
    session.

    Must NOT be derived from ``n_checkpoints`` / the headline curve: a SPARSE
    headline curve undercounts. S6's recall (compression) curve skips any
    session whose stable-fact pool is empty (MED-1: an empty pool is None, not
    a fabricated 1.0), so ``n_checkpoints`` can be < sessions run. Authority
    order: explicit n_sessions → len(session_results) → len(task_raw, which is
    dense across all sessions) → checkpoints count.
    """
    explicit = _coerce_int(metrics.get("n_sessions"), default=0)
    if explicit:
        return explicit
    sr = metrics.get("session_results")
    if isinstance(sr, list) and sr:
        return len(sr)
    for key in ("task_raw", "checkpoints"):
        L = metrics.get(key)
        if isinstance(L, list) and L:
            return len(L)
    return _coerce_int(metrics.get("n_checkpoints"), default=0)


def _unexercised_mechanisms(mech: dict) -> list[str]:
    """Return the names of mechanisms whose block carries NO measured signal.

    Used to attach ``mechanism_not_exercised:<name>`` transparency warnings so a
    shape-complete-but-all-None block is never read as a measured zero. Each
    mechanism has its own "was this measured" predicate (a bare ``n_probes=0``
    or empty ``shock_sessions`` does NOT count as exercised; a real ``0.0``
    accuracy DOES).
    """
    out: list[str] = []

    comp = mech.get("compression") or {}
    if not comp.get("trajectory") and comp.get("score") is None:
        out.append("compression")

    intf = mech.get("interference") or {}
    if not intf.get("n_probes") and not any(
        intf.get(k) is not None for k in
        ("binding_accuracy", "confusion_rate", "miss_rate",
         "resistance", "resistance_legacy")
    ):
        out.append("interference")

    rev = mech.get("revision") or {}
    if not any(
        rev.get(k) is not None for k in
        ("version_accuracy", "forget_accuracy", "accumulator_abs_error",
         "accumulator_rel_error", "compounding_detected", "compounding_score",
         "stale_residue_rate", "stale_residue_count",
         "revision_fidelity_excess", "coverage_verdict")
    ) and not rev.get("compounding_trajectory"):
        out.append("revision")

    maint = mech.get("maintenance") or {}
    if not maint.get("shock_sessions") and not any(
        maint.get(k) is not None for k in ("pre_shock", "post_shock", "delta")
    ):
        out.append("maintenance")

    return out


def _build_mechanism_block(metrics: dict, dep: dict) -> dict:
    """
    Map the four aging mechanisms to the available metric fields. All four
    blocks (compression, interference, revision, maintenance) are always
    present so the card shape is consistent across scenarios — fields are
    None when the source data does not contain them, which is normal for
    scenarios that don't exercise every mechanism.

    Per-scenario notes:
      • S1 surfaces the baseline-corrected revision-aging trident
        (stale_residue_rate, revision_fidelity_excess, coverage_verdict)
        from the LAST cycle's session_results[*].revision_aging dict.
      • S1's interference + maintenance blocks stay all-None by design
        (interference content not injected; no lifecycle shocks).
    """
    accum = (dep.get("accumulator_metrics") if isinstance(dep, dict) else None) or {}

    # Surface the trident from the final cycle's revision_aging entry (S1).
    trident_final = {}
    sr = metrics.get("session_results") or []
    if sr:
        last = sr[-1] if isinstance(sr[-1], dict) else {}
        trident_final = last.get("revision_aging") or {}

    # Canonical interference signal = forced-choice binding accuracy from
    # dedicated probes (resists the confusable). The legacy
    # interference_resistance reuses the session's generic headline score and
    # does NOT measure confusable selection, so it is demoted to a *_legacy
    # field and only used as a fallback when no binding probes ran.
    _ib = (dep.get("interference_binding") if isinstance(dep, dict) else None) or {}
    _resistance_legacy = dep.get("interference_resistance") if isinstance(dep, dict) else None
    _resistance = _ib.get("binding_accuracy")
    if _resistance is None:
        _resistance = _resistance_legacy

    return {
        "compression": {
            "score": metrics.get("m_final"),
            "trajectory": list(metrics.get("checkpoints") or []),
        },
        "interference": {
            "resistance": _resistance,                 # binding_accuracy (canonical)
            "binding_accuracy": _ib.get("binding_accuracy"),
            "confusion_rate": _ib.get("confusion_rate"),
            "miss_rate": _ib.get("miss_rate"),
            "n_probes": _ib.get("n_probes") if _ib.get("n_probes") is not None
                        else (dep.get("n_interference_probes") if isinstance(dep, dict) else None),
            "resistance_legacy": _resistance_legacy,   # session-score proxy (deprecated)
        },
        "revision": {
            "version_accuracy": dep.get("version_accuracy") if isinstance(dep, dict) else None,
            "forget_accuracy": dep.get("forget_accuracy") if isinstance(dep, dict) else None,
            "accumulator_abs_error": accum.get("mean_error"),
            # Cross-SUT-comparable relative error: mean(|err|/max(|gold|,1)).
            # The abs_error field above stays in raw units (dollars) for
            # backward-compatibility; this one is the apples-to-apples signal.
            "accumulator_rel_error": accum.get("mean_relative_error"),
            "compounding_detected": accum.get("compounding_detected"),
            # Continuous companion to the strict-monotonic flag — least-squares
            # slope of error vs session. Positive ⇒ trending upward; catches
            # noisy upward trends the binary flag misses.
            "compounding_slope": accum.get("compounding_slope"),
            # ─── S2's compounding curve (revision multi-dep signal) ─────
            # Compounding probes pass only if every declared dependency is
            # still recallable; the curve decays multiplicatively as facts
            # age out. None when the scenario didn't generate compounding
            # probes (most non-S2 scenarios).
            "compounding_score": (
                (metrics.get("compounding_checkpoints") or [[None, None]])[-1][1]
                if metrics.get("compounding_checkpoints")
                else None
            ),
            "compounding_trajectory": list(metrics.get("compounding_checkpoints") or []),
            # ─── Trident (S1's baseline-corrected revision signal) ──────
            # Pulled from the final cycle's revision_aging dict so the
            # card's headline revision number reflects end-of-run state.
            # All four may be None if the run didn't enable revisions.
            "stale_residue_rate": trident_final.get("stale_residue_rate"),
            "stale_residue_count": trident_final.get("stale_residue_count"),
            "revision_fidelity_excess": trident_final.get("revision_fidelity_excess"),
            "coverage_verdict": trident_final.get("coverage_verdict"),
        },
        "maintenance": {
            "pre_shock": metrics.get("pre_shock"),
            "post_shock": metrics.get("post_shock"),
            "delta": metrics.get("maintenance_delta") or metrics.get("delta_shock"),
            "shock_sessions": metrics.get("shock_sessions") or [],
        },
    }


def _build_cost_block(metrics: dict,
                      trace_path: Optional[Path] = None) -> dict:
    """Extract cost/efficiency fields from the run.

    Policy (post-v0.3):
      - Tokens (input/output) and total_calls are CANONICAL. Every AgingCard
        emitted from a run that made LLM calls should populate them. v0.3
        ships this via trace.jsonl aggregation; older runs whose traces
        predate the aggregator still populate these fields correctly.
      - total_cost_usd and latency_ms_p50/p95 are ADVISORY. They populate
        only when callers pass `duration_ms` / `cost_usd` to
        `TraceLogger.log_llm_call()`. We do NOT require runners to
        instrument per-call timing/cost because:
          * latency is dominated by network/provider load (Tier 1) or
            tool execution outside the LLM call (Tier 2), and is not
            an aging signal;
          * cost_usd is `tokens * provider-pricing` — labs compute it
            against their own negotiated rates, and the canonical token
            counts are sufficient for any external pricing calculation.
        Their absence is expected and not a defect.

    Priority order for each field:
      1. Aggregated value already in metrics.json (runners may pre-aggregate)
      2. Sum/percentile over per-call llm_call events in trace.jsonl
      3. session_results array in metrics.json (legacy path, for runners that
         emit per-session totals)
      4. None (field is genuinely unknown for this run)

    The trace.jsonl path supersedes session_results because per-call records
    are more precise than per-session aggregates and capture latency that
    session_results never recorded.
    """
    # Path 3: legacy session_results — only used if trace path isn't present.
    sr = metrics.get("session_results") or []
    if not isinstance(sr, list):
        sr = []
    sr_in = sum(_safe_int(s.get("input_tokens")) for s in sr if isinstance(s, dict))
    sr_out = sum(_safe_int(s.get("output_tokens")) for s in sr if isinstance(s, dict))
    sr_response = sum(
        _safe_int(s.get("tokens") or s.get("response_tokens"))
        for s in sr if isinstance(s, dict)
    )
    # Treat empty session_results as "no signal" (None) rather than 0; the
    # `or None` collapses 0-sums to None so they don't shadow trace-derived
    # values downstream and don't surface as a misleading "we measured zero."
    sr_in = sr_in or None
    sr_out = sr_out or None
    sr_response = sr_response or None

    # True session count for per-session token averages — must use the dense
    # session count, not the (possibly sparse) headline-curve checkpoint count,
    # or tokens_per_session_mean is inflated. See _true_n_sessions.
    n_sessions_candidate = _true_n_sessions(metrics) or len(sr) or 1

    # Path 2: walk trace.jsonl, sum per-call usage. Best-effort; failures are
    # silent (cost block remains null rather than crashing card emission).
    trace_agg = _aggregate_cost_from_trace(trace_path) if trace_path else {}

    # Path 1: top-level pre-aggregates win when present.
    def pick(*candidates):
        """First non-None, non-zero value; or 0 if explicitly 0; else None."""
        for v in candidates:
            if v is not None and v != 0:
                return v
        for v in candidates:
            if v == 0:
                return 0
        return None

    total_in = pick(
        metrics.get("total_input_tokens"),
        trace_agg.get("input_tokens"),
        sr_in,
    )
    total_out = pick(
        metrics.get("total_output_tokens"),
        trace_agg.get("output_tokens"),
        sr_out,
    )
    total_resp = pick(
        metrics.get("total_response_tokens"),
        sr_response,
    )
    total_calls = pick(
        metrics.get("total_calls"),
        trace_agg.get("n_llm_calls"),
    )
    total_cost = pick(
        metrics.get("total_cost_usd"),
        trace_agg.get("cost_usd"),
    )
    latency_p50 = pick(
        metrics.get("latency_ms_p50"),
        trace_agg.get("latency_ms_p50"),
    )
    latency_p95 = pick(
        metrics.get("latency_ms_p95"),
        trace_agg.get("latency_ms_p95"),
    )

    # tokens_per_session_mean: prefer measured n_sessions over trace event grouping
    tokens_total = (total_in or 0) + (total_out or 0)
    tps_mean = (tokens_total / n_sessions_candidate) if tokens_total else None

    block = {
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_response_tokens": total_resp,
        "tokens_per_session_mean": tps_mean,
        "total_cost_usd": total_cost,
        "total_calls": total_calls,
    }
    # Latency fields are ADVISORY — they only populate when per-call
    # duration_ms was instrumented by the LLM client. When latency wasn't
    # recorded (the common case for local/HF inference), emitting them as
    # null reads as "we measured latency and found nothing" — better to
    # omit so absence is unambiguous. They appear iff there's a value.
    if latency_p50 is not None:
        block["latency_ms_p50"] = latency_p50
    if latency_p95 is not None:
        block["latency_ms_p95"] = latency_p95
    return block


def _aggregate_cost_from_trace(trace_path: Optional[Path]) -> dict:
    """Walk trace.jsonl and aggregate per-llm_call cost/latency/tokens.

    Returns a dict with any of: input_tokens, output_tokens, n_llm_calls,
    cost_usd, latency_ms_p50, latency_ms_p95. Missing data is omitted from
    the dict (caller treats absence as None).
    """
    if trace_path is None:
        return {}
    p = Path(trace_path)
    if not p.is_file():
        return {}

    n_calls = 0
    sum_in = 0
    sum_out = 0
    sum_cost = 0.0
    have_cost = False
    durations: list[float] = []

    try:
        with p.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") != "llm_call":
                    continue
                n_calls += 1
                # Token fields — support both flat and gen_ai.* forms.
                sum_in += _safe_int(
                    rec.get("gen_ai.usage.input_tokens")
                    or rec.get("input_tokens")
                    or rec.get("prompt_tokens")
                )
                sum_out += _safe_int(
                    rec.get("gen_ai.usage.output_tokens")
                    or rec.get("output_tokens")
                    or rec.get("completion_tokens")
                )
                cost = rec.get("gen_ai.usage.cost_usd") or rec.get("cost_usd")
                if cost is not None:
                    try:
                        sum_cost += float(cost)
                        have_cost = True
                    except (TypeError, ValueError):
                        pass
                dur = rec.get("gen_ai.usage.duration_ms") or rec.get("duration_ms") or rec.get("latency_ms")
                if dur is not None:
                    try:
                        durations.append(float(dur))
                    except (TypeError, ValueError):
                        pass
    except OSError:
        return {}

    out: dict[str, Any] = {}
    if n_calls:
        out["n_llm_calls"] = n_calls
    if sum_in:
        out["input_tokens"] = sum_in
    if sum_out:
        out["output_tokens"] = sum_out
    if have_cost:
        out["cost_usd"] = round(sum_cost, 6)
    if durations:
        out["latency_ms_p50"] = _percentile(durations, 50)
        out["latency_ms_p95"] = _percentile(durations, 95)
    return out


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for empty inputs."""
    if not values:
        return 0.0
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    # Linear interpolation between closest ranks.
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _build_provenance_block(extra: dict, sut_cfg: Optional[dict] = None) -> dict:
    prov: dict[str, Any] = {
        "agingbench_version": _read_agingbench_version(),
        "git_sha": _read_git_sha(),
        "compute_environment": os.environ.get("AGINGBENCH_ENV", "local"),
        "agent_cli_version": _detect_agent_cli_version(sut_cfg or {}),
    }
    prov.update(extra)
    return prov


def _detect_agent_cli_version(sut_cfg: dict) -> Optional[str]:
    """Best-effort capture of the agent CLI version (e.g. Claude Code) for the
    provenance block. The agent harness is a reproducibility-relevant variable,
    so different CLI versions can change run outcomes. Returns None for non-CLI
    (Tier-1) SUTs or on any failure; never raises (must not break card writing).

    Note: detected at card-build time, which for the normal run-then-card flow
    equals the run-time version. Post-hoc card regeneration reflects the CLI
    version present at regeneration time.
    """
    try:
        adapter = (sut_cfg or {}).get("adapter") or {}
        cli_path = adapter.get("cli_path") or {
            "claude_code": "claude", "cursor": "agent", "codex": "codex",
        }.get(adapter.get("type"))
        if not cli_path:
            return None
        import subprocess
        out = subprocess.run(
            [cli_path, "--version"], capture_output=True, text=True, timeout=15,
        )
        text = (out.stdout or out.stderr or "").strip()
        return text.splitlines()[0].strip() if text else None
    except Exception:
        return None


def _build_links_block(extra: dict) -> dict:
    links = {
        "metrics_json": "metrics.json",
        "dependency_metrics_json": "dependency_metrics.json",
        "trace_jsonl": "trace.jsonl",
    }
    links.update(extra)
    return links


# ---------- internal helpers ----------

def _coerce_int(v, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(v) -> int:
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _read_agingbench_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("agingbench")
    except Exception:  # pylint: disable=broad-except
        return None


def _read_git_sha() -> Optional[str]:
    """Best-effort read of the current commit SHA. Returns None if unavailable."""
    head_path = Path(__file__).resolve().parent.parent.parent.parent / ".git" / "HEAD"
    try:
        if head_path.is_file():
            head = head_path.read_text().strip()
            if head.startswith("ref: "):
                ref_path = head_path.parent / head[5:]
                if ref_path.is_file():
                    return ref_path.read_text().strip()
            return head
    except OSError:
        return None
    return None


def _json_default(o):
    """JSON serializer that handles common non-default types (Path, set, dataclass.to_dict)."""
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
