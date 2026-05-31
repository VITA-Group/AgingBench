"""
agingbench/cli/runners.py — All per-scenario runner wrapper functions.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from .loaders import PROJECT_ROOT, SCENARIO_DIR, _discover_scenarios, _resolve_pressure


def _collect_response_token_diagnostics(session_results: list,
                                        max_tokens: int) -> dict | None:
    """
    Scan session_results for per-response token counts and split the cap-
    confound risk into two paths:
      - query_answer: tokens in scored query/task responses
                      (cap truncates the answer → evaluation confound)
      - memory_write: tokens emitted by the compaction step
                      (cap truncates the summary → memory corruption
                      confound that propagates forward)
    Overall `cap_confound_risk` is the worse of the two.

    Returns None if no response token data is available (e.g., older runs
    or scenarios that haven't been instrumented yet).
    """
    from agingbench.metrics.aging import flag_response_near_cap
    answer_tokens: list[int] = []
    write_tokens: list[int] = []
    for sr in session_results:
        for key in ("response_tokens_per_task", "response_tokens_probes",
                    "response_tokens_per_query"):
            val = sr.get(key)
            if isinstance(val, list):
                answer_tokens.extend(t for t in val if isinstance(t, int))
        for key in ("response_tokens_task", "response_tokens_probe",
                    "response_tokens_max"):
            val = sr.get(key)
            if isinstance(val, int):
                answer_tokens.append(val)
        mw = sr.get("memory_write_tokens")
        if isinstance(mw, int) and mw >= 0:
            write_tokens.append(mw)
        elif isinstance(mw, list):
            write_tokens.extend(t for t in mw if isinstance(t, int) and t >= 0)
    if not answer_tokens and not write_tokens:
        return None
    answer_diag = flag_response_near_cap(answer_tokens, max_tokens) if answer_tokens else None
    write_diag = flag_response_near_cap(write_tokens, max_tokens) if write_tokens else None
    _risk_rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": -1}
    risks = [d.get("cap_confound_risk") for d in (answer_diag, write_diag) if d]
    overall = max(risks, key=lambda r: _risk_rank.get(r, -1)) if risks else "unknown"
    out = {"cap_confound_risk": overall, "max_tokens_setting": max_tokens}
    if answer_diag:
        out["query_answer"] = answer_diag
    if write_diag:
        out["memory_write"] = write_diag
    return out


def _infer_max_tokens(sut_cfg: dict) -> int:
    """Extract the LLM's max_new_tokens / max_tokens setting from SUT YAML."""
    m = sut_cfg.get("model", {})
    return int(m.get("max_new_tokens", m.get("max_tokens", 700)))


# ------------------------------------------------------------------ helpers:
# mechanism-coverage trajectory curves (support Table 2 claims visually)

def _trajectory_curve(
    trajectory: dict,
    sut_id: str,
    scenario: str,
    metric_name: str,
):
    """Build an AgingCurve from a {session: value} trajectory dict.

    Used to turn the per-session outputs of dependency_scorer into plottable
    curves for revision/interference/forget_accuracy trajectories.
    """
    from agingbench.metrics.aging import AgingCurve
    if not trajectory:
        return None
    items = sorted(((int(k), float(v)) for k, v in trajectory.items()),
                   key=lambda x: x[0])
    exposures = [i for i, _ in items]
    scores = [v for _, v in items]
    return AgingCurve(
        sut_id=sut_id,
        scenario=scenario,
        exposures=exposures,
        scores=scores,
        metric_name=metric_name,
    )


def _emit_mechanism_aging_plot(
    primary_curve,
    dep_metrics: dict,
    output_dir: Path,
    sut_id: str,
    scenario: str,
    title: str,
    shock_sessions: list | None = None,
):
    """Write <output_dir>/aging_curve_mechanisms.png overlaying the headline
    curve with per-mechanism sub-metric trajectories.

    This is the figure that visually validates Table 2 claims:
      - Compression:   primary headline curve (passed in)
      - Interference:  interference_resistance_per_session (if data present)
      - Revision:      version_accuracy_per_session  +/or  forget_accuracy_per_session
      - Maintenance:   vertical red dashed lines at shock_sessions

    Curves whose trajectory dict is empty are skipped silently.
    """
    from agingbench.report.plot import compare_curves
    curves = [primary_curve]
    labels = [f"{primary_curve.metric_name} (headline)"
              if getattr(primary_curve, "metric_name", "")
              else "headline"]

    interf = _trajectory_curve(
        dep_metrics.get("interference_resistance_per_session") or {},
        sut_id, scenario, "interference_resistance",
    )
    if interf is not None:
        curves.append(interf)
        labels.append("interference_resistance")

    vacc = _trajectory_curve(
        dep_metrics.get("version_accuracy_per_session") or {},
        sut_id, scenario, "version_accuracy",
    )
    if vacc is not None:
        curves.append(vacc)
        labels.append("version_accuracy")

    forget = _trajectory_curve(
        dep_metrics.get("forget_accuracy_per_session") or {},
        sut_id, scenario, "forget_accuracy",
    )
    if forget is not None:
        curves.append(forget)
        labels.append("forget_accuracy")

    if len(curves) == 1 and not shock_sessions:
        # Nothing extra to show — the headline plot already has this.
        return

    compare_curves(
        curves,
        str(output_dir / "aging_curve_mechanisms.png"),
        title=title,
        labels=labels,
        shock_sessions=shock_sessions,
    )


# ------------------------------------------------------------------ S1 runner

def _run_s1(sut_cfg: dict, scenario_cfg: dict, output_dir: Path,
            n_cycles: int, oracle_mode: bool = False, oracle_retrieval: bool = False,
            oracle_store: bool = False, incontext_ceiling: bool = False,
            ceiling_max_tokens: int = 100_000,
            agent_class=None,
            generated: bool = False, gen_sessions: int = 0,
            score_via_response: bool = False) -> dict:
    """Execute S1 scenario (Research Literature Agent) and return metrics dict."""
    # SUT YAML can force oracle mode via top-level `oracle_mode: true` /
    # `oracle_retrieval: true`, letting users define a reusable "oracle source"
    # or "oracle retrieval" SUT variant without CLI flags. Never demotes a
    # CLI-enabled flag.
    oracle_mode = oracle_mode or sut_cfg.get("oracle_mode", False)
    oracle_retrieval = oracle_retrieval or sut_cfg.get("oracle_retrieval", False)
    oracle_store = oracle_store or sut_cfg.get("oracle_store", False)
    incontext_ceiling = incontext_ceiling or sut_cfg.get("incontext_ceiling", False)
    score_via_response = score_via_response or sut_cfg.get(
        "score_via_response", False)
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.runner.s1_runner import S1Runner
    from agingbench.runner.trace import TraceLogger
    from agingbench.metrics.aging import summarize, compute_half_life, compute_decay_slope
    from agingbench.scenarios.s1_research_literature.validator import score_all
    from agingbench.report.plot import compare_curves, plot_curve

    gen_data = None
    s1_dir = SCENARIO_DIR / "s1_research_literature"
    if generated:
        from agingbench.generators.s1_generator import S1Generator
        from agingbench.generators.pressure_config import PressureConfig
        gen_n = gen_sessions if gen_sessions > 0 else n_cycles
        gen_data = S1Generator(
            seed=sut_cfg.get("seed", 42),
            pressure=_resolve_pressure(sut_cfg, scenario_cfg),
            dense_revision=sut_cfg.get(
                "dense_revision",
                scenario_cfg.get("dense_revision", False),
            ),
        ).generate(gen_n)
        source_doc = gen_data["source_doc"]
        probes = gen_data["probes"]
        n_cycles = gen_n
    else:
        with open(s1_dir / "source_doc.json") as f:
            source_doc = json.load(f)
        with open(s1_dir / "probes.json") as f:
            probes = json.load(f)

    # Load compliance decision tasks from curated data (independent of seed)
    tasks = []
    tasks_path = s1_dir / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path) as f:
            tasks = [json.loads(line) for line in f if line.strip()]

    llm = load_llm(sut_cfg["model"])
    memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"

    with TraceLogger(str(trace_path)) as tracer:
        runner = S1Runner(
            source_doc_text=source_doc["text"],
            probes=probes,
            validator_fn=score_all,
            memory_policy=memory_policy,
            llm=llm,
            tracer=tracer,
            sut_id=sut_cfg["sut_id"],
            tasks=tasks,
            oracle_mode=oracle_mode,
            oracle_retrieval=oracle_retrieval,
            oracle_store=oracle_store,
            incontext_ceiling=incontext_ceiling,
            ceiling_max_tokens=ceiling_max_tokens,
            generated_data=gen_data,  # pass full generator output so runner uses seed-dependent paper_batches + session_facts
            score_via_response=score_via_response,
        )
        result = runner.run(
            n_cycles=n_cycles,
            seed=sut_cfg.get("seed", 42),
        )
        # S1Runner.run() now returns a dict (as of 2026-04-09 metric audit).
        # Older versions returned tuples; we keep the unpack for safety so a
        # cached older runner module won't crash the CLI.
        if isinstance(result, dict):
            keyword_curve = result["keyword_curve"]
            task_curve = result.get("task_curve")
            lag_recall_curve = result.get("lag_recall_curve")
            recall_matrix = result.get("recall_matrix")
            session_results = result.get("session_results", [])
            keyword_raw = result.get("keyword_raw", [])
            task_raw = result.get("task_raw", [])
            lag_recall_raw = result.get("lag_recall_raw", [])
            bloat_raw = result.get("bloat_raw", [])
        elif isinstance(result, tuple) and len(result) >= 4:
            keyword_curve, task_curve, lag_recall_curve, recall_matrix = result[:4]
            session_results, keyword_raw, task_raw, lag_recall_raw, bloat_raw = [], [], [], [], []
        elif isinstance(result, tuple):
            keyword_curve, task_curve = result
            lag_recall_curve, recall_matrix = None, None
            session_results, keyword_raw, task_raw, lag_recall_raw, bloat_raw = [], [], [], [], []
        else:
            keyword_curve, task_curve = result, None
            lag_recall_curve, recall_matrix = None, None
            session_results, keyword_raw, task_raw, lag_recall_raw, bloat_raw = [], [], [], [], []

    stats = summarize(keyword_curve)
    stats["scenario"] = "s1_research_literature"
    stats["metric_group"] = "G1"
    stats["headline_metric"] = "keyword_recall"
    # Persist every measured aging signal at the top level so consumers can
    # plot any of them without digging into session_results. The full
    # per-cycle records are also kept in session_results.
    stats["keyword_raw"] = keyword_raw
    stats["task_raw"] = task_raw
    stats["lag_recall_raw"] = lag_recall_raw
    stats["bloat_raw"] = bloat_raw
    stats["session_results"] = session_results
    if lag_recall_curve is not None:
        stats["lag_recall_curve"] = list(zip(lag_recall_curve.exposures, lag_recall_curve.scores))
    if recall_matrix is not None:
        stats["recall_matrix"] = recall_matrix
    if task_curve and task_curve.scores:
        stats["task_m0"] = task_curve.scores[0]
        stats["task_m_final"] = task_curve.scores[-1]
        stats["task_half_life"] = compute_half_life(task_curve)
        stats["task_decay_slope"] = round(compute_decay_slope(task_curve), 5)
        stats["task_checkpoints"] = list(zip(task_curve.exposures, task_curve.scores))

    # Attribution provenance (see s1_runner.run return dict). Guarded
    # because legacy runs returned a tuple rather than a dict.
    if isinstance(result, dict):
        if "attribution_schema" in result:
            stats["attribution_schema"] = result["attribution_schema"]
        if "attribution_mode" in result:
            stats["attribution_mode"] = result["attribution_mode"]
        if result.get("ceiling_max_tokens") is not None:
            stats["ceiling_max_tokens"] = result["ceiling_max_tokens"]
        # S1-specific: c2_abstain_s1 flags runs where oracle_retrieval was
        # requested but aliased to oracle_store (single-blob memory has no
        # distinct retrieve step). Propagated so downstream plots/tables can
        # honestly merge C2 and C3 bars for S1.
        if "c2_abstain_s1" in result:
            stats["c2_abstain_s1"] = result["c2_abstain_s1"]

    if oracle_mode:
        stats["oracle_mode"] = True

    if gen_data and "dependency_graph" in gen_data:
        from agingbench.metrics.dependency_scorer import score_dependency_chain
        # Use actual session_results (which include task_outputs_text for
        # forget_accuracy scanning), not synthetic ones from the aging curve.
        dep_session_results = session_results if session_results else [
            {"session": int(exp), "task_accuracy": float(score), "recall_accuracy": float(score)}
            for exp, score in zip(keyword_curve.exposures, keyword_curve.scores)
        ]
        dep_metrics = score_dependency_chain(dep_session_results, gen_data["dependency_graph"])
        stats["dependency_metrics"] = dep_metrics
        with open(output_dir / "dependency_metrics.json", "w") as f:
            json.dump(dep_metrics, f, indent=2)

    # Response-token cap-confound diagnostics (if runner populated the fields)
    _tok_diag = _collect_response_token_diagnostics(
        stats.get("session_results", []), _infer_max_tokens(sut_cfg)
    )
    if _tok_diag:
        stats["response_token_diagnostics"] = _tok_diag

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    title = f"S1 Aging — {sut_cfg['sut_id']}"
    if oracle_mode:
        title += " (oracle)"
    # Ensure the headline curve carries its metric name for axis-aware plotting
    keyword_curve.metric_name = "keyword_m"
    if task_curve:
        task_curve.metric_name = "task_m"
    if task_curve and task_curve.scores:
        compare_curves(
            [keyword_curve, task_curve],
            str(output_dir / "aging_curve.png"),
            title=title,
            labels=["keyword_m (headline)", "task_m"],
        )
    else:
        plot_curve(keyword_curve, str(output_dir / "aging_curve.png"), title=title)

    # Mechanism-coverage plot (Table 2: compression + revision via DAG)
    dep_metrics = stats.get("dependency_metrics") or {}
    _emit_mechanism_aging_plot(
        primary_curve=keyword_curve,
        dep_metrics=dep_metrics,
        output_dir=output_dir,
        sut_id=sut_cfg["sut_id"],
        scenario="s1_research_literature",
        title=f"{title} — mechanism trajectories",
    )

    return stats


# ------------------------------------------------------------------ S2 runner

def _run_s2(sut_cfg: dict, scenario_cfg: dict, output_dir: Path,
            n_cycles: int, oracle_mode: bool = False, oracle_retrieval: bool = False,
            oracle_store: bool = False, incontext_ceiling: bool = False,
            ceiling_max_tokens: int = 100_000,
            agent_class=None,
            generated: bool = False, gen_sessions: int = 0) -> dict:
    """Execute S2 scenario (Personal Finance & Lifestyle Assistant) and return metrics dict."""
    oracle_mode = oracle_mode or sut_cfg.get("oracle_mode", False)
    oracle_retrieval = oracle_retrieval or sut_cfg.get("oracle_retrieval", False)
    oracle_store = oracle_store or sut_cfg.get("oracle_store", False)
    incontext_ceiling = incontext_ceiling or sut_cfg.get("incontext_ceiling", False)
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.runner.s2_runner import S2Runner
    from agingbench.runner.trace import TraceLogger
    from agingbench.metrics.aging import summarize, compute_half_life, compute_decay_slope
    from agingbench.report.plot import compare_curves, plot_curve

    llm = load_llm(sut_cfg["model"])
    memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"

    n_sessions = n_cycles if n_cycles > 0 else scenario_cfg.get("n_cycles", 10)

    generated_data = None
    if generated:
        from agingbench.generators.s2_generator import S2Generator
        from agingbench.generators.pressure_config import PressureConfig
        gen_n = gen_sessions if gen_sessions > 0 else n_sessions
        generated_data = S2Generator(
            seed=sut_cfg.get("seed", 42),
            pressure=_resolve_pressure(sut_cfg, scenario_cfg),
            dense_accumulator=sut_cfg.get(
                "dense_accumulator",
                scenario_cfg.get("dense_accumulator", False),
            ),
        ).generate(gen_n)
        n_sessions = gen_n

    with TraceLogger(str(trace_path)) as tracer:
        agent_cls_kwarg = {"agent_class": agent_class} if agent_class else {}
        runner = S2Runner(
            memory_policy=memory_policy,
            llm=llm,
            tracer=tracer,
            sut_id=sut_cfg["sut_id"],
            oracle_mode=oracle_mode,
            oracle_retrieval=oracle_retrieval,
            oracle_store=oracle_store,
            incontext_ceiling=incontext_ceiling,
            ceiling_max_tokens=ceiling_max_tokens,
            generated_data=generated_data,
            **agent_cls_kwarg,
        )
        result = runner.run(
            n_sessions=n_sessions,
            seed=sut_cfg.get("seed", 42),
        )

    cvr_curve = result["cvr_curve"]
    tus_curve = result["tus_curve"]
    precision_curve = result["precision_curve"]
    lag_recall_curve = result["lag_recall_curve"]
    compounding_curve = result["compounding_curve"]

    # Headline aging metric for S2 is constraint_precision, not CVR.
    # CVR (regex violation detection) saturates at 0 for safety-tuned models
    # (Gemma 27B, Llama, etc.) — they hedge their way past violation patterns
    # even after losing the original constraint value from memory. Precision
    # measures specific knowledge of the constraint and decays monotonically
    # under compression — see runner/s2_runner.py:532.
    stats = summarize(precision_curve)
    stats["scenario"] = "s2_lifestyle_assistant"
    stats["metric_group"] = "G2"
    stats["headline_metric"] = "constraint_precision"
    stats["cvr_raw"] = result["cvr_raw"]
    stats["adherence_raw"] = result["adherence_raw"]
    stats["precision_raw"] = result["precision_raw"]
    stats["tus_raw"] = result["tus_raw"]
    stats["lag_recall_raw"] = result["lag_recall_raw"]
    stats["compounding_raw"] = result["compounding_raw"]
    stats["compounding_fresh_raw"] = result.get("compounding_fresh_raw", [])
    stats["session_results"] = result["session_results"]

    # Propagate attribution-schema provenance (v2_clean on all new runs via the
    # refactored S2 runner). Pre-2026-04-20 runs are flagged v1_conflated
    # retroactively by experiments/scripts/flag_attribution_schema_v1.py.
    if "attribution_schema" in result:
        stats["attribution_schema"] = result["attribution_schema"]
    if "attribution_mode" in result:
        stats["attribution_mode"] = result["attribution_mode"]
    if result.get("ceiling_max_tokens") is not None:
        stats["ceiling_max_tokens"] = result["ceiling_max_tokens"]

    if oracle_mode:
        stats["oracle_mode"] = True

    if generated_data and "dependency_graph" in generated_data:
        from agingbench.metrics.dependency_scorer import score_dependency_chain
        dep_metrics = score_dependency_chain(
            result.get("session_results", []), generated_data["dependency_graph"]
        )
        stats["dependency_metrics"] = dep_metrics
        with open(output_dir / "dependency_metrics.json", "w") as f:
            json.dump(dep_metrics, f, indent=2)

    # Response-token cap-confound diagnostics (if runner populated the fields)
    _tok_diag = _collect_response_token_diagnostics(
        stats.get("session_results", []), _infer_max_tokens(sut_cfg)
    )
    if _tok_diag:
        stats["response_token_diagnostics"] = _tok_diag

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    title = f"S2 Aging — {sut_cfg['sut_id']}"
    if oracle_mode:
        title += " (oracle)"

    # Label the [0,1] curves so axis auto-picks a sensible ylabel.
    for c, name in (
        (precision_curve, "constraint_precision"),
        (lag_recall_curve, "lag_recall"),
        (compounding_curve, "compounding_accuracy"),
        (cvr_curve, "constraint_adherence"),
    ):
        c.metric_name = name

    compare_curves(
        [precision_curve, lag_recall_curve, compounding_curve, cvr_curve],
        str(output_dir / "aging_curve.png"),
        title=title,
        labels=[
            "constraint_precision (headline)",
            "lag_recall",
            "compounding_accuracy",
            "constraint_adherence (1-CVR, ceiling-saturated)",
        ],
    )

    # Dual-axis headline: precision (Compression) + accumulator_error (Revision).
    # Supports Table 2's claim that S2 validates BOTH compression and revision
    # via its two headline metrics. accumulator_error is unbounded, so it
    # needs a twin axis rather than sharing the [0,1] scale above.
    dep_metrics = stats.get("dependency_metrics") or {}
    accum = dep_metrics.get("accumulator_metrics") or {}
    accum_errors = accum.get("accumulator_errors") or {}
    if accum_errors:
        from agingbench.report.plot import plot_dual_axis_curves
        accum_curve = _trajectory_curve(
            accum_errors, sut_cfg["sut_id"],
            "s2_lifestyle_assistant", "accumulator_error",
        )
        if accum_curve is not None:
            plot_dual_axis_curves(
                primary_curve=precision_curve,
                secondary_curve=accum_curve,
                output_path=str(output_dir / "aging_curve_dual.png"),
                title=f"{title} — precision (compression) vs accumulator_error (revision)",
                primary_label="constraint_precision",
                secondary_label="accumulator_error",
                primary_ylabel="Constraint precision m(t)",
                secondary_ylabel="Accumulator error (|agent − gold|)",
            )

    # Mechanism-coverage overlay (compression headline + revision + interference
    # trajectories from the DAG scorer). interference_resistance_per_session is
    # typically empty for S2 (no interference pairs) and is silently skipped.
    _emit_mechanism_aging_plot(
        primary_curve=precision_curve,
        dep_metrics=dep_metrics,
        output_dir=output_dir,
        sut_id=sut_cfg["sut_id"],
        scenario="s2_lifestyle_assistant",
        title=f"{title} — mechanism trajectories",
    )

    return stats


# ------------------------------------------------------------------ S3 runner

def _run_s3(sut_cfg: dict, scenario_cfg: dict, output_dir: Path,
            n_cycles: int, oracle_mode: bool = False, oracle_retrieval: bool = False,
            oracle_store: bool = False, incontext_ceiling: bool = False,
            ceiling_max_tokens: int = 100_000,
            agent_class=None,
            generated: bool = False, gen_sessions: int = 0) -> dict:
    """Execute S3 scenario (Project Knowledge Base Agent) and return metrics dict."""
    oracle_mode = oracle_mode or sut_cfg.get("oracle_mode", False)
    oracle_retrieval = oracle_retrieval or sut_cfg.get("oracle_retrieval", False)
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.runner.s3_runner import S3Runner
    from agingbench.runner.trace import TraceLogger
    from agingbench.metrics.aging import summarize
    from agingbench.report.plot import compare_curves

    llm = load_llm(sut_cfg["model"])
    memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"

    n_sessions = n_cycles if n_cycles > 0 else scenario_cfg.get("n_cycles", 12)

    generated_data = None
    if generated:
        from agingbench.generators.s3_generator import S3Generator
        from agingbench.generators.pressure_config import PressureConfig
        gen_n = gen_sessions if gen_sessions > 0 else n_sessions
        generated_data = S3Generator(seed=sut_cfg.get("seed", 42),
                                     pressure=_resolve_pressure(sut_cfg, scenario_cfg)).generate(gen_n)
        n_sessions = gen_n

    with TraceLogger(str(trace_path)) as tracer:
        agent_cls_kwarg = {"agent_class": agent_class} if agent_class else {}
        runner = S3Runner(
            memory_policy=memory_policy,
            llm=llm,
            tracer=tracer,
            sut_id=sut_cfg["sut_id"],
            oracle_mode=oracle_mode,
            oracle_retrieval=oracle_retrieval,
            generated_data=generated_data,
            **agent_cls_kwarg,
        )
        result = runner.run(
            n_sessions=n_sessions,
            seed=sut_cfg.get("seed", 42),
        )

    fidelity_curve = result["fidelity_curve"]

    stats = summarize(fidelity_curve)
    stats["scenario"] = "s3_knowledge_base"
    stats["metric_group"] = "G3"
    stats["headline_metric"] = "summarization_fidelity"
    stats["fidelity_raw"] = result["fidelity_raw"]
    stats["bloat_raw"] = result["bloat_raw"]
    stats["contradiction_raw"] = result["contradiction_raw"]
    stats["query_acc_raw"] = result["query_acc_raw"]
    stats["session_results"] = result["session_results"]

    if generated_data and "dependency_graph" in generated_data:
        from agingbench.metrics.dependency_scorer import score_dependency_chain
        dep_metrics = score_dependency_chain(
            result.get("session_results", []), generated_data["dependency_graph"]
        )
        stats["dependency_metrics"] = dep_metrics
        with open(output_dir / "dependency_metrics.json", "w") as f:
            json.dump(dep_metrics, f, indent=2)

    if oracle_mode:
        stats["oracle_mode"] = True

    # Response-token cap-confound diagnostics (if runner populated the fields)
    _tok_diag = _collect_response_token_diagnostics(
        stats.get("session_results", []), _infer_max_tokens(sut_cfg)
    )
    if _tok_diag:
        stats["response_token_diagnostics"] = _tok_diag

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    title = f"S3 Aging — {sut_cfg['sut_id']}"
    if oracle_mode:
        title += " (oracle)"

    fidelity_curve.metric_name = "summarization_fidelity"
    result["contradiction_curve"].metric_name = "consistency"
    result["query_curve"].metric_name = "query_accuracy"

    compare_curves(
        [fidelity_curve, result["contradiction_curve"], result["query_curve"]],
        str(output_dir / "aging_curve.png"),
        title=title,
        labels=["summarization_fidelity (headline)", "consistency (1-contradiction)", "query_accuracy"],
    )

    # Mechanism-coverage plot (Table 2 claims: compression + interference + revision)
    _emit_mechanism_aging_plot(
        primary_curve=fidelity_curve,
        dep_metrics=stats.get("dependency_metrics") or {},
        output_dir=output_dir,
        sut_id=sut_cfg["sut_id"],
        scenario="s3_knowledge_base",
        title=f"{title} — mechanism trajectories",
    )

    return stats


# ------------------------------------------------------------------ S4 runner

def _run_s4(sut_cfg: dict, scenario_cfg: dict, output_dir: Path,
            n_cycles: int, oracle_mode: bool = False, oracle_retrieval: bool = False,
            oracle_store: bool = False, incontext_ceiling: bool = False,
            ceiling_max_tokens: int = 100_000,
            agent_class=None,
            generated: bool = False, gen_sessions: int = 0) -> dict:
    """Execute S4 scenario (Software Engineering Agent) and return metrics dict."""
    oracle_mode = oracle_mode or sut_cfg.get("oracle_mode", False)
    oracle_retrieval = oracle_retrieval or sut_cfg.get("oracle_retrieval", False)
    oracle_store = oracle_store or sut_cfg.get("oracle_store", False)
    incontext_ceiling = incontext_ceiling or sut_cfg.get("incontext_ceiling", False)
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.runner.s4_runner import S4Runner
    from agingbench.runner.trace import TraceLogger
    from agingbench.metrics.aging import summarize
    from agingbench.report.plot import compare_curves

    llm = load_llm(sut_cfg["model"])
    memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"

    n_sessions = n_cycles if n_cycles > 0 else scenario_cfg.get("n_cycles", 8)

    generated_data = None
    if generated:
        from agingbench.generators.s4_generator import S4Generator
        from agingbench.generators.pressure_config import PressureConfig
        gen_n = gen_sessions if gen_sessions > 0 else n_sessions
        generated_data = S4Generator(seed=sut_cfg.get("seed", 42),
                                     pressure=_resolve_pressure(sut_cfg, scenario_cfg)).generate(gen_n)
        n_sessions = gen_n

    with TraceLogger(str(trace_path)) as tracer:
        agent_cls_kwarg = {"agent_class": agent_class} if agent_class else {}
        runner = S4Runner(
            memory_policy=memory_policy,
            llm=llm,
            tracer=tracer,
            sut_id=sut_cfg["sut_id"],
            oracle_mode=oracle_mode,
            oracle_retrieval=oracle_retrieval,
            oracle_store=oracle_store,
            incontext_ceiling=incontext_ceiling,
            ceiling_max_tokens=ceiling_max_tokens,
            generated_data=generated_data,
            **agent_cls_kwarg,
        )
        result = runner.run(
            n_sessions=n_sessions,
            seed=sut_cfg.get("seed", 42),
        )

    la_curve = result["la_curve"]
    dep_recall_curve = result["dep_recall_curve"]

    from agingbench.metrics.aging import AgingCurve
    # Headline aging metric for S4. We promote **dep_recall_faithful** — the
    # HELD-OUT dependency probe answered from compressed memory only (no
    # dep_context in the prompt) — to the headline, because the legacy
    # dep_recall scores the agent against dep_context that is re-injected into
    # the task prompt each sprint, i.e. prompt-attention rather than memory
    # recall (see test_s4_dep_recall_faithful). dep_recall and la are retained
    # as reported secondaries. Falls back to dep_recall when no faithful probe
    # fired (e.g. dependency_density/warmup left the curve empty).
    _faithful = result.get("dep_recall_faithful_raw") or []
    if _faithful:
        faithful_curve = AgingCurve(
            exposures=[int(e) for e, _ in _faithful],
            scores=[float(s) for _, s in _faithful],
            scenario="s4_software_engineering",
            sut_id=sut_cfg["sut_id"],
            metric_name="dep_recall_faithful",
        )
        primary_s4_curve = faithful_curve
        stats = summarize(faithful_curve)
        stats["headline_metric"] = "dep_recall_faithful"
    else:
        primary_s4_curve = dep_recall_curve
        stats = summarize(dep_recall_curve)
        stats["headline_metric"] = "dep_recall"
    stats["scenario"] = "s4_software_engineering"
    stats["metric_group"] = "G4"
    stats["la_raw"] = result["la_raw"]
    stats["cfr_raw"] = result["cfr_raw"]
    stats["fasr_raw"] = result["fasr_raw"]
    stats["rr_raw"] = result["rr_raw"]
    stats["task_m_raw"] = result["task_m_raw"]
    stats["dep_recall_raw"] = result["dep_recall_raw"]
    # Faithful per-probe dep_recall (LLM answers held-out questions using only
    # compressed memory_text — no dep_context). Sparse: present only at sessions
    # where dependency_density fired a probe. Side-car metric for empirical
    # comparison against the substring-on-dep_context proxy reported above.
    _faithful = result.get("dep_recall_faithful_raw") or []
    stats["dep_recall_faithful_raw"] = _faithful
    if _faithful:
        _faithful_scores = [s for _, s in _faithful]
        stats["dep_recall_faithful_mean"] = round(
            sum(_faithful_scores) / len(_faithful_scores), 4
        )
        stats["dep_recall_faithful_final"] = round(_faithful_scores[-1], 4)
        stats["dep_recall_faithful_n_probes"] = len(_faithful_scores)
    stats["session_results"] = result["session_results"]

    if generated_data and "dependency_graph" in generated_data:
        from agingbench.metrics.dependency_scorer import score_dependency_chain
        dep_metrics = score_dependency_chain(
            result.get("session_results", []), generated_data["dependency_graph"]
        )
        stats["dependency_metrics"] = dep_metrics
        with open(output_dir / "dependency_metrics.json", "w") as f:
            json.dump(dep_metrics, f, indent=2)

    if result.get("life_event"):
        stats["life_event"] = result["life_event"]

    # Attribution provenance (see s4_runner.run return dict).
    if "attribution_schema" in result:
        stats["attribution_schema"] = result["attribution_schema"]
    if "attribution_mode" in result:
        stats["attribution_mode"] = result["attribution_mode"]
    if result.get("ceiling_max_tokens") is not None:
        stats["ceiling_max_tokens"] = result["ceiling_max_tokens"]

    if oracle_mode:
        stats["oracle_mode"] = True

    # Response-token cap-confound diagnostics (if runner populated the fields)
    _tok_diag = _collect_response_token_diagnostics(
        stats.get("session_results", []), _infer_max_tokens(sut_cfg)
    )
    if _tok_diag:
        stats["response_token_diagnostics"] = _tok_diag

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    title = f"S4 Aging — {sut_cfg['sut_id']}"
    if oracle_mode:
        title += " (oracle)"

    dep_recall_curve.metric_name = "dep_recall"
    la_curve.metric_name = "lookahead_accuracy"
    result["cfr_curve"].metric_name = "test_stability"
    result["fasr_curve"].metric_name = "first_attempt_success"

    # Plot the faithful (held-out memory) curve as headline when available,
    # with dep_recall (prompt-echo proxy) and la as secondaries.
    if _faithful:
        _curves = [primary_s4_curve, dep_recall_curve, la_curve, result["fasr_curve"]]
        _labels = [
            "dep_recall_faithful (headline)",
            "dep_recall (prompt-echo)",
            "lookahead_accuracy",
            "first_attempt_success",
        ]
    else:
        _curves = [dep_recall_curve, la_curve, result["cfr_curve"], result["fasr_curve"]]
        _labels = [
            "dep_recall (headline)",
            "lookahead_accuracy",
            "test_stability (1-CFR)",
            "first_attempt_success",
        ]
    compare_curves(_curves, str(output_dir / "aging_curve.png"),
                   title=title, labels=_labels)

    # Mechanism-coverage plot (Table 2 claims: compression + interference)
    _emit_mechanism_aging_plot(
        primary_curve=primary_s4_curve,
        dep_metrics=stats.get("dependency_metrics") or {},
        output_dir=output_dir,
        sut_id=sut_cfg["sut_id"],
        scenario="s4_software_engineering",
        title=f"{title} — mechanism trajectories",
    )

    return stats


# ------------------------------------------------------------------ S6 runner

def _run_s6(sut_cfg: dict, scenario_cfg: dict, output_dir: Path,
            n_cycles: int,
            diagnose: bool = False,
            agent_class=None,
            generated: bool = False, gen_sessions: int = 0,
            **_legacy_kwargs) -> dict:
    """Execute S6 scenario (Naturalistic Aging — WebArena-derived) and return metrics dict."""
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.runner.s6_runner import S6Runner
    from agingbench.runner.trace import TraceLogger
    from agingbench.metrics.aging import summarize, compute_half_life, compute_decay_slope
    from agingbench.report.plot import compare_curves

    llm = load_llm(sut_cfg["model"])
    memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"

    n_sessions = n_cycles if n_cycles > 0 else scenario_cfg.get("n_cycles", 15)

    generated_data = None
    if generated:
        from agingbench.generators.s6_generator import S6Generator
        from agingbench.generators.pressure_config import PressureConfig
        gen_n = gen_sessions if gen_sessions > 0 else n_sessions
        generated_data = S6Generator(seed=sut_cfg.get("seed", 42),
                                     pressure=_resolve_pressure(sut_cfg, scenario_cfg)).generate(gen_n)
        n_sessions = gen_n

    # Load maintenance events from SUT config (empty list if not configured)
    from agingbench.core.maintenance import load_maintenance_config
    maintenance_events = load_maintenance_config(sut_cfg)

    with TraceLogger(str(trace_path)) as tracer:
        agent_cls_kwarg = {"agent_class": agent_class} if agent_class else {}
        runner = S6Runner(
            memory_policy=memory_policy,
            llm=llm,
            tracer=tracer,
            sut_id=sut_cfg["sut_id"],
            diagnose=diagnose,
            generated_data=generated_data,
            maintenance_events=maintenance_events,
            **agent_cls_kwarg,
        )
        result = runner.run(
            n_sessions=n_sessions,
            seed=sut_cfg.get("seed", 42),
        )

    task_curve = result["task_curve"]
    recall_curve = result["recall_curve"]

    stats = summarize(recall_curve)
    stats["scenario"] = "s6_naturalistic"
    stats["metric_group"] = "G1"
    stats["headline_metric"] = "recall_rate"
    stats["task_raw"] = result["task_raw"]
    stats["recall_raw"] = result["recall_raw"]
    stats["recall_matrix"] = {
        str(k): v for k, v in result["recall_matrix"].items()
    }
    stats["lag_curves"] = {
        str(k): v for k, v in result["lag_curves"].items()
    }
    stats["session_results"] = result["session_results"]

    # Diagnostic error partitioning results (when --diagnose was used).
    if "diagnostic_partition" in result:
        stats["diagnostic_partition"] = result["diagnostic_partition"]
        stats["diagnostic_per_session"] = result["diagnostic_per_session"]
        with open(output_dir / "diagnostic_partition.json", "w") as f:
            json.dump(result["diagnostic_partition"], f, indent=2)

    if generated_data and "dependency_graph" in generated_data:
        from agingbench.metrics.dependency_scorer import score_dependency_chain
        dep_metrics = score_dependency_chain(
            result.get("session_results", []), generated_data["dependency_graph"]
        )
        stats["dependency_metrics"] = dep_metrics
        with open(output_dir / "dependency_metrics.json", "w") as f:
            json.dump(dep_metrics, f, indent=2)

    _tok_diag = _collect_response_token_diagnostics(
        stats.get("session_results", []), _infer_max_tokens(sut_cfg)
    )
    if _tok_diag:
        stats["response_token_diagnostics"] = _tok_diag

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    title = f"S6 Naturalistic Aging — {sut_cfg['sut_id']}"
    if diagnose:
        title += " (P1/P2/P3 diagnostics)"

    recall_curve.metric_name = "recall_rate"
    task_curve.metric_name = "task_accuracy"

    shock_sessions = []
    for sr in result.get("session_results", []):
        ev = sr.get("maintenance_event") or sr.get("life_event")
        if ev:
            shock_sessions.append(sr.get("session", -1))
    shock_sessions = [s for s in shock_sessions if s >= 0]

    compare_curves(
        [recall_curve, task_curve],
        str(output_dir / "aging_curve.png"),
        title=title,
        labels=["recall_rate (headline)", "task_accuracy (compensated)"],
        shock_sessions=shock_sessions or None,
    )

    _emit_mechanism_aging_plot(
        primary_curve=recall_curve,
        dep_metrics=stats.get("dependency_metrics") or {},
        output_dir=output_dir,
        sut_id=sut_cfg["sut_id"],
        scenario="s6_naturalistic",
        title=f"{title} — mechanism trajectories",
        shock_sessions=shock_sessions or None,
    )

    return stats




# ------------------------------------------------------------------ Self-planning runner

def _run_self_planning(sut_cfg: dict, scenario_cfg: dict, output_dir: Path,
                       n_cycles: int, oracle_mode: bool = False, oracle_retrieval: bool = False,
                       agent_class=None,
                       generated: bool = False, gen_sessions: int = 0) -> dict:
    """Execute a self-planning scenario variant."""
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.runner.self_planning_runner import SelfPlanningRunner
    from agingbench.runner.trace import TraceLogger
    from agingbench.metrics.aging import summarize
    from agingbench.report.plot import compare_curves

    # Determine which scenario is wrapped
    sid = scenario_cfg["scenario_id"]
    wrapped = sid.replace("self_planning_", "")  # "s2", "s3", "s6"

    llm = load_llm(sut_cfg["model"])
    memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"

    n_sessions = n_cycles if n_cycles > 0 else scenario_cfg.get("n_cycles", 10)

    # Generate or load data
    generated_data = None
    if generated or True:  # self-planning always needs explicit data
        gen_map = {
            "s2": ("agingbench.generators.s2_generator", "S2Generator"),
            "s3": ("agingbench.generators.s3_generator", "S3Generator"),
            "s6": ("agingbench.generators.s6_generator", "S6Generator"),
        }
        if wrapped in gen_map:
            mod_path, cls_name = gen_map[wrapped]
            mod = importlib.import_module(mod_path)
            gen_cls = getattr(mod, cls_name)
            gen_n = gen_sessions if gen_sessions > 0 else n_sessions
            generated_data = gen_cls(seed=sut_cfg.get("seed", 42)).generate(gen_n)
            n_sessions = gen_n

    with TraceLogger(str(trace_path)) as tracer:
        runner = SelfPlanningRunner(
            wrapped_scenario=wrapped,
            memory_policy=memory_policy,
            llm=llm,
            tracer=tracer,
            sut_id=sut_cfg["sut_id"],
            oracle_mode=oracle_mode,
            agent_class=agent_class or __import__("agingbench.core.agent", fromlist=["ReferenceAgent"]).ReferenceAgent,
            generated_data=generated_data,
        )
        result = runner.run(n_sessions=n_sessions, seed=sut_cfg.get("seed", 42))

    recall_curve = result["recall_curve"]
    task_curve = result["task_curve"]

    # Headline aging metric for S7 is recall_accuracy. task_accuracy is
    # floor-saturated for current models (starting between 0.07 and 0.47
    # across the runs we collected) and several models even show task
    # accuracy *increasing* with session, which is sampling noise rather
    # than aging. Do NOT use task_accuracy as an aging signal for S7;
    # treat it as a difficulty indicator only.
    stats = summarize(recall_curve)
    stats["scenario"] = f"self_planning_{wrapped}"
    stats["metric_group"] = "G1"
    stats["headline_metric"] = "recall_accuracy"
    stats["task_accuracy_note"] = (
        "task_accuracy is floor-saturated for current models and is NOT "
        "a valid aging signal for S7; reported only as a difficulty indicator."
    )
    stats["task_raw"] = result["task_raw"]
    stats["recall_raw"] = result["recall_raw"]
    stats["recall_matrix"] = {str(k): v for k, v in result["recall_matrix"].items()}
    stats["lag_curves"] = {str(k): v for k, v in result["lag_curves"].items()}
    stats["session_results"] = result["session_results"]

    # Response-token cap-confound diagnostics (if runner populated the fields)
    _tok_diag = _collect_response_token_diagnostics(
        stats.get("session_results", []), _infer_max_tokens(sut_cfg)
    )
    if _tok_diag:
        stats["response_token_diagnostics"] = _tok_diag

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    title = f"Self-Planning {wrapped.upper()} — {sut_cfg['sut_id']}"
    compare_curves(
        [recall_curve, task_curve],
        str(output_dir / "aging_curve.png"),
        title=title,
        labels=["recall_accuracy (headline)", "task_accuracy (floor-saturated)"],
    )

    return stats


# ------------------------------------------------------------------ Dynamic runner

def _run_dynamic(runner_cls, sut_cfg, scenario_cfg, output_dir, n_sessions, oracle=False, **kwargs):
    """Generic runner dispatch for manifest-declared runner classes."""
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.runner.trace import TraceLogger

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = load_llm(sut_cfg.get("model", {}))
    memory = build_memory_policy(sut_cfg.get("memory_policy", {"type": "no_memory"}),
                                  project_root=PROJECT_ROOT)
    tracer = TraceLogger(str(output_dir / "trace.jsonl"))

    runner = runner_cls(
        llm=llm,
        memory_policy=memory,
        tracer=tracer,
        sut_id=sut_cfg.get("sut_id", "unknown"),
        oracle_mode=oracle,
    )
    result = runner.run(n_sessions=n_sessions, seed=sut_cfg.get("seed", 42))
    tracer.close()

    curve = result.get("recall_curve") or result.get("task_curve")
    stats = {"m0": 1.0, "m_final": 0.0, "decay_slope": 0.0}
    if curve and hasattr(curve, "scores") and curve.scores:
        stats["m0"] = curve.scores[0]
        stats["m_final"] = curve.scores[-1]
        from agingbench.metrics.aging import compute_decay_slope
        stats["decay_slope"] = round(compute_decay_slope(curve), 5)
    return stats


# ---- Scenario dispatch table ----

# ------------------------------------------------------------------ S5 runner

def _run_s5(sut_cfg: dict, scenario_cfg: dict, output_dir: Path,
            n_cycles: int, oracle_mode: bool = False, oracle_retrieval: bool = False,
            oracle_store: bool = False, incontext_ceiling: bool = False,
            ceiling_max_tokens: int = 100_000,
            agent_class=None,
            generated: bool = False, gen_sessions: int = 0) -> dict:
    """Execute S5 Self-Planning Notebook and return metrics dict.

    Renamed from S7 self_planning in v0.2.x. The runner + generator
    module names mirror the new scenario_id (s5_*).
    """
    from agingbench.core.llm import load_llm
    from agingbench.core.adapters.react_file_adapter import ReactFileAdapter
    from agingbench.runner.s5_runner import S5Runner
    from agingbench.runner.trace import TraceLogger
    from agingbench.generators.s5_generator import S5Generator
    from agingbench.generators.pressure_config import PressureConfig
    from agingbench.metrics.aging import summarize
    from agingbench.metrics.dependency_scorer import score_dependency_chain
    from agingbench.report.plot import compare_curves

    n_sessions = n_cycles if n_cycles > 0 else scenario_cfg.get("n_cycles", 8)
    gen_n = gen_sessions if gen_sessions > 0 else n_sessions

    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = str(output_dir / "workspace")

    adapter_cfg = sut_cfg.get("adapter", {})
    adapter_type = adapter_cfg.get("type", "react")

    if adapter_type == "react":
        llm = load_llm(sut_cfg["model"])
        max_turns = adapter_cfg.get("max_turns", 8)
        adapter = ReactFileAdapter(llm=llm, workspace_dir=workspace_dir, max_turns=max_turns)
    elif adapter_type == "claude_code":
        from agingbench.core.adapters.claude_code_agent_adapter import ClaudeCodeAgentAdapter
        model = adapter_cfg.get("model", "claude-sonnet-4-6-20250514")
        max_turns = adapter_cfg.get("max_turns", 50)
        cli_path = adapter_cfg.get("cli_path", "claude")
        adapter = ClaudeCodeAgentAdapter(
            model=model, cwd=workspace_dir, max_turns=max_turns, cli_path=cli_path,
        )
    elif adapter_type == "codex":
        from agingbench.core.adapters.codex_adapter import CodexAdapter
        model = adapter_cfg.get("model", "codex-mini")
        max_turns = adapter_cfg.get("max_turns", 25)
        cli_path = adapter_cfg.get("cli_path", "codex")
        adapter = CodexAdapter(
            model=model, cwd=workspace_dir, max_turns=max_turns, cli_path=cli_path,
        )
    elif adapter_type == "openhands":
        from agingbench.core.adapters.openhands_adapter import OpenHandsAdapter
        adapter = OpenHandsAdapter(
            model=adapter_cfg.get("model", "gpt-4o-mini"),
            cwd=workspace_dir,
            max_turns=adapter_cfg.get("max_turns", 30),
            bridge_python=adapter_cfg.get("bridge_python"),
            system_prompt=adapter_cfg.get("system_prompt"),
            api_key_env=adapter_cfg.get("api_key_env", "OPENAI_API_KEY"),
            reasoning_effort=adapter_cfg.get("reasoning_effort"),
            preset=adapter_cfg.get("preset"),
            subprocess_timeout=adapter_cfg.get("subprocess_timeout", 1800),
        )
    elif adapter_type == "custom":
        from agingbench.core.agent_adapter import build_custom_adapter
        adapter = build_custom_adapter(adapter_cfg, workspace_dir)
    else:
        raise ValueError(f"S7 CLI runner: unknown adapter type '{adapter_type}'")

    domain = scenario_cfg.get("domain", "assistant")
    gen = S5Generator(seed=sut_cfg.get("seed", 42), domain=domain,
                      pressure=_resolve_pressure(sut_cfg, scenario_cfg))
    generated_data = gen.generate(n_sessions=gen_n)
    n_sessions = gen_n

    # Load maintenance events (workspace_flush, workspace_recompact, etc.)
    from agingbench.core.maintenance import load_maintenance_config
    maintenance_events = load_maintenance_config(sut_cfg)

    trace_path = output_dir / "trace.jsonl"

    with TraceLogger(str(trace_path)) as tracer:
        runner = S5Runner(
            adapter=adapter,
            tracer=tracer,
            sut_id=sut_cfg["sut_id"],
            session_length=scenario_cfg.get("session_length", 10),
            generated_data=generated_data,
            reset_history=True,
            maintenance_events=maintenance_events,
        )
        result = runner.run(n_sessions=n_sessions, seed=sut_cfg.get("seed", 42))

    stats = summarize(result.primary_curve)
    stats["scenario"] = "s5_self_planning"
    stats["sut_id"] = sut_cfg["sut_id"]
    stats["metric_group"] = "G1"
    stats["headline_metric"] = "recall_accuracy"
    # task_accuracy is floor-saturated for current models on S7 (starts in
    # the 0.07-0.47 range and several models show task accuracy *increasing*
    # across sessions, which is sampling noise rather than aging). Do NOT
    # use task_accuracy as an aging signal for S7; treat it as a difficulty
    # indicator only. The runner still measures it and we still save it.
    stats["task_accuracy_note"] = (
        "task_accuracy is floor-saturated for current models and is NOT "
        "a valid aging signal for S7; reported only as a difficulty indicator."
    )
    stats["adapter"] = adapter_type
    stats["domain"] = domain
    stats["n_sessions"] = n_sessions
    stats["recall_raw"] = result.raw.get("recall_raw", [])
    stats["task_raw"] = result.raw.get("task_raw", [])
    stats["recall_matrix"] = result.raw.get("recall_matrix", {})
    stats["lag_curves"] = result.raw.get("lag_curves", {})
    # workspace_snapshots is in result.raw (used by Fig 1c interference
    # scatter); previously dropped before save. Older S7 files have this
    # field at top level; restore it.
    stats["workspace_snapshots"] = result.raw.get("workspace_snapshots", [])
    stats["session_results"] = result.session_results

    if generated_data and "dependency_graph" in generated_data:
        dep_metrics = score_dependency_chain(
            result.session_results, generated_data["dependency_graph"]
        )
        stats["dependency_metrics"] = dep_metrics
        with open(output_dir / "dependency_metrics.json", "w") as f:
            json.dump(dep_metrics, f, indent=2)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    curves = [result.primary_curve]
    result.primary_curve.metric_name = "recall_accuracy"
    labels = ["recall_accuracy (headline)"]
    if "task_accuracy" in (result.secondary_curves or {}):
        curves.append(result.secondary_curves["task_accuracy"])
        result.secondary_curves["task_accuracy"].metric_name = "task_accuracy"
        labels.append("task_accuracy")

    # Extract maintenance shock sessions (workspace_flush / workspace_recompact)
    shock_sessions = []
    for sr in result.session_results:
        ev = sr.get("maintenance_event") or sr.get("life_event")
        if ev:
            shock_sessions.append(sr.get("session", -1))
    shock_sessions = [s for s in shock_sessions if s >= 0]

    title_s7 = f"S7 Aging — {sut_cfg['sut_id']}"
    compare_curves(
        curves, str(output_dir / "aging_curve.png"),
        title=title_s7, labels=labels,
        shock_sessions=shock_sessions or None,
    )

    # Mechanism-coverage plot (Table 2: compression + interference + revision,
    # plus maintenance via shock_sessions markers when maintenance_events active)
    _emit_mechanism_aging_plot(
        primary_curve=result.primary_curve,
        dep_metrics=stats.get("dependency_metrics") or {},
        output_dir=output_dir,
        sut_id=sut_cfg["sut_id"],
        scenario="s5_self_planning",
        title=f"{title_s7} — mechanism trajectories",
        shock_sessions=shock_sessions or None,
    )

    return stats


def _run_s7(sut_cfg, scenario_cfg, output_dir, n_cycles, *,
                oracle_mode=False, oracle_retrieval=False, self_plan=False,
                oracle_store=False, incontext_ceiling=False,
                ceiling_max_tokens=100_000,
                agent_class=None, generated=False, gen_sessions=0, **kwargs):
    """S7+ dispatch: Tier-2 adapter on the scripted research-notes coding task."""
    import json as _json
    import subprocess as _sp
    from pathlib import Path as _Path
    from ..runner.trace import TraceLogger
    from ..generators.s7_generator import S7Generator
    from ..runner.s7_runner import S7Runner

    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = output_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    adapter_cfg = sut_cfg.get("adapter", {})
    adapter_type = adapter_cfg.get("type", "openhands")
    if adapter_type == "openhands":
        from ..core.adapters.openhands_adapter import OpenHandsAdapter
        adapter = OpenHandsAdapter(
            model=adapter_cfg.get("model", "gpt-4o-mini"),
            cwd=str(workspace_dir),
            max_turns=adapter_cfg.get("max_turns", 30),
            bridge_python=adapter_cfg.get("bridge_python"),
            system_prompt=adapter_cfg.get("system_prompt"),
            api_key_env=adapter_cfg.get("api_key_env", "OPENAI_API_KEY"),
            reasoning_effort=adapter_cfg.get("reasoning_effort"),
            preset=adapter_cfg.get("preset"),
            subprocess_timeout=adapter_cfg.get("subprocess_timeout", 1800),
        )
    elif adapter_type == "claude_code":
        from ..core.adapters.claude_code_agent_adapter import ClaudeCodeAgentAdapter
        # Isolate the workspace OUTSIDE the AgingBench repo to prevent
        # Claude Code's upward-CLAUDE.md traversal from reading our
        # benchmark-defining files. Auto-memory (skills, ~/.claude/, hooks)
        # stays enabled — we want realistic production behavior.
        if adapter_cfg.get("isolated_workspace", True):
            import tempfile as _tf
            isolated_ws = _Path(_tf.mkdtemp(prefix="aging_cc_ws_"))
            print(f"  [S7+] Claude Code workspace isolated at {isolated_ws}")
            effective_workspace = isolated_ws
        else:
            effective_workspace = _Path(str(workspace_dir))
        adapter = ClaudeCodeAgentAdapter(
            model=adapter_cfg.get("model", "claude-sonnet-4-6-20250514"),
            cwd=str(effective_workspace),
            max_turns=adapter_cfg.get("max_turns", 50),
            cli_path=adapter_cfg.get("cli_path", "claude"),
        )
        # Re-point workspace_dir so pytest + probe scoring target the
        # isolated dir where the agent's files actually live.
        workspace_dir = effective_workspace
    elif adapter_type == "cursor":
        from ..core.adapters.cursor_agent_adapter import CursorAgentAdapter
        if adapter_cfg.get("isolated_workspace", True):
            import tempfile as _tf
            isolated_ws = _Path(_tf.mkdtemp(prefix="aging_cursor_ws_"))
            print(f"  [S7+] Cursor workspace isolated at {isolated_ws}")
            effective_workspace = isolated_ws
        else:
            effective_workspace = _Path(str(workspace_dir))
        adapter = CursorAgentAdapter(
            model=adapter_cfg.get("model", "composer-2"),
            cwd=str(effective_workspace),
            cli_path=adapter_cfg.get("cli_path", "agent"),
            max_turns=adapter_cfg.get("max_turns", 50),
            timeout_sec=adapter_cfg.get("subprocess_timeout", 600),
            sandbox=adapter_cfg.get("sandbox"),
        )
        workspace_dir = effective_workspace
    elif adapter_type == "custom":
        from agingbench.core.agent_adapter import build_custom_adapter
        adapter = build_custom_adapter(adapter_cfg, workspace_dir)
    else:
        raise ValueError(f"S7+ runner: unknown adapter type '{adapter_type}'")

    n_sessions = gen_sessions if gen_sessions > 0 else (n_cycles if n_cycles > 0 else 5)
    scenario_path = output_dir / "scenario.json"
    if scenario_path.exists():
        data = _json.loads(scenario_path.read_text())
        print(f"  [S7+] Loaded existing scenario from {scenario_path}")
    else:
        gen = S7Generator(
            seed=sut_cfg.get("seed", 42),
            pressure=_resolve_pressure(sut_cfg, scenario_cfg),
        )
        data = gen.generate(n_sessions=n_sessions)
        scenario_path.write_text(_json.dumps(data, indent=2))

    # Data-driven pytest: when procedural blocks (>=10) emitted test specs,
    # write them out and point the generated-tests module at them. Curriculum-
    # only runs (n<=10) have no specs, so the env var stays unset and
    # tests/test_generated.py contributes nothing.
    import os as _os
    _gen_tests = data.get("generated_tests")
    if _gen_tests:
        _gt_path = output_dir / "generated_tests.json"
        _gt_path.write_text(_json.dumps(_gen_tests))
        _os.environ["AGINGBENCH_S7_GENERATED_TESTS"] = str(_gt_path)
    else:
        _os.environ.pop("AGINGBENCH_S7_GENERATED_TESTS", None)

    tests_dir = _Path(__file__).parent.parent / "scenarios" / "s7_research_notes" / "tests"

    trace_path = output_dir / "trace.jsonl"
    tracer = TraceLogger(trace_path)

    runner = S7Runner(
        adapter=adapter, tracer=tracer, sut_id=sut_cfg["sut_id"],
        generated_data=data, tests_dir=tests_dir, workspace_dir=workspace_dir,
        snapshots_dir=output_dir / "snapshots",
        archive_dir=(output_dir / "workspace_archive"
                     if adapter_type in ("claude_code", "cursor") and
                        adapter_cfg.get("isolated_workspace", True)
                     else None),
        checkpoint_dir=output_dir,
    )
    result = runner.run(n_sessions=n_sessions, seed=sut_cfg.get("seed", 42))
    runner.archive_workspace_if_set()
    result.setdefault("scenario", "s7_research_notes")
    result.setdefault("sut_id", sut_cfg["sut_id"])
    result.setdefault("metric_group", "G1")
    result.setdefault("headline_metric", "recall_accuracy")
    result.setdefault("run_status", "complete")
    (output_dir / "metrics.json").write_text(_json.dumps(result, indent=2))
    print(f"  m0={result['m0']:.3f}  m_final={result['m_final']:.3f}  "
          f"half_life={result['half_life']}  slope={result['decay_slope']:.5f}")
    return result


def _run_s8(sut_cfg: dict, scenario_cfg: dict, output_dir: Path,
            n_cycles: int, *, oracle_mode: bool = False,
            agent_class=None, generated: bool = False, gen_sessions: int = 0) -> dict:
    """S8 SWE-bench-Aging dispatcher.

    Two execution modes:

      1. **Real-runner mode** (preferred): SUT yaml has both an
         `agent.adapter` block AND `docker.enabled` is not False.
         Spins up the S8SweBenchRunner, which orchestrates per-session
         Docker containers, mounts /agentmemory, applies lifecycle
         events from PressureConfig, and (in Phases 3+) hands off to
         the real agent + verifier.

      2. **Stub mode** (CI / smoke): SUT yaml has no agent block, OR
         Docker isn't reachable, OR scenario_cfg["force_stub"]=True.
         Returns a metrics.json that records pressure flow + chain
         metadata but doesn't actually invoke containers.

    Phase 2 contract: real-runner mode lands containers + lifecycle
    events + memory persistence. Aging curve / mechanism probes /
    AgingCard population land in Phases 4-5.
    """
    import json

    from agingbench.generators.s8_swe_bench_generator import (
        S8SweBenchGenerator,
    )
    from agingbench.scenarios.s8_swe_bench.docker_runner import docker_available

    gen_n = gen_sessions if gen_sessions > 0 else n_cycles
    seed = int(sut_cfg.get("seed", 42))
    pressure = _resolve_pressure(sut_cfg, scenario_cfg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mode selection.
    has_agent_block = bool((sut_cfg.get("agent") or {}).get("adapter"))
    docker_ok = docker_available() if has_agent_block else False
    force_stub = bool(scenario_cfg.get("force_stub"))
    real_mode = has_agent_block and docker_ok and not force_stub

    if real_mode:
        return _run_s8_real(sut_cfg, scenario_cfg, output_dir, gen_n, seed, pressure)
    return _run_s8_stub(sut_cfg, scenario_cfg, output_dir, gen_n, seed, pressure,
                        docker_ok=docker_ok, has_agent_block=has_agent_block)


def _run_s8_stub(sut_cfg, scenario_cfg, output_dir, gen_n, seed, pressure,
                 *, docker_ok, has_agent_block) -> dict:
    """Phase-0 stub path: no containers, just dispatch contract."""
    import json
    from agingbench.generators.s8_swe_bench_generator import (
        S8SweBenchGenerator,
    )

    gen_data = S8SweBenchGenerator(seed=seed, pressure=pressure).generate(gen_n)

    with (output_dir / "session_issues.json").open("w") as f:
        json.dump(gen_data["session_issues"], f, indent=2, default=str)
    with (output_dir / "dependency_graph.json").open("w") as f:
        json.dump(gen_data["dependency_graph"], f, indent=2, default=str)
    with (output_dir / "lifecycle_events.json").open("w") as f:
        json.dump(gen_data["lifecycle_events"], f, indent=2)

    reason = (
        "force_stub=True"
        if scenario_cfg.get("force_stub")
        else ("no agent.adapter in SUT yaml" if not has_agent_block
              else "Docker daemon not reachable")
    )
    stats = {
        "scenario": "s8_swe_bench",
        "sut_id": sut_cfg.get("sut_id", "unknown"),
        "metric_group": scenario_cfg.get("metric_group", "G1"),
        "headline_metric": None,
        "seed": seed,
        "n_sessions": gen_n,
        "scaffold_status": (
            f"S8 SWE-bench-Aging stub mode ({reason}). Set "
            "agent.adapter in the SUT yaml + ensure Docker is running "
            "to engage the real-runner mode (Phase 2)."
        ),
        "pressure_used": gen_data["pressure_used"],
        "chain_used": gen_data.get("chain_used"),
        "lifecycle_events": gen_data["lifecycle_events"],
        "dependency_edge_count": len(gen_data["dependency_graph"]["dependency_edges"]),
        "interference_pair_count": len(gen_data["dependency_graph"]["interference_pairs"]),
        "phase": "phase_0_stub",
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(stats, f, indent=2)
    return stats


def _run_s8_real(sut_cfg, scenario_cfg, output_dir, gen_n, seed, pressure) -> dict:
    """Phase-2 real-runner path: spins containers + applies lifecycle events.

    Shared by S8 (sphinx + pytest chains) AND S9 (django chain) — the
    runner pipeline is identical, only chain_id + scenario_dir differ.
    """
    import json
    from agingbench.runner.s8_runner import S8SweBenchRunner, S8RunnerConfig

    # Determine which scenario this run is for (S8 vs S9) and pull its
    # default chain. scenario_cfg["scenario_id"] is the routing key.
    scenario_id = (scenario_cfg.get("scenario_id")
                   or sut_cfg.get("scenario_default")
                   or "s8_swe_bench")
    manifest = _SCENARIO_MANIFESTS.get(scenario_id) or {}
    data = manifest.get("data") or {}
    chain_id = (
        scenario_cfg.get("chain_id")
        or sut_cfg.get("chain_id")
        or data.get("default_chain", "django_orm_query")
    )
    image_pattern = data.get(
        "docker_image_pattern", "sweb.eval.x86_64.{instance_id}:latest"
    )

    cfg = S8RunnerConfig(
        seed=seed,
        n_sessions=gen_n,
        pressure=pressure,
        chain_id=chain_id,
        sut_id=sut_cfg.get("sut_id", "unknown"),
        docker_image_pattern=image_pattern,
        workspace_root=output_dir / "workspace",
        sut_cfg=sut_cfg,                          # Phase 3: agent.adapter / agent.model
    )
    runner = S8SweBenchRunner(cfg)
    pre = runner.precondition_check()
    if not pre["all_images_present"]:
        # Don't crash — degrade gracefully to stub mode with a useful note.
        scenario_cfg = dict(scenario_cfg)
        scenario_cfg["force_stub"] = True
        scenario_cfg["_missing_images"] = pre["missing_images"]
        return _run_s8_stub(sut_cfg, scenario_cfg, output_dir, gen_n, seed, pressure,
                            docker_ok=True, has_agent_block=True)

    result = runner.run()

    with (output_dir / "session_results.json").open("w") as f:
        json.dump(result["session_results"], f, indent=2, default=str)
    with (output_dir / "lifecycle_events.json").open("w") as f:
        json.dump(result["lifecycle_events_planned"], f, indent=2)

    # ---- Phase 4: aging curve + 4-mechanism probes -----
    headline, mechanism, dep_metrics = _s8_compute_aging_curve(
        result, output_dir,
    )

    stats = {
        "scenario": "s8_swe_bench",
        "sut_id": sut_cfg.get("sut_id", "unknown"),
        "metric_group": scenario_cfg.get("metric_group", "G1"),
        "headline_metric": "task_pass_rate",
        "seed": seed,
        "n_sessions": result["n_sessions"],
        "scaffold_status": (
            "S8 SWE-bench-Aging Phase 4: real-agent + Docker verifier + "
            "4-mechanism probes wired. Aging curve = per-session "
            "task_pass_rate against SWE-bench's FAIL_TO_PASS + "
            "PASS_TO_PASS tests; mechanism breakdowns computed from "
            "artifacts (notes, diffs, lifecycle events)."
        ),
        "pressure_used": result["pressure_used"],
        "chain_id": result["chain_id"],
        "workspace_root": result["workspace_root"],
        "n_lifecycle_events_planned": len(result["lifecycle_events_planned"]),
        "n_sessions_with_container_started": sum(
            1 for s in result["session_results"] if s["container_started"]
        ),
        "phase": "phase_4_verified",
        **headline,
        "mechanism_metrics": mechanism,
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(stats, f, indent=2)
    with (output_dir / "dependency_metrics.json").open("w") as f:
        json.dump(dep_metrics, f, indent=2)
    return stats


def _s8_compute_aging_curve(result: dict, output_dir) -> tuple[dict, dict, dict]:
    """Phase 4 aggregator: turn per-session verification + agent records
    into the aging-curve fields + the 4-mechanism breakdown."""
    import json as _json
    from agingbench.metrics.aging import AgingCurve, summarize
    from agingbench.scenarios.s8_swe_bench.probes import (
        compute_compression_probe,
        compute_interference_probe,
        compute_revision_probe,
        compute_maintenance_probe,
    )
    from agingbench.scenarios.s8_swe_bench.verifier import get_instance_metadata

    sessions = sorted(result["session_results"], key=lambda s: s["session"])
    binary_pass_seq: list[float] = []     # strict SWE-bench scoring (all f2p AND all p2p)
    partial_pass_seq: list[float] = []     # continuous: (f2p_passed + p2p_passed)/(total)
    checkpoints: list[list] = []            # uses partial (smooth headline)
    binary_checkpoints: list[list] = []     # sidecar for the strict curve
    for s in sessions:
        v = (s.get("agent_action") or {}).get("verification") or {}
        binary = 1.0 if v.get("passed") else 0.0
        f2p_t = v.get("n_fail_to_pass_total") or 0
        f2p_p = v.get("n_fail_to_pass_passed") or 0
        p2p_t = v.get("n_pass_to_pass_total") or 0
        p2p_p = v.get("n_pass_to_pass_passed") or 0
        denom = f2p_t + p2p_t
        partial = ((f2p_p + p2p_p) / denom) if denom > 0 else 0.0
        binary_pass_seq.append(binary)
        partial_pass_seq.append(partial)
        checkpoints.append([int(s["session"]), round(partial, 4)])
        binary_checkpoints.append([int(s["session"]), binary])

    # Headline curve = partial_pass_rate (continuous; smoother than the
    # strict SWE-bench rule). The strict binary curve ships in the sidecar.
    curve = AgingCurve(
        exposures=[c[0] for c in checkpoints],
        scores=partial_pass_seq,
        scenario="s8_swe_bench",
        sut_id=result.get("sut_id", "unknown"),
        metric_name="partial_pass_rate",
    )
    headline = summarize(curve)
    headline.pop("scenario", None)
    headline.pop("sut_id", None)
    headline["checkpoints"] = checkpoints
    headline["binary_pass_rate_curve"] = binary_checkpoints  # sidecar

    # ---- mechanism probes (per session, then aggregated) ----
    # Each session's agent_action.verification is treated as that session's
    # verdict. Notes/diffs come from agent_action. The compression probe
    # measures TASK-CRITICAL memory: it scans the agent's notes for facts
    # (files + symbols + concept tokens) derived from each prior session's
    # gold patch — i.e. the facts the agent definitionally had to know to
    # solve those issues.
    probe_records = []
    enriched_sessions: list[dict] = []
    # Phase 13 fix: read PER-SESSION notes snapshots. Using the final
    # notes.md for every session under-scored runs with lifecycle flushes
    # (the persistent notes file gets wiped, so all sessions appeared to
    # have empty memory). Per-session snapshots live at
    # `agentmemory/agent_work/session_<t>_<iid>/.aging/notes.md`.
    from pathlib import Path as _P
    import re as _re
    per_sess_notes: dict[int, str] = {}
    ws = result.get("workspace_root")
    if ws:
        work_dir = _P(ws) / "agentmemory" / "agent_work"
        if work_dir.is_dir():
            for sub in work_dir.iterdir():
                m_ = _re.match(r"session_(\d+)_", sub.name)
                if not m_:
                    continue
                idx = int(m_.group(1))
                nf = sub / ".aging" / "notes.md"
                if nf.is_file():
                    per_sess_notes[idx] = nf.read_text(
                        encoding="utf-8", errors="replace"
                    )
    final_notes_fallback = ""
    if ws:
        fnf = _P(ws) / "agentmemory" / ".aging" / "notes.md"
        if fnf.is_file():
            final_notes_fallback = fnf.read_text(encoding="utf-8", errors="replace")

    for s in sessions:
        t = int(s["session"])
        prior_facts: list[dict] = []
        for p in sessions:
            if p["session"] >= t:
                continue
            meta = get_instance_metadata(p["instance_id"])
            prior_facts.append({
                "session": p["session"],
                "instance_id": p["instance_id"],
                "problem_statement": meta.get("problem_statement", ""),
                "gold_patch": meta.get("patch", ""),
            })
        notes_t = per_sess_notes.get(t, final_notes_fallback)
        comp = compute_compression_probe(t, notes_t, prior_facts)
        probe_records.append({"session": t, "compression": comp.to_dict()})
        s2 = dict(s)
        s2["agent_notes_text"] = notes_t
        s2["prior_facts"] = prior_facts
        enriched_sessions.append(s2)

    # Interference: needs gold patches per instance + agent diffs.
    chain_path = result.get("chain_path")
    interference_pairs: list[list[str]] = []
    if chain_path:
        try:
            import yaml as _yaml
            with open(chain_path, "r") as _f:
                chain = _yaml.safe_load(_f)
            interference_pairs = chain.get("interference_pairs") or []
        except Exception:                                       # noqa: BLE001
            interference_pairs = []
    gold = {}
    for s in enriched_sessions:
        iid = s["instance_id"]
        meta = get_instance_metadata(iid)
        gold[iid] = meta.get("patch") or ""
    inter = compute_interference_probe(
        enriched_sessions, interference_pairs, gold,
    )

    # Revision: post-bump acknowledgment.
    rev = compute_revision_probe(enriched_sessions, result.get("lifecycle_events_planned") or [])

    # Maintenance: pre/post-flush pass-rate delta.
    maint = compute_maintenance_probe(enriched_sessions,
                                      result.get("lifecycle_events_planned") or [])

    # Compression mechanism summary: average task-critical-memory recall
    # across sessions. This is the headline "memory" signal — independent
    # of execution capability (the agent can fail tasks while still
    # remembering or forgetting the relevant files/symbols).
    recall_rates = [r["compression"]["recall_rate"] for r in probe_records
                    if r["compression"]["n_priors_total"] > 0]
    avg_recall = sum(recall_rates) / len(recall_rates) if recall_rates else 1.0

    mechanism_metrics = {
        "compression": {
            "score": round(avg_recall, 4),
            "task_critical_memory_recall": round(avg_recall, 4),
            "trajectory": [(r["session"], r["compression"]["recall_rate"])
                            for r in probe_records],
            "per_session": probe_records,
        },
        "interference": inter.to_dict(),
        "revision": rev.to_dict(),
        "maintenance": maint.to_dict(),
    }

    # ---- Phase 16: ORTHOGONAL four-mechanism probes + predictive validity ----
    from agingbench.scenarios.s8_swe_bench.probes import compute_orthogonal_probes
    import yaml as _yaml
    chain_dict: dict = {}
    cp = result.get("chain_path")
    if cp:
        try:
            with open(cp, "r") as _f:
                chain_dict = _yaml.safe_load(_f) or {}
        except Exception:                                       # noqa: BLE001
            chain_dict = {}
    seed_manifest = {"sessions": [
        {"session": s["session"], "instance_id": s["instance_id"]}
        for s in enriched_sessions
    ]}
    ortho = compute_orthogonal_probes(
        session_results=enriched_sessions,
        chain=chain_dict,
        seed_manifest=seed_manifest,
        lifecycle_events=result.get("lifecycle_events_planned") or [],
    )
    mechanism_metrics["orthogonal"] = ortho

    # Predictive validity: Pearson r between each orthogonal mechanism
    # trajectory and the task partial_pass_rate at the same session.
    def _pearson(xs, ys):
        if len(xs) < 3:
            return None
        n = len(xs)
        mx = sum(xs) / n; my = sum(ys) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        dx = (sum((xs[i] - mx) ** 2 for i in range(n))) ** 0.5
        dy = (sum((ys[i] - my) ** 2 for i in range(n))) ** 0.5
        if dx == 0 or dy == 0:
            return None
        return num / (dx * dy)

    pass_by_t = dict(checkpoints)
    pv = {}
    for traj_key in ("compression_clean_trajectory",
                      "interference_contrast_trajectory"):
        traj = ortho.get(traj_key) or []
        xs, ys = [], []
        for t, v in traj:
            if t in pass_by_t:
                xs.append(float(v))
                ys.append(float(pass_by_t[t]))
        r = _pearson(xs, ys)
        pv[traj_key.replace("_trajectory", "")] = (
            {"n": len(xs), "pearson_r": round(r, 4)} if r is not None
            else {"n": len(xs), "pearson_r": None}
        )
    # Revision: aggregate per-session-mean over multiple fact_ids
    rev_per = ortho.get("revision_per_fact") or []
    by_t: dict[int, list[float]] = {}
    for r in rev_per:
        by_t.setdefault(int(r["session"]), []).append(float(r["score"]))
    xs, ys = [], []
    for t, vals in sorted(by_t.items()):
        if t in pass_by_t:
            xs.append(sum(vals) / len(vals))
            ys.append(float(pass_by_t[t]))
    r = _pearson(xs, ys)
    pv["revision"] = ({"n": len(xs), "pearson_r": round(r, 4)} if r is not None
                       else {"n": len(xs), "pearson_r": None})
    mechanism_metrics["predictive_validity"] = pv

    dep_metrics = {
        "interference_resistance": mechanism_metrics["interference"]["resistance"],
        "n_interference_probes": mechanism_metrics["interference"]["n_pairs_evaluated"],
        "version_accuracy": mechanism_metrics["revision"]["version_accuracy"],
        "shock_sessions": mechanism_metrics["maintenance"]["shock_sessions"],
        "delta_shock": mechanism_metrics["maintenance"]["delta"],
        "task_critical_memory_recall": round(avg_recall, 4),
        "chain_recall": {"rate": round(avg_recall, 4),
                         "n_probes": len(recall_rates)},
    }
    return headline, mechanism_metrics, dep_metrics


_SCENARIO_RUNNERS = {
    # ---- canonical 8 scenarios ----
    "s1_research_literature": _run_s1,
    "s2_lifestyle_assistant": _run_s2,
    "s3_knowledge_base": _run_s3,
    "s4_software_engineering": _run_s4,
    "s5_self_planning": _run_s5,
    "s6_naturalistic": _run_s6,
    "s7_research_notes": _run_s7,
    "s8_swe_bench": _run_s8,
    # ---- Layer-2 self-planning variants (orthogonal to S5) ----
    # These wrap S2/S3/S6 task streams with the Layer-2 SelfPlanningRunner
    # (agent receives a high-level goal + tools and plans autonomously,
    # versus the per-task feeding loop the Layer-1 runners use). Distinct
    # from S5 (Self-Planning Notebook), which is its own scenario with a
    # workspace-managed notebook rather than tools-on-task-stream.
    "self_planning_s2": _run_self_planning,
    "self_planning_s3": _run_self_planning,
    "self_planning_s6": _run_self_planning,
}

# Discovered at import time so suites can reference any registered scenario.
_SCENARIO_MANIFESTS = _discover_scenarios()
