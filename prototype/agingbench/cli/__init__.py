"""
agingbench/cli — Command-line interface.

Usage
-----
  # Run core suite with one SUT
  agingbench run --suite core --sut agingbench/registry/suts/llama3_lossy_compress.yaml

  # Run core suite with all registered SUTs
  agingbench run --suite core

  # Run single scenario directly
  agingbench run --scenario s1_research_literature --sut <path>

  # Run with oracle-memory ablation (S6)
  agingbench run --scenario s1_research_literature --sut <path> --oracle memory

  # Run with multiple seeds for confidence intervals
  agingbench run --suite core --seeds 3

  # Compare existing result directories
  agingbench compare experiments/results/llama3_no_memory experiments/results/llama3_lossy_compress
"""

from __future__ import annotations

import importlib
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

from .loaders import (
    PROJECT_ROOT,
    SUITE_DIR,
    SUT_DIR,
    SCENARIO_DIR,
    _load_yaml,
    _load_suite,
    _resolve_suts,
    _load_agent_class,
    _discover_scenarios,
)
from .runners import (
    _run_s1,
    _run_s2,
    _run_s3,
    _run_s4,
    _run_s5,
    _run_s6,
    _run_s7,
    _run_self_planning,
    _run_dynamic,
    _SCENARIO_RUNNERS,
    _SCENARIO_MANIFESTS,
)


# ------------------------------------------------------------------ commands

def cmd_run(suite_id: Optional[str], scenario_id: Optional[str],
            sut_arg: Optional[str], cycles: int, output: str,
            seeds: int = 1,
            diagnose: bool = False,
            agent_spec: Optional[str] = None,
            adapter_spec: Optional[str] = None,
            memory_policy_spec: Optional[str] = None,
            generated: bool = False, gen_sessions: int = 0,
            emit_card: bool = False) -> None:
    """Run one or more scenarios from a suite against one or more SUTs.

    Parameters
    ----------
    emit_card : bool
        When True, emit a consolidated aging_card.json alongside the
        existing metrics.json after each run completes. Default False
        so existing CI scripts that don't pass `--card` produce
        unchanged output.
    """

    if suite_id:
        suite = _load_suite(suite_id)
        scenarios = suite["scenarios"]
    elif scenario_id:
        suite = {"suite_id": "adhoc"}
        scenarios = [{"scenario_id": scenario_id, "n_cycles": cycles or 8}]
    else:
        print("[error] Provide --suite or --scenario.", file=sys.stderr)
        sys.exit(1)

    agent_class = _load_agent_class(agent_spec)

    sut_paths = _resolve_suts(suite if suite_id else {}, sut_arg)
    all_results: list[dict] = []

    for sut_path in sut_paths:
        sut_cfg = _load_yaml(sut_path)
        sut_id = sut_cfg["sut_id"]

        # CLI override hooks: --adapter / --memory-policy stomp the SUT YAML's
        # blocks with a {type: custom, class: <spec>} stub. Preserves any
        # other keys (max_turns, model, etc.) so users can mix flag+YAML.
        if adapter_spec:
            if ":" not in adapter_spec:
                print(f"[error] --adapter must be 'module.path:ClassName', got '{adapter_spec}'",
                      file=sys.stderr)
                sys.exit(1)
            existing = dict(sut_cfg.get("adapter") or {})
            existing.update({"type": "custom", "class": adapter_spec})
            sut_cfg["adapter"] = existing
        if memory_policy_spec:
            if ":" not in memory_policy_spec:
                print(f"[error] --memory-policy must be 'module.path:ClassName', got '{memory_policy_spec}'",
                      file=sys.stderr)
                sys.exit(1)
            existing = dict(sut_cfg.get("memory_policy") or {})
            existing.update({"type": "custom", "class": memory_policy_spec})
            sut_cfg["memory_policy"] = existing

        for scen in scenarios:
            sid = scen["scenario_id"]
            n_cyc = cycles if cycles > 0 else scen.get("n_cycles", 8)

            # Resolve default_sessions from manifest if available
            manifest = _SCENARIO_MANIFESTS.get(sid)
            if manifest and n_cyc == scen.get("n_cycles", 8):
                n_cyc = cycles if cycles > 0 else manifest.get("default_sessions", n_cyc)

            runner_fn = _SCENARIO_RUNNERS.get(sid)
            if runner_fn is None and manifest:
                # Dynamic dispatch: load runner class from manifest YAML
                runner_cfg = manifest.get("runner", {})
                if runner_cfg.get("module") and runner_cfg.get("class"):
                    try:
                        mod = importlib.import_module(runner_cfg["module"])
                        runner_cls = getattr(mod, runner_cfg["class"])
                        print(f"[info] Loaded runner {runner_cfg['class']} from manifest")
                        # Wrap in a function matching _run_sX signature
                        def _dynamic_runner(sut, scen_cfg, out, n, oracle=False, **kw):
                            return _run_dynamic(runner_cls, sut, scen_cfg, out, n, oracle, **kw)
                        runner_fn = _dynamic_runner
                    except Exception as e:
                        print(f"[warn] Failed to load runner from manifest: {e}")
            if runner_fn is None:
                print(f"[warn] No runner registered for scenario '{sid}', skipping.")
                print(f"       Discovered scenarios: {list(_SCENARIO_MANIFESTS.keys())}")
                continue

            for seed_idx in range(seeds):
                seed_val = sut_cfg.get("seed", 42) + seed_idx
                sut_cfg_copy = {**sut_cfg, "seed": seed_val}

                if seeds > 1:
                    out_dir = (
                        Path(output) / sid / sut_id / f"seed_{seed_idx}"
                        if output
                        else PROJECT_ROOT / "experiments" / "results" / sid / sut_id / f"seed_{seed_idx}"
                    )
                else:
                    out_dir = (
                        Path(output) / sut_id
                        if output
                        else PROJECT_ROOT / "experiments" / "results" / sid / sut_id
                    )

                # Print the active diagnostic mode.
                diag_suffix = " (P1/P2/P3 diagnostics)" if diagnose else ""
                seed_suffix = f" seed={seed_val}" if seeds > 1 else ""
                print(f"\n{'='*60}")
                print(f"AgingBench  suite={suite_id or 'adhoc'}  "
                      f"scenario={sid}  sut={sut_id}  "
                      f"cycles={n_cyc}{seed_suffix}{diag_suffix}")
                print(f"{'='*60}")

                # Only pass `diagnose` when explicitly requested. Most runner
                # signatures (S1-S5, S7, S8) don't accept it; only S6 does.
                runner_kwargs = {
                    "agent_class": agent_class,
                    "generated": generated,
                    "gen_sessions": gen_sessions,
                }
                if diagnose:
                    runner_kwargs["diagnose"] = diagnose
                stats = runner_fn(sut_cfg_copy, scen, out_dir, n_cyc,
                                  **runner_kwargs)
                stats["output_dir"] = str(out_dir)
                if seeds > 1:
                    stats["seed"] = seed_val
                all_results.append(stats)

                # Emit AgingCard JSON when opted in via --card. Pure
                # post-processor: never modifies metrics.json or
                # dependency_metrics.json.
                if emit_card:
                    try:
                        from agingbench.metrics.aging_card import (
                            build_and_write_aging_card,
                        )
                        card_path = build_and_write_aging_card(
                            output_dir=out_dir,
                            sut_cfg=sut_cfg_copy,
                            suite_id=(suite_id or "adhoc"),
                            seed=seed_val if seeds > 1 else sut_cfg_copy.get("seed"),
                        )
                        if card_path is not None:
                            print(f"[info] AgingCard -> {card_path}")
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"[warn] AgingCard emission failed (non-fatal): {e}")

                _print_row(stats)

            # Aggregate across seeds if multi-seed
            if seeds > 1:
                _aggregate_seeds(all_results[-seeds:], sid, sut_id, output)

    if len(all_results) > 1:
        _print_summary_table(all_results)
        _save_suite_report(all_results, suite_id or "adhoc", output)


def _aggregate_seeds(seed_results: list[dict], scenario_id: str,
                     sut_id: str, output: str) -> None:
    """Compute mean +/- std across seeds and write aggregated metrics.json."""
    m0s = [r["m0"] for r in seed_results if r.get("m0") is not None]
    m_finals = [r["m_final"] for r in seed_results if r.get("m_final") is not None]
    slopes = [r["decay_slope"] for r in seed_results]

    agg = {
        "scenario": scenario_id,
        "sut_id": sut_id,
        "seeds": len(seed_results),
        "m0_mean": round(statistics.mean(m0s), 4) if m0s else None,
        "m0_std": round(statistics.stdev(m0s), 4) if len(m0s) > 1 else 0.0,
        "m_final_mean": round(statistics.mean(m_finals), 4) if m_finals else None,
        "m_final_std": round(statistics.stdev(m_finals), 4) if len(m_finals) > 1 else 0.0,
        "decay_slope_mean": round(statistics.mean(slopes), 5),
        "decay_slope_std": round(statistics.stdev(slopes), 5) if len(slopes) > 1 else 0.0,
    }

    agg_dir = (
        Path(output) / scenario_id / sut_id
        if output
        else PROJECT_ROOT / "experiments" / "results" / scenario_id / sut_id
    )
    agg_dir.mkdir(parents=True, exist_ok=True)
    with open(agg_dir / "metrics_aggregated.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  Aggregated {len(seed_results)} seeds → {agg_dir / 'metrics_aggregated.json'}")


def cmd_compare(result_dirs: list[str]) -> None:
    """Overlay aging curves from multiple result directories."""
    from agingbench.metrics.aging import AgingCurve
    from agingbench.report.plot import compare_curves

    curves, labels, stats_list = [], [], []
    for d in result_dirs:
        metrics_path = Path(d) / "metrics.json"
        if not metrics_path.exists():
            print(f"[warn] No metrics.json in {d}, skipping.")
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        exps, scores = zip(*m["checkpoints"])
        curves.append(AgingCurve(
            exposures=list(exps), scores=list(scores),
            scenario=m["scenario"], sut_id=m["sut_id"],
        ))
        labels.append(m["sut_id"])
        stats_list.append(m)

    if not curves:
        print("[error] No valid result directories found.", file=sys.stderr)
        sys.exit(1)

    _print_summary_table(stats_list)
    out_path = Path(result_dirs[0]).parent / "comparison.png"
    compare_curves(curves, str(out_path), title="AgingBench — SUT Comparison", labels=labels)
    print(f"\nComparison plot → {out_path}")


# ------------------------------------------------------------------ formatting

def _print_row(stats: dict) -> None:
    m0 = stats.get("m0")
    m_final = stats.get("m_final")
    if m0 is None or m_final is None:
        phase = stats.get("phase", "unknown")
        status = stats.get("scaffold_status") or stats.get("headline_metric") or ""
        print(f"  (no aging curve — phase={phase})")
        if status:
            print(f"  {str(status)[:120]}")
        return
    hl = stats.get("half_life", float("inf"))
    hl_str = f"{hl:.2f}" if hl != float("inf") else "\u221e"
    hp = stats.get("hazard_proxy", 0.0)
    slope = stats.get("decay_slope", 0.0)
    print(f"  m0={m0:.3f}  m_final={m_final:.3f}  "
          f"half_life={hl_str}  slope={slope:.5f}  "
          f"hazard={hp:.4f}")
    if "task_m0" in stats:
        print(f"  task_m0={stats['task_m0']:.3f}  task_m_final={stats['task_m_final']:.3f}  "
              f"task_slope={stats['task_decay_slope']:.5f}")


def _print_summary_table(results: list[dict]) -> None:
    half_label = "t\u00bd"
    header = f"\n{'SUT':<35} {'scenario':<28} {'m0':>5} {'m_final':>7} {half_label:>6} {'slope':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        hl = r.get("half_life", float("inf"))
        hl_str = f"{hl:.1f}" if hl != float("inf") else "\u221e"
        m0 = r.get("m0")
        m_final = r.get("m_final")
        slope = r.get("decay_slope", float("nan"))
        if m0 is None or m_final is None:
            m0_str = "  n/a"
            m_final_str = "    n/a"
            slope_str = "      n/a"
        else:
            m0_str = f"{m0:>5.3f}"
            m_final_str = f"{m_final:>7.3f}"
            slope_str = f"{slope:>9.5f}"
        print(f"  {r['sut_id']:<33} {r['scenario']:<28} "
              f"{m0_str:>5} {m_final_str:>7} {hl_str:>6} {slope_str:>9}")
    print("=" * len(header))


def _save_suite_report(results: list[dict], suite_id: str, output: str) -> None:
    out_dir = Path(output) if output else PROJECT_ROOT / "experiments" / "results"
    report = {"suite_id": suite_id, "n_suts": len(results), "results": results}
    report_path = out_dir / f"suite_{suite_id}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSuite report → {report_path}")


# ------------------------------------------------------------------ entry point

def _build_parser():
    """Construct the AgingBench argparse parser.

    Factored out of ``main()`` so tests (and downstream tooling) can
    introspect the parser without invoking it.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="agingbench",
        description="AgingBench — Longitudinal Reliability Benchmark for Memory-Enabled Agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    run_p = sub.add_parser("run", help="Run a suite or scenario")
    run_p.add_argument("--suite", metavar="SUITE_ID",
                       help="Suite to run (e.g. 'core')")
    run_p.add_argument("--scenario", metavar="SCENARIO_ID",
                       help="Single scenario to run (e.g. 's1_research_literature')")
    run_p.add_argument("--sut", metavar="PATH",
                       help="SUT YAML config (default: all SUTs registered in suite)")
    run_p.add_argument("--cycles", type=int, default=0,
                       help="Override n_cycles (0 = use suite/YAML value)")
    run_p.add_argument("--output", default="",
                       help="Output root dir (default: experiments/results/<scenario>/<sut>)")
    run_p.add_argument("--seeds", type=int, default=1,
                       help="Number of seeds to run per SUT (default: 1)")
    run_p.add_argument("--diagnose", action="store_true", default=False,
                       help="Run P1/P2/P3 diagnostic error partitioning on recall probes. "
                            "Adds ~2x LLM calls per probe (P2 + P3) but requires only a "
                            "single run. Results include per-session write/read/utilization "
                            "error decomposition.")
    run_p.add_argument("--agent", metavar="MODULE:CLASS",
                       help="Override the Tier-1 ReferenceAgent reasoning class "
                            "(advanced; subclass of agingbench.core.agent.AgentInterface). "
                            "For BYO Tier-2 agents, use --adapter instead.")
    run_p.add_argument("--adapter", metavar="MODULE:CLASS",
                       help="Tier-2 BYO agent: subclass of AgentAdapter "
                            "(e.g. 'my_pkg.my_agent:MyAgent'). Overrides the "
                            "SUT YAML's adapter: block, replacing it with "
                            "{type: custom, class: <spec>}. See "
                            "examples/byo_agent_minimal.py.")
    run_p.add_argument("--memory-policy", metavar="MODULE:CLASS", dest="memory_policy",
                       help="Tier-1 BYO memory backbone: subclass of MemoryPolicy "
                            "(e.g. 'my_pkg.my_memory:MyMemory'). Overrides the "
                            "SUT YAML's memory_policy: block, replacing it with "
                            "{type: custom, class: <spec>}. See "
                            "examples/byo_memory_minimal.py.")
    run_p.add_argument("--generated", action="store_true",
                       help="Use programmatic scenario generator instead of curated data. "
                            "Note: this does NOT disable LLM calls — the agent still queries its "
                            "configured model, so an API key (e.g. ANTHROPIC_API_KEY) is still required "
                            "unless the SUT uses a local model.")
    run_p.add_argument("--sessions", type=int, default=0,
                       help="Override session count for --generated or single-scenario mode. "
                            "NOTE: per-runner semantics differ and are intentionally preserved "
                            "for paper reproducibility: S1 treats N as 'through session N "
                            "inclusive' (N+1 cycles); S2/S3/S4/S6 treat N as 'N sessions' "
                            "(exclusive); S7+ treats N as 'N agent blocks'. Suite-driven runs "
                            "(`--suite lite/full/core/...`) bypass this flag entirely and use "
                            "n_cycles from the suite YAML.")
    # AgingCard emission. Default OFF so existing CI scripts that don't
    # pass --card produce unchanged output.
    run_p.add_argument("--card", action="store_true", default=False,
                       help="Emit a consolidated aging_card.json alongside metrics.json. "
                            "Schema validated against agingbench/metrics/aging_card_schema.json.")

    # ---- compare ----
    cmp_p = sub.add_parser("compare", help="Compare results from multiple result directories")
    cmp_p.add_argument("dirs", nargs="+", metavar="RESULT_DIR",
                       help="Result directories containing metrics.json")

    return parser


def main(argv: list | None = None) -> int:
    """AgingBench CLI entry point.

    Returns
    -------
    int
        Process exit code. 0 on success; non-zero on argparse errors.
    """
    try:
        from dotenv import load_dotenv
        for candidate in (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate, override=False)
    except ImportError:
        pass

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(
            suite_id=args.suite,
            scenario_id=args.scenario,
            sut_arg=args.sut,
            cycles=args.cycles,
            output=args.output,
            seeds=args.seeds,
            diagnose=getattr(args, 'diagnose', False),
            agent_spec=args.agent,
            adapter_spec=getattr(args, 'adapter', None),
            memory_policy_spec=getattr(args, 'memory_policy', None),
            generated=args.generated,
            gen_sessions=args.sessions,
            emit_card=getattr(args, 'card', False),
        )
        return 0
    elif args.command == "compare":
        cmd_compare(args.dirs)
        return 0
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
