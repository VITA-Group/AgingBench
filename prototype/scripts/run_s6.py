#!/usr/bin/env python3
"""
Run S6 scenario — Naturalistic Aging (WebArena-derived multi-domain workflows).

Track B: measures whether memory aging occurs in realistic, non-adversarial
task sequences where memory carryover is rational.

Usage:
    # Single SUT
    python run_s6.py --sut agingbench/registry/suts/gpt4o_summarize_store.yaml

    # Quick validation (3 sessions)
    python run_s6.py --sut agingbench/registry/suts/gpt4o_no_memory.yaml --sessions 3

    # Specific model family (all configs)
    python run_s6.py --model-family gpt4o --sessions 5

    # Full run with oracle ablation
    python run_s6.py --sut agingbench/registry/suts/gpt4o_lossy_compress.yaml --oracle

Environment:
    ANTHROPIC_API_KEY  — required for Claude models
    OPENAI_API_KEY     — required for GPT models
"""

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _find_suts_by_family(family: str) -> list[Path]:
    """Find all SUT YAMLs matching a model family prefix."""
    sut_dir = PROJECT_ROOT / "agingbench" / "registry" / "suts"
    matches = sorted(sut_dir.glob(f"{family}_*.yaml"))
    if not matches:
        # Also try from CWD
        matches = sorted(Path("agingbench/registry/suts").glob(f"{family}_*.yaml"))
    return matches


def run_single_sut(sut_path: Path, n_sessions: int, output_base: str, oracle: bool,
                    generated: bool = False, n_seeds: int = 1,
                    multi_sut: bool = False):
    """Run S6 for a single SUT config, optionally with multiple seeds."""
    import yaml
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.runner.s6_runner import S6Runner
    from agingbench.runner.trace import TraceLogger
    from agingbench.metrics.aging import summarize, aggregate_curves
    from agingbench.report.plot import compare_curves, plot_curves_with_ci

    with open(sut_path) as f:
        sut_cfg = yaml.safe_load(f)

    sut_id = sut_cfg["sut_id"]
    if output_base and not multi_sut:
        # --output with a single --sut: use the path directly
        output_dir = Path(output_base)
    elif output_base:
        # --model-family: append sut_id to distinguish runs
        output_dir = Path(output_base) / sut_id
    else:
        output_dir = PROJECT_ROOT / "experiments" / "results" / "s6_naturalistic" / sut_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"S6 — Naturalistic Aging (WebArena-derived)")
    print(f"SUT: {sut_id}")
    print(f"Model: {sut_cfg['model'].get('model') or sut_cfg['model'].get('model_id', '?')}")
    print(f"Memory: {sut_cfg['memory_policy']['type']}")
    print(f"Sessions: {n_sessions}")
    print(f"Seeds: {n_seeds}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Load LLM (shared across seeds)
    print("\nLoading LLM...")
    llm = load_llm(sut_cfg["model"])
    print(f"LLM ready: {sut_cfg['model'].get('model', sut_cfg['model'].get('model_id', '?'))}")

    base_seed = sut_cfg.get("seed", 42)
    all_task_curves = []
    all_recall_curves = []
    all_results = []

    for seed_i in range(n_seeds):
        seed = base_seed + seed_i
        seed_label = f" [seed {seed_i}/{n_seeds}]" if n_seeds > 1 else ""
        print(f"\n--- Seed {seed}{seed_label} ---")

        # Fresh memory policy per seed
        memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)

        # Generate data with this seed
        generated_data = None
        if generated:
            from agingbench.generators.s6_generator import S6Generator
            from agingbench.cli.loaders import _resolve_pressure
            # Read pressure from sut_cfg (yaml). Previously hard-coded
            # PressureConfig.medium() — silently ignored confusable knobs,
            # update_rate, forget_rate, etc. set in the yaml.
            pressure = _resolve_pressure(sut_cfg=sut_cfg)
            generated_data = S6Generator(seed=seed, pressure=pressure).generate(n_sessions)
            if seed_i == 0:
                dep_summary = generated_data.get("dependency_graph", {}).get("summary", {})
                print(f"Generated {n_sessions} sessions (deps={dep_summary.get('total_dependency_tasks', 0)}, "
                      f"versioned={dep_summary.get('total_versioned', 0)}); "
                      f"pressure: confusable_start={pressure.confusable_start_session}, "
                      f"n_pairs={pressure.n_confusable_pairs}, "
                      f"update_rate={pressure.update_rate}, forget_rate={pressure.forget_rate}, "
                      f"similar_names={getattr(pressure, 'confusable_similar_names', False)}")

        # Run
        seed_dir = output_dir / f"seed_{seed_i}" if n_seeds > 1 else output_dir
        seed_dir.mkdir(parents=True, exist_ok=True)
        trace_path = seed_dir / "trace.jsonl"

        # Load maintenance events from SUT config (empty list if not configured)
        from agingbench.core.maintenance import load_maintenance_config
        maintenance_events = load_maintenance_config(sut_cfg)

        with TraceLogger(str(trace_path)) as tracer:
            runner = S6Runner(
                memory_policy=memory_policy,
                llm=llm,
                tracer=tracer,
                sut_id=sut_id,
                oracle_mode=oracle,
                generated_data=generated_data,
                maintenance_events=maintenance_events,
            )
            result = runner.run(n_sessions=n_sessions, seed=seed)

        all_task_curves.append(result["task_curve"])
        all_recall_curves.append(result["recall_curve"])
        all_results.append(result)

        # Save per-seed metrics
        stats = summarize(result["recall_curve"])

        # Dependency-aware metrics (if generated with pressure)
        if generated_data and "dependency_graph" in generated_data:
            from agingbench.metrics.dependency_scorer import score_dependency_chain
            dep_metrics = score_dependency_chain(
                result.get("session_results", []), generated_data["dependency_graph"]
            )
            stats["dependency_metrics"] = dep_metrics
            with open(seed_dir / "dependency_metrics.json", "w") as f:
                json.dump(dep_metrics, f, indent=2)

        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(stats, f, indent=2)

    # Use last result for single-seed backward compat
    result = all_results[-1]

    # --- Output ---
    if n_seeds > 1:
        # Multi-seed: aggregate and plot with CI
        agg = aggregate_curves(all_recall_curves)
        agg_task = aggregate_curves(all_task_curves)
        stats = {
            "scenario": "s6_naturalistic",
            "sut_id": sut_id,
            "n_seeds": n_seeds,
            **agg["summary"],
        }
        with open(output_dir / "metrics_aggregated.json", "w") as f:
            json.dump(stats, f, indent=2)

        title = f"S6 Naturalistic Aging — {sut_id} ({n_seeds} seeds)"
        plot_curves_with_ci(
            {"task_accuracy": all_task_curves, "recall_rate": all_recall_curves},
            str(output_dir / "aging_curve_ci.png"),
            title=title,
        )

        print(f"\n  Aggregated ({n_seeds} seeds):")
        print(f"    m0     = {agg['summary']['m0_mean']:.3f} ± {agg['summary']['m0_std']:.3f}")
        print(f"    m_final= {agg['summary']['m_final_mean']:.3f} ± {agg['summary']['m_final_std']:.3f}")
        print(f"    slope  = {agg['summary']['slope_mean']:.5f} ± {agg['summary']['slope_std']:.5f}")
    else:
        stats = summarize(result["recall_curve"])

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
    if oracle:
        stats["oracle_mode"] = True

    # Merge dependency_metrics from seed_dir into final metrics.json (single-seed only)
    dep_file = (output_dir / "dependency_metrics.json") if n_seeds == 1 else None
    if dep_file and dep_file.exists():
        with open(dep_file) as f:
            stats["dependency_metrics"] = json.load(f)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    title = f"S6 Naturalistic Aging — {sut_id}"
    if oracle:
        title += " (oracle)"
    compare_curves(
        [all_task_curves[-1], all_recall_curves[-1]],
        str(output_dir / "aging_curve.png"),
        title=title,
        labels=["task_accuracy", "recall_rate"],
    )

    # Print summary
    if n_seeds == 1:
        print(f"\n  Recall: m0={stats['m0']:.3f}  m_final={stats['m_final']:.3f}  "
              f"slope={stats['decay_slope']:.5f}  half_life={stats['half_life']}")

    print(f"\n  Recall matrix (rows=eval_time, cols=origin_session):")
    for t, row in sorted(result["recall_matrix"].items()):
        cells = "  ".join(f"s{s}={r:.1f}" for s, r in sorted(row.items()))
        print(f"    t={t:2d}: {cells}")

    print(f"\n  Lag curves:")
    for lag, points in sorted(result["lag_curves"].items()):
        rates = [r for _, r in points]
        avg = sum(rates) / len(rates) if rates else 0
        print(f"    lag={lag}: avg_recall={avg:.3f}  ({len(points)} points)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Run S6 — Naturalistic Aging")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sut", help="Path to a single SUT YAML config")
    group.add_argument("--model-family", help="Model family prefix (e.g. 'gpt4o', 'claude37')")
    parser.add_argument("--sessions", type=int, default=15, help="Number of sessions (default: 15)")
    parser.add_argument("--output", default="", help="Output directory base")
    parser.add_argument("--oracle", action="store_true", help="Oracle mode (no compression loss)")
    parser.add_argument("--generated", action="store_true",
                        help="Use programmatic generator instead of curated data")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds for CI (default: 1)")
    args = parser.parse_args()

    if args.sut:
        sut_paths = [Path(args.sut)]
    elif args.model_family:
        sut_paths = _find_suts_by_family(args.model_family)
        if not sut_paths:
            print(f"[error] No SUTs found for family '{args.model_family}'")
            sys.exit(1)

    multi_sut = len(sut_paths) > 1
    all_results = []
    for sut_path in sut_paths:
        try:
            stats = run_single_sut(sut_path, args.sessions, args.output, args.oracle,
                                   args.generated, n_seeds=args.seeds,
                                   multi_sut=multi_sut)
            all_results.append(stats)
        except Exception as e:
            print(f"\n[ERROR] Failed on {sut_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"{'SUT':<40} {'m0':>5} {'m_final':>7} {'slope':>9}")
        print(f"{'-'*70}")
        for r in all_results:
            print(f"  {r['sut_id']:<38} {r['m0']:>5.3f} {r['m_final']:>7.3f} {r['decay_slope']:>9.5f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
