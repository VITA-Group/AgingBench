#!/usr/bin/env python3
"""Run S6 (Naturalistic Aging) with the unified FullReactAgent.

Side-by-side variant of scripts/run_s6.py. Same SUT YAMLs, same generators,
same scoring — only the agent loop is replaced (memory accessed via
search_memory tool instead of being dumped into the system prompt).

Usage:
    python scripts/run_s6_unified.py --sut <yaml>
    python scripts/run_s6_unified.py --sut <yaml> --sessions 5 --seeds 1
    python scripts/run_s6_unified.py --sut <yaml> --max-turns 12 --top-k 5

For comparison against the original runner, run scripts/run_s6.py with the
SAME --sut and seed, and compare metrics.json side by side. The
``loop_utilization`` field is unique to this runner (the original doesn't
emit it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Run S6 with FullReactAgent (unified)")
    p.add_argument("--sut", required=True, help="Path to SUT YAML config")
    p.add_argument("--sessions", type=int, default=15)
    p.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    p.add_argument("--max-turns", type=int, default=10,
                   help="FullReactAgent max_turns (default 10)")
    p.add_argument("--top-k", type=int, default=3,
                   help="search_memory top_k (default 3)")
    p.add_argument("--output", default="", help="Output directory base")
    args = p.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))

    import yaml
    import statistics
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.core.maintenance import load_maintenance_config
    from agingbench.runner.s6_runner_unified import S6UnifiedRunner
    from agingbench.runner.trace import TraceLogger
    from agingbench.generators.s6_generator import S6Generator
    from agingbench.cli.loaders import _resolve_pressure
    from agingbench.metrics.aging import summarize
    from agingbench.metrics.dependency_scorer import score_dependency_chain

    with open(args.sut) as f:
        sut_cfg = yaml.safe_load(f)
    sut_id = sut_cfg["sut_id"]

    if args.output:
        out_base = Path(args.output)
    else:
        out_base = (
            PROJECT_ROOT / "experiments" / "results" / "s6_naturalistic_unified" / sut_id
        )

    print(f"{'='*60}")
    print(f"S6 (UNIFIED) — Naturalistic Aging")
    print(f"SUT: {sut_id}  Sessions: {args.sessions}  Seeds: {args.seeds}")
    print(f"max_turns={args.max_turns}  search_memory_top_k={args.top_k}")
    print(f"Output base: {out_base}")
    print(f"{'='*60}")

    base_seed = sut_cfg.get("seed", 42)
    all_summaries: list[dict] = []

    for seed_i in range(args.seeds):
        seed = base_seed + seed_i
        if args.seeds > 1:
            out_dir = out_base / f"seed_{seed_i}"
        else:
            out_dir = out_base
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- seed {seed} ---")

        llm = load_llm(sut_cfg["model"])
        memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)
        gen = S6Generator(
            seed=seed,
            pressure=_resolve_pressure(sut_cfg=sut_cfg),
        )
        generated_data = gen.generate(args.sessions)
        maintenance_events = load_maintenance_config(sut_cfg)

        trace_path = out_dir / "trace.jsonl"
        with TraceLogger(str(trace_path)) as tracer:
            runner = S6UnifiedRunner(
                memory_policy=memory_policy,
                llm=llm,
                tracer=tracer,
                sut_id=sut_id,
                generated_data=generated_data,
                maintenance_events=maintenance_events,
                agent_max_turns=args.max_turns,
                search_memory_top_k=args.top_k,
            )
            result = runner.run(n_sessions=args.sessions, seed=seed)

        # Compose metrics.json (same shape as run_s6.py's _run_s6, minus oracle bits)
        recall_curve = result["recall_curve"]
        stats = summarize(recall_curve)
        stats["scenario"] = "s6_naturalistic"
        stats["agent"] = "full_react_agent"
        stats["headline_metric"] = "recall_compression"
        stats["n_sessions"] = len(result["task_raw"])
        stats["task_raw"] = result["task_raw"]
        stats["recall_raw"] = result["recall_raw"]
        stats["recall_matrix"] = {str(k): v for k, v in result["recall_matrix"].items()}
        stats["lag_curves"] = {str(k): v for k, v in result["lag_curves"].items()}
        stats["session_results"] = result["session_results"]
        stats["loop_utilization"] = result["loop_utilization"]

        if generated_data and "dependency_graph" in generated_data:
            dep_metrics = score_dependency_chain(
                result.get("session_results", []), generated_data["dependency_graph"]
            )
            stats["dependency_metrics"] = dep_metrics
            with open(out_dir / "dependency_metrics.json", "w") as fdep:
                json.dump(dep_metrics, fdep, indent=2)

        with open(out_dir / "metrics.json", "w") as fm:
            json.dump(stats, fm, indent=2)

        all_summaries.append({
            "seed": seed,
            "m0": stats["m0"],
            "m_final": stats["m_final"],
            "decay_slope": stats["decay_slope"],
            "loop": stats["loop_utilization"],
        })

        print(
            f"  seed={seed}  m0={stats['m0']:.3f}  m_final={stats['m_final']:.3f}  "
            f"slope={stats['decay_slope']:.5f}  "
            f"turns_med={stats['loop_utilization']['turns_median_overall']}  "
            f"tools/sess={stats['loop_utilization']['tool_calls_per_session']:.1f}  "
            f"exhausted={stats['loop_utilization']['exhausted_session_share']:.2%}"
        )

    if args.seeds > 1:
        def _mean_std(vals: list[float]) -> tuple[float, float]:
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
            return m, sd
        m0_m, m0_s = _mean_std([s["m0"] for s in all_summaries])
        mf_m, mf_s = _mean_std([s["m_final"] for s in all_summaries])
        sl_m, sl_s = _mean_std([s["decay_slope"] for s in all_summaries])
        agg = {
            "scenario": "s6_naturalistic",
            "agent": "full_react_agent",
            "sut_id": sut_id,
            "n_seeds": args.seeds,
            "m0_mean": m0_m, "m0_std": m0_s,
            "m_final_mean": mf_m, "m_final_std": mf_s,
            "decay_slope_mean": sl_m, "decay_slope_std": sl_s,
        }
        with open(out_base / "metrics_aggregated.json", "w") as fagg:
            json.dump(agg, fagg, indent=2)
        print(
            f"\n  Aggregated ({args.seeds} seeds):  "
            f"m0={m0_m:.3f}±{m0_s:.3f}  m_final={mf_m:.3f}±{mf_s:.3f}  "
            f"slope={sl_m:.5f}±{sl_s:.5f}"
        )


if __name__ == "__main__":
    main()
