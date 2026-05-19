#!/usr/bin/env python3
"""
Run S2 scenario with a real LLM.

Usage:
    python run_s2.py --sut agingbench/registry/suts/deepseek14b/lossy_compress.yaml
    python run_s2.py --sut agingbench/registry/suts/deepseek14b/lossy_compress.yaml --sessions 3
"""

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run S2 scenario")
    parser.add_argument("--sut", required=True, help="Path to SUT YAML config")
    parser.add_argument("--sessions", type=int, default=10, help="Number of sessions (default: 10)")
    parser.add_argument("--output", default="", help="Output directory")
    parser.add_argument("--oracle", action="store_true", help="Oracle mode (always use fresh profile)")
    parser.add_argument("--generated", action="store_true",
                        help="Use programmatic generator instead of curated data")
    args = parser.parse_args()

    import yaml
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy

    with open(args.sut) as f:
        sut_cfg = yaml.safe_load(f)

    sut_id = sut_cfg["sut_id"]
    oracle_mode = args.oracle

    output_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "experiments" / "results" / "s2_lifestyle_assistant" / sut_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"S2 — Personal Finance & Lifestyle Assistant")
    print(f"SUT: {sut_id}")
    print(f"Sessions: {args.sessions}")
    print(f"Oracle: {oracle_mode}")
    print(f"Generated: {args.generated}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Load LLM
    print("\nLoading LLM...")
    llm = load_llm(sut_cfg["model"])
    print(f"LLM loaded: {sut_cfg['model'].get('model_id', sut_cfg['model'].get('model', '?'))}")

    # Load memory policy
    memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)
    print(f"Memory policy: {sut_cfg['memory_policy']['type']}")

    # Generate data if requested
    generated_data = None
    if args.generated:
        from agingbench.generators.s2_generator import S2Generator
        from agingbench.generators.pressure_config import PressureConfig
        generated_data = S2Generator(
            seed=sut_cfg.get("seed", 42),
            pressure=PressureConfig.medium(),
        ).generate(args.sessions)
        n_acc = len(generated_data.get("accumulator_probes", []))
        dep_s = generated_data.get("dependency_graph", {}).get("summary", {})
        print(f"Generated {args.sessions} sessions (invalidated={dep_s.get('total_invalidated',0)}, "
              f"accumulator_probes={n_acc})")

    # Run S2
    from agingbench.runner.s2_runner import S2Runner
    from agingbench.runner.trace import TraceLogger
    from agingbench.metrics.aging import summarize
    from agingbench.report.plot import compare_curves

    trace_path = output_dir / "trace.jsonl"

    with TraceLogger(str(trace_path)) as tracer:
        runner = S2Runner(
            memory_policy=memory_policy,
            llm=llm,
            tracer=tracer,
            sut_id=sut_id,
            oracle_mode=oracle_mode,
            generated_data=generated_data,
        )
        result = runner.run(
            n_sessions=args.sessions,
            seed=sut_cfg.get("seed", 42),
        )

    precision_curve = result["precision_curve"]
    cvr_curve = result["cvr_curve"]
    tus_curve = result["tus_curve"]

    # Primary metric: constraint_precision (monotonically decaying)
    stats = summarize(precision_curve)
    stats["scenario"] = "s2_lifestyle_assistant"
    stats["metric_group"] = "G2"
    stats["primary_metric"] = "constraint_precision"
    stats["precision_raw"] = result["precision_raw"]
    stats["cvr_raw"] = result["cvr_raw"]
    stats["adherence_raw"] = result["adherence_raw"]
    stats["tus_raw"] = result["tus_raw"]
    stats["lag_recall_raw"] = result.get("lag_recall_raw", [])
    stats["compounding_raw"] = result.get("compounding_raw", [])
    stats["session_results"] = result["session_results"]
    if oracle_mode:
        stats["oracle_mode"] = True

    # Score dependency metrics (forget_accuracy, version_accuracy, etc.)
    if generated_data and "dependency_graph" in generated_data:
        from agingbench.metrics.dependency_scorer import score_dependency_chain, score_accumulator
        dep_metrics = score_dependency_chain(
            result.get("session_results", []), generated_data["dependency_graph"]
        )
        stats["dependency_metrics"] = dep_metrics
        with open(output_dir / "dependency_metrics.json", "w") as f_dep:
            json.dump(dep_metrics, f_dep, indent=2)
        print(f"\n  Dependency metrics: forget_acc={dep_metrics.get('forget_accuracy','N/A')}, "
              f"version_acc={dep_metrics.get('version_accuracy','N/A')}")

        # Score accumulator (Ledger-QA revision aging)
        if generated_data.get("accumulator_probes"):
            import re
            acc_probes = generated_data["accumulator_probes"]
            acc_results = []
            for probe in acc_probes:
                t = probe["session"]
                # Find the session result and extract agent output for the accumulator probe task
                sr = next((s for s in result.get("session_results", []) if s["session"] == t), None)
                if sr is None:
                    continue
                # Search through task outputs for the accumulator probe response
                task_outputs = sr.get("task_outputs", [])
                probe_resp = None
                for i, output in enumerate(task_outputs):
                    # Match by checking if it's the accumulator probe (last or second-to-last task)
                    if isinstance(output, str) and any(kw in output for kw in [str(int(probe["gold_value"])), "budget", "remaining", "balance"]):
                        probe_resp = output
                        break
                if probe_resp is None and task_outputs:
                    probe_resp = task_outputs[-1]  # fallback: last task output

                # Extract number from response
                gold = probe["gold_value"]
                agent_val = None
                if probe_resp:
                    nums = re.findall(r'\$?([\d,]+)', probe_resp.replace(",", ""))
                    for n in nums:
                        try:
                            agent_val = float(n)
                            break
                        except ValueError:
                            pass
                error = abs(agent_val - gold) if agent_val is not None else abs(gold)
                acc_results.append({
                    "session": t,
                    "gold_value": gold,
                    "agent_value": agent_val,
                    "error": error,
                    "response_preview": (probe_resp or "")[:200],
                })

            stats["accumulator_results"] = acc_results
            mean_err = sum(r["error"] for r in acc_results) / len(acc_results) if acc_results else 0
            print(f"\n  Accumulator (revision aging):")
            for r in acc_results:
                status = "✓" if r["error"] < 1 else f"✗ (off by {r['error']:.0f})"
                print(f"    session={r['session']} gold={r['gold_value']:.0f} agent={r['agent_value']} {status}")
            print(f"  Mean accumulator error: {mean_err:.1f}")

    stats.setdefault("scenario", "s2_lifestyle_assistant")
    stats.setdefault("sut_id", sut_id)
    stats.setdefault("metric_group", "G2")
    stats.setdefault("headline_metric", "constraint_precision")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    title = f"S2 Aging — {sut_id}"
    if oracle_mode:
        title += " (oracle)"

    compare_curves(
        [precision_curve, cvr_curve],
        str(output_dir / "aging_curve.png"),
        title=title,
        labels=["constraint_precision", "constraint_adherence (1-CVR)"],
    )

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  precision m0     = {stats['m0']:.3f}")
    print(f"  precision m_final= {stats['m_final']:.3f}")
    print(f"  decay_slope      = {stats['decay_slope']:.5f}")
    print(f"  half_life        = {stats['half_life']}")
    print(f"\n  Precision by session (primary aging metric):")
    for t, p in result["precision_raw"]:
        bar = "#" * int(p * 40)
        print(f"    Session {t:2d}: precision={p:.3f}  {bar}")
    print(f"\n  CVR by session (secondary):")
    for t, cvr in result["cvr_raw"]:
        print(f"    Session {t:2d}: CVR={cvr:.3f}")
    print(f"\n  Per-session detail:")
    for sr in result["session_results"]:
        violated = sr["violated_constraints"]
        v_str = ", ".join(violated) if violated else "(none)"
        cp = sr.get("constraint_precision", "?")
        print(f"    Session {sr['session']:2d}: precision={cp:.3f}  CVR={sr['cvr']:.2f}  violations={v_str}")

    print(f"\n  Output: {output_dir}")
    print(f"  Metrics: {output_dir / 'metrics.json'}")
    print(f"  Plot: {output_dir / 'aging_curve.png'}")
    print(f"  Trace: {trace_path}")


if __name__ == "__main__":
    main()
