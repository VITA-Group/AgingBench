#!/usr/bin/env python3
"""Run S3 (Project Knowledge Base) with the unified FullReactAgent.

Side-by-side variant of scripts/run_s3.py. Same SUT YAMLs, same generator,
same scoring (fidelity, query_accuracy, contradiction, revision_aging) — only
the agent loop is replaced. Memory is accessed via search_memory tool instead
of being dumped into the user prompt.

Usage:
    python scripts/run_s3_unified.py --sut <yaml>
    python scripts/run_s3_unified.py --sut <yaml> --sessions 12 --max-turns 12 --top-k 3
    python scripts/run_s3_unified.py --sut <yaml> --held-out-probes 2

To diff against the original S3Runner under matched seed, run scripts/run_s3.py
with the same --sut and compare metrics.json. The ``loop_utilization`` field
is unique to this runner — the original doesn't emit it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Run S3 with FullReactAgent (unified)")
    p.add_argument("--sut", required=True, help="Path to SUT YAML config")
    p.add_argument("--sessions", type=int, default=12)
    p.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    p.add_argument("--max-turns", type=int, default=12,
                   help="FullReactAgent max_turns (default 12)")
    p.add_argument("--top-k", type=int, default=3,
                   help="search_memory top_k (default 3)")
    p.add_argument("--held-out-probes", type=int, default=2,
                   help="Held-out probes per session. Default 2 = clean substrate-decay "
                        "channel ON. Pass 0 to disable.")
    p.add_argument("--output", default="", help="Output directory base")
    args = p.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))

    import yaml
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.runner.s3_runner_unified import S3UnifiedRunner
    from agingbench.runner.trace import TraceLogger
    from agingbench.generators.s3_generator import S3Generator
    from agingbench.cli.loaders import _resolve_pressure
    from agingbench.metrics.aging import summarize

    with open(args.sut) as f:
        sut_cfg = yaml.safe_load(f)
    sut_id = sut_cfg["sut_id"]

    if args.output:
        out_base = Path(args.output)
    else:
        out_base = (
            PROJECT_ROOT / "experiments" / "results" / "s3_knowledge_base_unified" / sut_id
        )

    print(f"{'='*60}")
    print(f"S3 (UNIFIED) — Project Knowledge Base Agent")
    print(f"SUT: {sut_id}  Sessions: {args.sessions}  Seeds: {args.seeds}")
    print(f"max_turns={args.max_turns}  search_memory_top_k={args.top_k}")
    print(f"held_out_probes_per_session={args.held_out_probes}")
    print(f"Output base: {out_base}")
    print(f"{'='*60}")

    base_seed = sut_cfg.get("seed", 42)

    for seed_i in range(args.seeds):
        seed = base_seed + seed_i
        out_dir = out_base / f"seed_{seed_i}" if args.seeds > 1 else out_base
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- seed {seed} ---")

        llm = load_llm(sut_cfg["model"])
        memory_policy = build_memory_policy(sut_cfg["memory_policy"], PROJECT_ROOT)
        gen = S3Generator(
            seed=seed,
            pressure=_resolve_pressure(sut_cfg=sut_cfg),
        )
        generated_data = gen.generate(args.sessions)

        trace_path = out_dir / "trace.jsonl"
        with TraceLogger(str(trace_path)) as tracer:
            runner = S3UnifiedRunner(
                memory_policy=memory_policy,
                llm=llm,
                tracer=tracer,
                sut_id=sut_id,
                generated_data=generated_data,
                agent_max_turns=args.max_turns,
                search_memory_top_k=args.top_k,
                held_out_probes_per_session=args.held_out_probes,
            )
            result = runner.run(n_sessions=args.sessions, seed=seed)

        fidelity_curve = result["fidelity_curve"]
        stats = summarize(fidelity_curve)
        stats["scenario"] = "s3_knowledge_base"
        stats["agent"] = "full_react_agent"
        stats["headline_metric"] = "summarization_fidelity"
        stats["n_sessions"] = args.sessions
        stats["fidelity_raw"] = result["fidelity_raw"]
        stats["bloat_raw"] = result["bloat_raw"]
        stats["contradiction_raw"] = result["contradiction_raw"]
        stats["query_acc_raw"] = result["query_acc_raw"]
        stats["held_out_acc_raw"] = result["held_out_acc_raw"]
        stats["held_out_probes_per_session"] = result["held_out_probes_per_session"]
        stats["session_results"] = result["session_results"]
        stats["loop_utilization"] = result["loop_utilization"]

        with open(out_dir / "metrics.json", "w") as fm:
            json.dump(stats, fm, indent=2)

        # CLI summary
        qa = dict(result["query_acc_raw"])
        ho = dict(result["held_out_acc_raw"])
        shared = sorted(set(qa) & set(ho))
        if shared:
            qa_mean = sum(qa[t] for t in shared) / len(shared)
            ho_mean = sum(ho[t] for t in shared) / len(shared)
            print(
                f"  seed={seed}  m_final={stats['m_final']:.3f}  "
                f"slope={stats['decay_slope']:+.5f}  "
                f"in_chan={qa_mean:.3f}  held_out={ho_mean:.3f}  "
                f"Δ={qa_mean-ho_mean:+.3f}  "
                f"turns_med={result['loop_utilization']['turns_median_overall']}  "
                f"tools/sess={result['loop_utilization']['tool_calls_per_session']:.1f}"
            )
        else:
            print(
                f"  seed={seed}  m_final={stats['m_final']:.3f}  "
                f"slope={stats['decay_slope']:+.5f}  "
                f"turns_med={result['loop_utilization']['turns_median_overall']}  "
                f"tools/sess={result['loop_utilization']['tool_calls_per_session']:.1f}"
            )


if __name__ == "__main__":
    main()
