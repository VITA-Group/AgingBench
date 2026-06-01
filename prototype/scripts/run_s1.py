"""
run_s1.py — End-to-end entry point for S1 (Research Literature).

Equivalent to `agingbench run --scenario s1_research_literature --sut <yaml>`
but exposes scenario-specific knobs (--cycles, --compare).

Usage:
  # Baseline (no memory, should stay flat near 1.0)
  python run_s1.py --sut agingbench/registry/suts/llama3/llama3_no_memory.yaml

  # Primary condition (summarize_store — decay expected by cycle 3-6)
  python run_s1.py --sut agingbench/registry/suts/llama3/llama3_summarize_store.yaml

  # Quick calibration run (3 cycles)
  python run_s1.py --sut agingbench/registry/suts/llama3/llama3_summarize_store.yaml --cycles 3

  # Compare two prior runs
  python run_s1.py --compare \\
    experiments/results/s1_llama3_no_memory/metrics.json \\
    experiments/results/s1_llama3_summarize_store/metrics.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import yaml


SCENARIO_DIR = Path(__file__).parent.parent / "agingbench" / "scenarios" / "s1_research_literature"


# ------------------------------------------------------------------ helpers

def load_sut(sut_path: str) -> dict:
    with open(sut_path) as f:
        return yaml.safe_load(f)


def build_memory_policy(cfg: dict, project_root: Path):
    policy_type = cfg["memory_policy"]["type"]
    if policy_type == "no_memory":
        from agingbench.core.memory.no_memory import NoMemoryPolicy
        return NoMemoryPolicy()
    if policy_type == "summarize_store":
        from agingbench.core.memory.summarize_store import SummarizeStorePolicy, COMPACT_MEDIUM
        prompt_path = cfg["memory_policy"].get("compaction_prompt")
        if prompt_path:
            full = project_root / prompt_path
            prompt_template = full.read_text()
        else:
            prompt_template = COMPACT_MEDIUM
        return SummarizeStorePolicy(prompt_template=prompt_template)
    if policy_type == "growing_history":
        from agingbench.core.memory.growing_history import GrowingHistoryStorePolicy
        from agingbench.core.memory.summarize_store import COMPACT_MEDIUM
        prompt_path = cfg["memory_policy"].get("compaction_prompt")
        if prompt_path:
            full = project_root / prompt_path
            prompt_template = full.read_text()
        else:
            prompt_template = COMPACT_MEDIUM
        word_budget = cfg["memory_policy"].get("word_budget", 300)
        return GrowingHistoryStorePolicy(prompt_template=prompt_template, word_budget=word_budget)
    if policy_type == "append_only":
        from agingbench.core.memory.append_only import AppendOnlyPolicy
        return AppendOnlyPolicy()
    raise ValueError(f"Unknown memory policy: {policy_type}")


def print_table(results: list[dict]) -> None:
    """Print a summary table of aging curve statistics."""
    header = f"{'SUT':<35} {'m0':>5} {'m_final':>7} {'half_life':>10} {'slope':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        hl = r["half_life"]
        hl_str = f"{hl:.2f}" if hl != float("inf") else "  ∞"
        print(f"{r['sut_id']:<35} {r['m0']:>5.3f} {r['m_final']:>7.3f} "
              f"{hl_str:>10} {r['decay_slope']:>10.5f}")
    print("=" * len(header))


# ------------------------------------------------------------------ compare mode

def compare_mode(metrics_paths: list[str]) -> None:
    results = []
    curves = []
    for p in metrics_paths:
        with open(p) as f:
            m = json.load(f)
        results.append(m)
        from agingbench.metrics.aging import AgingCurve
        exps, scores = zip(*m["checkpoints"])
        curves.append(AgingCurve(
            exposures=list(exps), scores=list(scores),
            scenario=m["scenario"], sut_id=m["sut_id"]
        ))
    print_table(results)
    from agingbench.report.plot import compare_curves
    out_dir = Path(metrics_paths[0]).parent.parent
    compare_curves(curves, str(out_dir / "p2_comparison.png"), title="P2 Aging — SUT Comparison")


# ------------------------------------------------------------------ main run

def run(sut_path: str, cycles: int, output_dir: Path, generated: bool = False,
        score_via_response: bool = False) -> None:
    project_root = Path(__file__).parent.parent
    sut_cfg = load_sut(sut_path)
    sut_id = sut_cfg["sut_id"]
    seed = sut_cfg.get("seed", 42)
    n_cycles = cycles if cycles > 0 else sut_cfg.get("n_cycles", 8)

    print(f"\n{'='*60}")
    print(f"AgingBench — Scenario P2 (Summarization Drift)")
    print(f"  SUT:     {sut_id}")
    print(f"  Policy:  {sut_cfg['memory_policy']['type']}")
    print(f"  Cycles:  {n_cycles}")
    print(f"  Output:  {output_dir}")
    print(f"  Mode:    {'generated (seed-dependent)' if generated else 'curated (disk)'}")
    print(f"{'='*60}\n")

    gen_data = None
    if generated:
        from agingbench.generators.s1_generator import S1Generator
        from agingbench.cli.loaders import _resolve_pressure
        pressure = _resolve_pressure(sut_cfg=sut_cfg)
        gen_data = S1Generator(
            seed=seed,
            pressure=pressure,
            dense_revision=sut_cfg.get("dense_revision", False),
        ).generate(n_cycles)
        source_doc = gen_data["source_doc"]
        probes = gen_data["probes"]
    else:
        with open(SCENARIO_DIR / "source_doc.json") as f:
            source_doc = json.load(f)
        with open(SCENARIO_DIR / "probes.json") as f:
            probes = json.load(f)

    # Compliance tasks are seed-independent (curated)
    tasks_path = SCENARIO_DIR / "tasks.jsonl"
    tasks = []
    if tasks_path.exists():
        with open(tasks_path) as f:
            tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Source doc: {len(source_doc['text'])} chars  |  "
          f"{len(probes)} probes  |  {len(tasks)} tasks\n")

    # Load LLM via provider-agnostic factory (§6.1.1 adapter layer)
    print("Loading LLM …")
    from agingbench.core.llm import load_llm
    llm = load_llm(sut_cfg["model"])
    model_id = sut_cfg["model"].get("model_id") or sut_cfg["model"].get("model", "unknown")
    print(f"Model loaded: {model_id}\n")

    # Build memory policy
    memory_policy = build_memory_policy(sut_cfg, project_root)

    # Sanity-check: cycle-0 validator on raw source doc
    from agingbench.scenarios.s1_research_literature.validator import score_all
    scores_0, m_0 = score_all(source_doc["text"], probes)
    print(f"[Sanity] cycle-0 score on raw source_doc: {m_0:.3f} "
          f"({sum(scores_0)}/{len(scores_0)} probes)")
    if m_0 < 0.95:
        print("[WARNING] Cycle-0 score < 0.95 — check keyword coverage in probes.json")

    # Set up output
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"

    from agingbench.runner.trace import TraceLogger
    from agingbench.runner.s1_runner import S1Runner
    from agingbench.metrics.aging import summarize
    from agingbench.report.plot import plot_curve

    with TraceLogger(str(trace_path)) as tracer:
        runner = S1Runner(
            source_doc_text=source_doc["text"],
            probes=probes,
            validator_fn=score_all,
            memory_policy=memory_policy,
            llm=llm,
            tracer=tracer,
            sut_id=sut_id,
            tasks=tasks,
            generated_data=gen_data,
            score_via_response=(
                score_via_response
                or sut_cfg.get("score_via_response", False)
            ),
        )
        result = runner.run(n_cycles=n_cycles, seed=seed)
        keyword_curve = result["keyword_curve"]
        task_curve = result["task_curve"]
        lag_recall_curve = result.get("lag_recall_curve")
        recall_matrix = result.get("recall_matrix")
        session_results = result.get("session_results", [])

    # Metrics
    stats = summarize(keyword_curve)
    stats["scenario"] = "s1_research_literature"
    stats["metric_group"] = "G1"
    stats["headline_metric"] = "keyword_recall"
    if task_curve and task_curve.scores:
        from agingbench.metrics.aging import compute_half_life, compute_decay_slope
        stats["task_m0"] = task_curve.scores[0]
        stats["task_m_final"] = task_curve.scores[-1]
        stats["task_half_life"] = compute_half_life(task_curve)
        stats["task_decay_slope"] = round(compute_decay_slope(task_curve), 5)
        stats["task_checkpoints"] = list(zip(task_curve.exposures, task_curve.scores))

    if session_results:
        stats["session_results"] = session_results
    if gen_data and "dependency_graph" in gen_data and session_results:
        from agingbench.metrics.dependency_scorer import score_dependency_chain
        dep_metrics = score_dependency_chain(session_results, gen_data["dependency_graph"])
        stats["dependency_metrics"] = dep_metrics
        with open(output_dir / "dependency_metrics.json", "w") as f:
            json.dump(dep_metrics, f, indent=2)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Plot keyword curve; overlay task curve if present
    if task_curve and task_curve.scores:
        from agingbench.report.plot import compare_curves
        compare_curves(
            [keyword_curve, task_curve],
            str(output_dir / "aging_curve.png"),
            title=f"P2 Aging — {sut_id}",
            labels=["keyword_m", "task_m"],
        )
    else:
        plot_curve(keyword_curve, str(output_dir / "aging_curve.png"),
                   title=f"P2 Aging — {sut_id}")

    # Summary
    print_table([stats])
    if task_curve and task_curve.scores:
        print(f"\n  task_m0={stats['task_m0']:.3f}  task_m_final={stats['task_m_final']:.3f}  "
              f"task_half_life={stats.get('task_half_life', float('inf')):.2f}  "
              f"task_slope={stats['task_decay_slope']:.5f}")
    print(f"\nTrace  → {trace_path}")
    print(f"Metrics → {metrics_path}")
    print(f"Plot    → {output_dir / 'aging_curve.png'}")


# ------------------------------------------------------------------ CLI

def main():
    parser = argparse.ArgumentParser(description="AgingBench P2 — Summarization Drift")
    parser.add_argument("--sut", help="Path to SUT YAML config")
    parser.add_argument("--cycles", "--sessions", dest="cycles", type=int, default=8,
                        help="Number of S1 cycles/sessions (default: 8). "
                             "S1 historically used 'cycles' while S2-S6 use 'sessions' "
                             "— both flag names are accepted.")
    parser.add_argument("--output", default="",
                        help="Output directory (default: experiments/results/<sut_id>)")
    parser.add_argument("--score-via-response", action="store_true",
                        help="Ask the LLM each probe (keyword + trend) with current memory "
                             "as context and score the response. End-to-end W+R+U; "
                             "default is memory-based substring check (W+R only). "
                             "Adds ~(N_kw_probes * N_cycles + N_trend_probes) LLM calls.")
    parser.add_argument("--generated", action="store_true",
                        help="Use S1Generator (seed-dependent) instead of curated disk data")
    parser.add_argument("--compare", nargs="+", metavar="METRICS_JSON",
                        help="Compare mode: pass 2+ metrics.json paths to overlay curves")
    args = parser.parse_args()

    if args.compare:
        compare_mode(args.compare)
        return

    if not args.sut:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output) if args.output else (
        Path("experiments/results") / Path(args.sut).stem
    )
    run(sut_path=args.sut, cycles=args.cycles, output_dir=output_dir,
        generated=args.generated,
        score_via_response=args.score_via_response)


if __name__ == "__main__":
    main()
