#!/usr/bin/env python3
"""
run_s5.py — End-to-end entry point for S5 (Self-Planning Notebook).

S5 is the self-planning scenario where the agent manages its own workspace
files (Tier 1 with workspace-file access via the ReactFileAdapter or a
custom AgentAdapter). This wrapper exposes scenario-specific knobs
(--session-length, --adapter, --domain, --workspace, --no-reset) that the
generic `agingbench run` CLI does not surface.

Usage:
    # ReactFileAdapter with API model (default)
    python run_s5.py --sut agingbench/registry/suts/haiku45/haiku45_self_planning.yaml

    # ReactFileAdapter with local model
    python run_s5.py --sut agingbench/registry/suts/llama3/llama3_self_planning.yaml

    # Claude Code adapter
    python run_s5.py --sut agingbench/registry/suts/claude_code/claude_code_self_planning.yaml --adapter claude_code

    # Short validation run
    python run_s5.py --sut <config> --sessions 2 --session-length 5

    # Different domain
    python run_s5.py --sut <config> --domain coding

(S5 was historically called S7; renamed in v0.2. The runner module is
agingbench.runner.s5_runner.S5Runner.)
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run S7 — True Self-Planning")
    parser.add_argument("--sut", required=True, help="Path to SUT YAML config")
    parser.add_argument("--sessions", type=int, default=10, help="Number of session blocks")
    parser.add_argument("--session-length", type=int, default=12, help="Interactions per block")
    parser.add_argument("--output", default="", help="Output directory")
    parser.add_argument("--adapter", default="react",
                        choices=["react", "claude_code", "codex", "custom"],
                        help="Adapter type")
    parser.add_argument("--domain", default="assistant",
                        choices=["assistant", "knowledge_base", "coding"],
                        help="Task domain")
    parser.add_argument("--workspace", default="", help="Workspace directory (default: temp)")
    parser.add_argument("--no-reset", action="store_true",
                        help="Don't clear conversation history between blocks (continuous session)")
    parser.add_argument("--pressure", default="none",
                        choices=["none", "light", "medium", "heavy"],
                        help="Dependency pressure level (controls output_dependency pairs)")
    args = parser.parse_args()

    import yaml
    with open(args.sut) as f:
        sut_cfg = yaml.safe_load(f)

    sut_id = sut_cfg.get("sut_id", "unknown")
    output_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "experiments" / "results" / "s5_self_planning" / sut_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Workspace
    if args.workspace:
        workspace_dir = args.workspace
    else:
        workspace_dir = str(output_dir / "workspace")

    os.makedirs(workspace_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"S7 — True Self-Planning Evaluation")
    print(f"SUT: {sut_id}")
    print(f"Adapter: {args.adapter}")
    print(f"Domain: {args.domain}")
    print(f"Sessions: {args.sessions} x {args.session_length} = {args.sessions * args.session_length} interactions")
    print(f"Workspace: {workspace_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Build adapter
    adapter_cfg = sut_cfg.get("adapter", {})
    adapter_type = args.adapter or adapter_cfg.get("type", "react")

    if adapter_type == "react":
        from agingbench.core.llm import load_llm
        from agingbench.core.adapters.react_file_adapter import ReactFileAdapter

        print("\nLoading LLM...")
        llm = load_llm(sut_cfg["model"])
        print(f"LLM loaded: {sut_cfg['model'].get('model_id', sut_cfg['model'].get('model', '?'))}")

        max_turns = adapter_cfg.get("max_turns", 15)
        adapter = ReactFileAdapter(llm=llm, workspace_dir=workspace_dir, max_turns=max_turns)

    elif adapter_type == "claude_code":
        from agingbench.core.adapters.claude_code_agent_adapter import ClaudeCodeAgentAdapter

        model = adapter_cfg.get("model", "claude-sonnet-4-6-20250514")
        max_turns = adapter_cfg.get("max_turns", 50)
        cli_path = adapter_cfg.get("cli_path", os.environ.get(
            "CLAUDE_CLI_PATH",
            "claude",  # rely on PATH; matches ClaudeCodeAgentAdapter's own default
        ))
        adapter = ClaudeCodeAgentAdapter(
            model=model, cwd=workspace_dir, max_turns=max_turns,
            cli_path=cli_path,
        )

    elif adapter_type == "codex":
        from agingbench.core.adapters.codex_adapter import CodexAdapter

        model = adapter_cfg.get("model", "codex-mini")
        max_turns = adapter_cfg.get("max_turns", 25)
        cli_path = adapter_cfg.get("cli_path", "codex")
        adapter = CodexAdapter(
            model=model, cwd=workspace_dir, max_turns=max_turns,
            cli_path=cli_path,
        )
        print(f"Codex adapter: model={model}, max_turns={max_turns}")

    elif adapter_type == "custom":
        # Dynamic import: adapter.class = "module.path:ClassName"
        import importlib
        class_spec = adapter_cfg.get("class", "")
        if ":" not in class_spec:
            print(f"ERROR: custom adapter requires 'class' in 'module:ClassName' format")
            sys.exit(1)
        module_path, class_name = class_spec.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        adapter_cls = getattr(mod, class_name)
        adapter_kwargs = {k: v for k, v in adapter_cfg.items() if k not in ("type", "class")}
        adapter = adapter_cls(**adapter_kwargs)

    else:
        print(f"ERROR: unknown adapter type '{adapter_type}'")
        sys.exit(1)

    # Generate task stream
    from agingbench.generators.s5_generator import S5Generator
    from agingbench.generators.pressure_config import PressureConfig

    pressure_map = {
        "none": PressureConfig.none(),
        "light": PressureConfig.light(),
        "medium": PressureConfig.medium(),
        "heavy": PressureConfig.heavy(),
    }
    pressure = pressure_map[args.pressure]

    print(f"\nGenerating {args.domain} task stream (pressure={args.pressure})...")
    gen = S5Generator(seed=sut_cfg.get("seed", 42), domain=args.domain, pressure=pressure)
    generated_data = gen.generate(n_sessions=args.sessions)
    n_tasks = len(generated_data["task_stream"]["tasks"])
    n_probes = len(generated_data["recall_probes"]["probes"])
    n_facts = len(generated_data["facts_registry"])
    n_dep_pairs = len(generated_data.get("output_dependency_pairs", []))
    print(f"Generated: {n_tasks} tasks, {n_probes} recall probes, "
          f"{n_facts} facts, {n_dep_pairs} output_dependency pairs")

    # Run
    from agingbench.runner.s5_runner import S5Runner
    from agingbench.runner.trace import TraceLogger
    from agingbench.metrics.aging import summarize
    from agingbench.report.plot import plot_curve, compare_curves

    trace_path = output_dir / "trace.jsonl"

    with TraceLogger(str(trace_path)) as tracer:
        runner = S5Runner(
            adapter=adapter,
            tracer=tracer,
            sut_id=sut_id,
            session_length=args.session_length,
            generated_data=generated_data,
            reset_history=not args.no_reset,
        )
        result = runner.run(n_sessions=args.sessions, seed=sut_cfg.get("seed", 42))

    # Summarize
    stats = summarize(result.primary_curve)
    stats["scenario"] = "s5_self_planning"
    stats["sut_id"] = sut_id
    stats["adapter"] = adapter_type
    stats["domain"] = args.domain
    stats["n_sessions"] = args.sessions
    stats["session_length"] = args.session_length
    stats["recall_raw"] = result.raw.get("recall_raw", [])
    stats["task_raw"] = result.raw.get("task_raw", [])
    stats["recall_matrix"] = result.raw.get("recall_matrix", {})
    stats["lag_curves"] = result.raw.get("lag_curves", {})
    stats["workspace_snapshots"] = result.raw.get("workspace_snapshots", [])
    stats["session_results"] = result.session_results

    # Score execution drift if output dependency pairs were generated
    dep_pairs = generated_data.get("output_dependency_pairs", [])
    if dep_pairs:
        from agingbench.metrics.dependency_scorer import score_execution_drift
        drift_metrics = score_execution_drift(result.session_results, dep_pairs)
        stats["execution_drift_metrics"] = drift_metrics
        with open(output_dir / "execution_drift_metrics.json", "w") as f:
            json.dump(drift_metrics, f, indent=2)

    stats.setdefault("scenario", "s5_self_planning")
    stats.setdefault("metric_group", "G2")
    stats.setdefault("headline_metric", "recall_accuracy")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Plot
    curves = [result.primary_curve]
    labels = ["recall_accuracy"]
    if "task_accuracy" in (result.secondary_curves or {}):
        curves.append(result.secondary_curves["task_accuracy"])
        labels.append("task_accuracy")

    compare_curves(
        curves,
        str(output_dir / "aging_curve.png"),
        title=f"S7 Aging — {sut_id} ({args.domain})",
        labels=labels,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Recall: m0={stats['m0']:.3f}  m_final={stats['m_final']:.3f}  "
          f"slope={stats['decay_slope']:.5f}  half_life={stats['half_life']}")
    print(f"\n  Per-session:")
    for sr in result.session_results:
        km = sr.get("keyword_m")
        km_str = f"kw_m={km:.3f}" if km is not None else "kw_m=n/a"
        print(f"    Block {sr['session']:2d}: task={sr['task_accuracy']:.3f}  "
              f"recall={sr['recall_accuracy']:.3f}  {km_str}  "
              f"files={len(sr.get('n_files_in_workspace', []))}  "
              f"proactive={sr['proactive_check_rate']:.2f}")
    if dep_pairs:
        dm = stats.get("execution_drift_metrics", {})
        print(f"\n  Execution drift: overall={dm.get('overall_execution_drift', 'n/a')}  "
              f"n_pairs={dm.get('n_pairs_total', 0)}")
        for dist, d in dm.get("drift_by_distance", {}).items():
            print(f"    dist={dist}: prod={d['producer_accuracy']:.3f}  "
                  f"cons={d['consumer_accuracy']:.3f}  drift={d['drift']:.3f}  "
                  f"n={d['n_pairs']}")
    print(f"\n  Output: {output_dir}")


if __name__ == "__main__":
    main()
