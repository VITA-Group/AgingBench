#!/usr/bin/env python3
"""
run_s5.py — End-to-end entry point for S5 (Self-Planning Notebook).

Thin wrapper around agingbench.cli._run_s5 (the canonical path) so results
match `agingbench run --scenario s5_self_planning --sut <yaml>` exactly.
Exposes the scenario-specific knobs the generic CLI doesn't surface
(--adapter, --domain, --workspace, --session-length, --no-reset, --pressure).

S5 always uses the programmatic generator. (Historically called S7; the runner
module is agingbench.runner.s5_runner.S5Runner.)

Usage:
    python scripts/run_s5.py --sut <yaml>
    python scripts/run_s5.py --sut <yaml> --adapter claude_code --domain coding
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run S5 — Self-Planning Notebook")
    parser.add_argument("--sut", required=True, help="Path to SUT YAML config")
    parser.add_argument("--sessions", type=int, default=10, help="Number of session blocks")
    parser.add_argument("--session-length", type=int, default=12, help="Interactions per block")
    parser.add_argument("--output", default="", help="Output directory")
    parser.add_argument("--adapter", default="react",
                        choices=["react", "claude_code", "codex", "openhands", "custom"],
                        help="Adapter type (overrides adapter.type in the yaml)")
    parser.add_argument("--domain", default="assistant",
                        choices=["assistant", "knowledge_base", "coding"],
                        help="Task domain")
    parser.add_argument("--workspace", default="", help="Workspace directory (default: <output>/workspace)")
    parser.add_argument("--no-reset", action="store_true",
                        help="Don't clear conversation history between blocks (continuous session)")
    parser.add_argument("--pressure", default="none",
                        choices=["none", "light", "medium", "heavy"],
                        help="Dependency pressure preset (used only when the yaml has no `pressure`)")
    args = parser.parse_args()

    import yaml
    with open(args.sut) as f:
        sut_cfg = yaml.safe_load(f)

    sut_id = sut_cfg.get("sut_id", "unknown")
    output_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "experiments" / "results" / "s5_self_planning" / sut_id
    )

    # Map script flags into the cfg dicts _run_s5 reads. The CLI flag wins for
    # the adapter type (matches the original run_s5 precedence).
    sut_cfg.setdefault("adapter", {})["type"] = args.adapter
    scenario_cfg = {
        "n_cycles": args.sessions,
        "domain": args.domain,
        "session_length": args.session_length,
        "reset_history": not args.no_reset,
        # _resolve_pressure prefers sut_cfg["pressure"]; this preset is the
        # fallback when the yaml declares none.
        "pressure": args.pressure,
    }
    if args.workspace:
        scenario_cfg["workspace_dir"] = args.workspace

    sys.path.insert(0, str(PROJECT_ROOT))
    from agingbench.cli import _run_s5

    print(f"{'='*60}")
    print(f"S5 — Self-Planning Notebook")
    print(f"SUT: {sut_id}  Adapter: {args.adapter}  Domain: {args.domain}")
    print(f"Sessions: {args.sessions} x {args.session_length}  Output: {output_dir}")
    print(f"{'='*60}")

    stats = _run_s5(sut_cfg, scenario_cfg, output_dir, args.sessions,
                    generated=True, gen_sessions=args.sessions)

    print(f"\n{'='*60}")
    print(f"RESULTS: m0={stats['m0']:.3f}  m_final={stats['m_final']:.3f}  "
          f"slope={stats['decay_slope']:.5f}  half_life={stats['half_life']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
