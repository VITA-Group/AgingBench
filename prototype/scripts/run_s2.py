#!/usr/bin/env python3
"""
Run S2 scenario — Personal Finance & Lifestyle Assistant.

Thin wrapper around agingbench.cli._run_s2 (the canonical path) so results
match `agingbench run --scenario s2_lifestyle_assistant --sut <yaml>` exactly.

Default mode is `--generated` (programmatic generator, seed-dependent,
reproducible across machines). Pass `--no-generated` to use curated disk data.

Usage:
    python scripts/run_s2.py --sut <yaml> [--sessions 10] [--no-generated]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run S2 scenario")
    parser.add_argument("--sut", required=True, help="Path to SUT YAML config")
    parser.add_argument("--sessions", type=int, default=10, help="Number of sessions (default: 10)")
    parser.add_argument("--output", default="", help="Output directory")
    parser.add_argument("--generated", action=argparse.BooleanOptionalAction, default=True,
                        help="Use programmatic generator instead of curated data. "
                             "Default: --generated. Pass --no-generated to use curated.")
    args = parser.parse_args()

    import yaml
    with open(args.sut) as f:
        sut_cfg = yaml.safe_load(f)

    sut_id = sut_cfg["sut_id"]
    output_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "experiments" / "results" / "s2_lifestyle_assistant" / sut_id
    )
    scenario_cfg = {"n_cycles": args.sessions}

    sys.path.insert(0, str(PROJECT_ROOT))
    from agingbench.cli import _run_s2

    print(f"{'='*60}")
    print(f"S2 — Personal Finance & Lifestyle Assistant")
    print(f"SUT: {sut_id}  Sessions: {args.sessions}  Generated: {args.generated}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    stats = _run_s2(sut_cfg, scenario_cfg, output_dir, args.sessions,
                    generated=args.generated, gen_sessions=args.sessions)

    print(f"\n{'='*60}")
    print(f"RESULTS: m0={stats['m0']:.3f}  m_final={stats['m_final']:.3f}  "
          f"slope={stats['decay_slope']:.5f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
