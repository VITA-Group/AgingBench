#!/usr/bin/env python3
"""
run_s1.py — End-to-end entry point for S1 (Research Literature).

Thin wrapper around agingbench.cli._run_s1 (the canonical path) so results
match `agingbench run --scenario s1_research_literature --sut <yaml>` exactly.
Exposes the scenario-specific knobs the generic CLI doesn't surface
(--cycles, --score-via-response, --compare).

Default mode is `--generated` (programmatic generator, seed-dependent,
reproducible across machines). Pass `--no-generated` to use curated disk data.

Usage:
  python scripts/run_s1.py --sut <yaml> [--cycles 8] [--no-generated]
  python scripts/run_s1.py --sut <yaml> --score-via-response
  python scripts/run_s1.py --compare run_a/metrics.json run_b/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


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


def compare_mode(metrics_paths: list[str]) -> None:
    """Overlay aging curves from 2+ prior metrics.json files."""
    from agingbench.metrics.aging import AgingCurve
    from agingbench.report.plot import compare_curves
    results, curves = [], []
    for p in metrics_paths:
        with open(p) as f:
            m = json.load(f)
        results.append(m)
        exps, scores = zip(*m["checkpoints"])
        curves.append(AgingCurve(
            exposures=list(exps), scores=list(scores),
            scenario=m["scenario"], sut_id=m["sut_id"],
        ))
    print_table(results)
    out_dir = Path(metrics_paths[0]).parent.parent
    compare_curves(curves, str(out_dir / "p2_comparison.png"),
                   title="P2 Aging — SUT Comparison")


def main():
    parser = argparse.ArgumentParser(description="Run S1 — Research Literature")
    parser.add_argument("--sut", help="Path to SUT YAML config")
    parser.add_argument("--cycles", "--sessions", dest="cycles", type=int, default=8,
                        help="Number of S1 cycles/sessions (default: 8). Both flag "
                             "names are accepted.")
    parser.add_argument("--output", default="",
                        help="Output directory (default: experiments/results/<sut_id>)")
    parser.add_argument("--score-via-response", action="store_true",
                        help="Ask the LLM each probe with current memory as context and "
                             "score the response (end-to-end W+R+U) instead of the "
                             "memory-based substring check.")
    parser.add_argument("--generated", action=argparse.BooleanOptionalAction, default=True,
                        help="Use S1Generator (seed-dependent) instead of curated disk data. "
                             "Default: --generated.")
    parser.add_argument("--compare", nargs="+", metavar="METRICS_JSON",
                        help="Compare mode: pass 2+ metrics.json paths to overlay curves")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))

    if args.compare:
        compare_mode(args.compare)
        return

    if not args.sut:
        parser.print_help()
        sys.exit(1)

    import yaml
    with open(args.sut) as f:
        sut_cfg = yaml.safe_load(f)

    sut_id = sut_cfg["sut_id"]
    output_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "experiments" / "results" / Path(args.sut).stem
    )
    scenario_cfg = {"n_cycles": args.cycles}

    from agingbench.cli import _run_s1

    print(f"{'='*60}")
    print(f"S1 — Research Literature Agent")
    print(f"SUT: {sut_id}  Cycles: {args.cycles}  Generated: {args.generated}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    stats = _run_s1(sut_cfg, scenario_cfg, output_dir, args.cycles,
                    generated=args.generated, gen_sessions=args.cycles,
                    score_via_response=args.score_via_response)

    print_table([stats])
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
