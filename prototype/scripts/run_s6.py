#!/usr/bin/env python3
"""
Run S6 scenario — Naturalistic Aging (WebArena-derived multi-domain workflows).

Thin wrapper around agingbench.cli._run_s6 (the canonical path) so results
match `agingbench run --scenario s6_naturalistic --sut <yaml>` exactly
(headline metric: recall_compression). Adds the script-only conveniences the
generic CLI doesn't surface: --model-family (run every SUT for a family) and
--seeds (repeat over seeds and aggregate the summary stats).

Usage:
    python scripts/run_s6.py --sut <yaml> [--sessions 15] [--no-generated]
    python scripts/run_s6.py --model-family gpt4o --sessions 5
    python scripts/run_s6.py --sut <yaml> --seeds 3
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def _find_suts_by_family(family: str) -> list[Path]:
    sut_dir = PROJECT_ROOT / "agingbench" / "registry" / "suts"
    matches = sorted(sut_dir.glob(f"{family}_*.yaml"))
    if not matches:
        matches = sorted(Path("agingbench/registry/suts").glob(f"{family}_*.yaml"))
    return matches


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return m, var ** 0.5


def run_single_sut(sut_path: Path, n_sessions: int, output_base: str,
                   generated: bool, n_seeds: int, multi_sut: bool) -> dict:
    import yaml
    from agingbench.cli import _run_s6

    with open(sut_path) as f:
        sut_cfg = yaml.safe_load(f)
    sut_id = sut_cfg["sut_id"]

    if output_base and not multi_sut:
        out_dir = Path(output_base)
    elif output_base:
        out_dir = Path(output_base) / sut_id
    else:
        out_dir = PROJECT_ROOT / "experiments" / "results" / "s6_naturalistic" / sut_id

    print(f"\n{'='*60}")
    print(f"S6 — Naturalistic Aging | SUT: {sut_id} | sessions={n_sessions} seeds={n_seeds}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    base_seed = sut_cfg.get("seed", 42)
    per_seed = []
    for seed_i in range(n_seeds):
        seed_cfg = dict(sut_cfg)
        seed_cfg["seed"] = base_seed + seed_i
        seed_dir = (out_dir / f"seed_{seed_i}") if n_seeds > 1 else out_dir
        scenario_cfg = {"n_cycles": n_sessions}
        stats = _run_s6(seed_cfg, scenario_cfg, seed_dir, n_sessions,
                        generated=generated, gen_sessions=n_sessions)
        per_seed.append(stats)

    if n_seeds > 1:
        # Aggregate the scalar summary across seeds (per-seed metrics.json holds
        # the full curves). NB: this replaces the old curve-band CI plot.
        agg = {"scenario": "s6_naturalistic", "sut_id": sut_id, "n_seeds": n_seeds}
        for key in ("m0", "m_final", "decay_slope"):
            mean, std = _mean_std([s[key] for s in per_seed])
            agg[f"{key}_mean"], agg[f"{key}_std"] = mean, std
        with open(out_dir / "metrics_aggregated.json", "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n  Aggregated ({n_seeds} seeds): "
              f"m0={agg['m0_mean']:.3f}±{agg['m0_std']:.3f}  "
              f"m_final={agg['m_final_mean']:.3f}±{agg['m_final_std']:.3f}  "
              f"slope={agg['decay_slope_mean']:.5f}±{agg['decay_slope_std']:.5f}")
        return {"sut_id": sut_id, "m0": agg["m0_mean"],
                "m_final": agg["m_final_mean"], "decay_slope": agg["decay_slope_mean"]}

    return per_seed[-1]


def main():
    parser = argparse.ArgumentParser(description="Run S6 — Naturalistic Aging")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sut", help="Path to a single SUT YAML config")
    group.add_argument("--model-family", help="Model family prefix (e.g. 'gpt4o')")
    parser.add_argument("--sessions", type=int, default=15, help="Number of sessions (default: 15)")
    parser.add_argument("--output", default="", help="Output directory base")
    parser.add_argument("--generated", action=argparse.BooleanOptionalAction, default=True,
                        help="Use programmatic generator instead of curated data. "
                             "Default: --generated.")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds (default: 1)")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))

    if args.sut:
        sut_paths = [Path(args.sut)]
    else:
        sut_paths = _find_suts_by_family(args.model_family)
        if not sut_paths:
            print(f"[error] No SUTs found for family '{args.model_family}'")
            sys.exit(1)

    multi_sut = len(sut_paths) > 1
    summary = []
    for sut_path in sut_paths:
        try:
            summary.append(run_single_sut(sut_path, args.sessions, args.output,
                                           args.generated, args.seeds, multi_sut))
        except Exception as e:
            print(f"\n[ERROR] Failed on {sut_path.name}: {e}")
            import traceback
            traceback.print_exc()

    if len(summary) > 1:
        print(f"\n{'='*70}")
        print(f"{'SUT':<40} {'m0':>5} {'m_final':>7} {'slope':>9}")
        print(f"{'-'*70}")
        for r in summary:
            print(f"  {r['sut_id']:<38} {r['m0']:>5.3f} {r['m_final']:>7.3f} {r['decay_slope']:>9.5f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
