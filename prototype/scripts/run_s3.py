#!/usr/bin/env python3
"""
Run S3 scenario — Project Knowledge Base Agent.

Default mode is `--generated` (programmatic generator, seed-dependent, reproducible
across machines). Pass `--no-generated` to fall back to the curated disk data
(queries.json, gold_timeline.json, transcripts.json).

Usage:
    # Generated mode (default)
    python scripts/run_s3.py --sut agingbench/registry/suts/llama3/lossy_compress.yaml
    python scripts/run_s3.py --sut agingbench/registry/suts/llama3/lossy_compress.yaml --sessions 6

    # Curated-data mode (legacy disk-loaded canned scenario)
    python scripts/run_s3.py --sut <yaml> --no-generated
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run S3 scenario")
    parser.add_argument("--sut", required=True, help="Path to SUT YAML config")
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--output", default="")
    parser.add_argument("--generated", action=argparse.BooleanOptionalAction, default=True,
                        help="Use programmatic generator instead of curated data. "
                             "Default: --generated. Pass --no-generated to use curated.")
    parser.add_argument("--held-out-probes", type=int, default=2,
                        help="Number of held-out probes per session, sampled from "
                             "strictly-earlier sessions, disjoint from in-channel "
                             "queries. Asked and scored, but never written to memory. "
                             "Default 2 = held-out channel ON (clean substrate-decay "
                             "measurement). Pass --held-out-probes 0 to disable.")
    args = parser.parse_args()

    import yaml
    with open(args.sut) as f:
        sut_cfg = yaml.safe_load(f)

    sut_id = sut_cfg["sut_id"]
    output_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "experiments" / "results" / "s3_knowledge_base" / sut_id
    )

    scenario_cfg = {"n_cycles": args.sessions}

    sys.path.insert(0, str(PROJECT_ROOT))
    from agingbench.cli import _run_s3

    print(f"{'='*60}")
    print(f"S3 — Project Knowledge Base Agent")
    print(f"SUT: {sut_id}  Sessions: {args.sessions}  Generated: {args.generated}")
    print(f"Held-out probes/session: {args.held_out_probes}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    stats = _run_s3(sut_cfg, scenario_cfg, output_dir, args.sessions,
                     generated=args.generated, gen_sessions=args.sessions,
                     held_out_probes_per_session=args.held_out_probes)

    print(f"\n{'='*60}")
    print(f"RESULTS: m0={stats['m0']:.3f}  m_final={stats['m_final']:.3f}  "
          f"slope={stats['decay_slope']:.5f}")
    if stats.get("held_out_acc_raw"):
        ho = stats["held_out_acc_raw"]
        ic = stats["query_acc_raw"]
        # Align on shared exposures; held-out skips session 0 so it can be shorter.
        ho_by_t = {t: v for t, v in ho}
        ic_by_t = {t: v for t, v in ic}
        shared = sorted(set(ho_by_t) & set(ic_by_t))
        if shared:
            ho_mean = sum(ho_by_t[t] for t in shared) / len(shared)
            ic_mean = sum(ic_by_t[t] for t in shared) / len(shared)
            print(
                f"CHANNELS (on shared exposures n={len(shared)}):  "
                f"in_channel_acc_mean={ic_mean:.3f}  held_out_acc_mean={ho_mean:.3f}  "
                f"delta_testing_effect={ic_mean - ho_mean:+.3f}"
            )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
