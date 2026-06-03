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
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    stats = _run_s3(sut_cfg, scenario_cfg, output_dir, args.sessions,
                     generated=args.generated, gen_sessions=args.sessions)

    print(f"\n{'='*60}")
    print(f"RESULTS: m0={stats['m0']:.3f}  m_final={stats['m_final']:.3f}  "
          f"slope={stats['decay_slope']:.5f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
