#!/usr/bin/env python3
"""
run_s8.py — End-to-end entry point for S8 (SWE-bench-Aging, Tier-2).

S8 runs a long-running developer agent on a curated chain of real Django
GitHub issues from SWE-bench-Verified. Each session uses a SWE-bench-
pre-built Docker container holding Django at the issue's pre-resolution
commit; verification = upstream Django test suite + injected synthetic
consistency tests.

All Tier-2 plumbing (Docker per-session container, agent adapter,
verification routing, four-mechanism probes) is handled by
`agingbench run`; this wrapper is a thin pass-through provided for
consistency with run_s1..s7.

Prerequisites:
    Docker images pre-pulled — see the snippet in
    agingbench/scenarios/s8_swe_bench/README.md (~24 GB cache).

Usage:
    # Claude Code Sonnet 4.6
    python run_s8.py \\
        --sut agingbench/registry/suts/claude_code/claude_code_sonnet46_s8.yaml \\
        --sessions 8 --card

For all options run:
    agingbench run --help
"""
import sys
from agingbench.cli import main as cli_main


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        sys.argv = [sys.argv[0], "run", "--help"]
    else:
        sys.argv = [sys.argv[0], "run", "--scenario", "s8_swe_bench", *args]
    return cli_main() or 0


if __name__ == "__main__":
    sys.exit(main())
