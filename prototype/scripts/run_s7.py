#!/usr/bin/env python3
"""
run_s7.py — End-to-end entry point for S7 (Research Notes, Tier-2).

S7 is a Tier-2 scenario: it drives a production CLI (OpenHands or
Claude Code) wrapped as an AgentAdapter against a research-notes
codebase. All Tier-2 plumbing (adapter loading, persistent workspace,
pytest verification) is handled by `agingbench run`; this wrapper is a
thin pass-through provided for consistency with run_s1..s6.

Usage:
    # OpenHands SUT
    python run_s7.py \\
        --sut agingbench/registry/suts/openhands/openhands_gpt4omini_s7.yaml \\
        --sessions 12 --card

    # Claude Code SUT
    python run_s7.py \\
        --sut agingbench/registry/suts/claude_code/claude_code_sonnet46_s7.yaml \\
        --sessions 12 --card

For all options run:
    agingbench run --help

(S7 was historically called S7+; renamed in v0.2 from S7+ -> S7 to free
"S7" for the Tier-2 production-CLI scenario.)
"""
import sys
from agingbench.cli import main as cli_main


def main() -> int:
    # Inject the scenario flag at position 1 so any user-supplied flags
    # (e.g. --sut, --sessions, --card, --output) are preserved verbatim.
    args = sys.argv[1:]
    # Allow `python run_s7.py --help` to fall through to the CLI's help.
    if not args or args[0] in {"-h", "--help"}:
        sys.argv = [sys.argv[0], "run", "--help"]
    else:
        sys.argv = [sys.argv[0], "run", "--scenario", "s7_research_notes", *args]
    return cli_main() or 0


if __name__ == "__main__":
    sys.exit(main())
