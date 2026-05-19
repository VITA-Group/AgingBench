"""
agingbench._lite_entrypoint — Console-script entry point for the
``agingbench-lite`` PyPI package.

This is a thin wrapper around `agingbench.cli.main` that:
  1. Validates the requested suite is one of the lite-supported suites
     (defaults to `lite`).
  2. Restricts available scenario IDs to the lite subset (S1, S2, S7).
  3. Forwards remaining arguments to the main CLI.

The full `agingbench` CLI is unchanged. AgingBench-Lite is purely a
narrower entry point so product teams can `pip install agingbench-lite`
without the heavy ML deps.
"""
from __future__ import annotations

import os
import sys


_LITE_SUITES = {"lite", "core"}
_LITE_SCENARIOS = {
    "s1_research_literature",
    "s2_lifestyle_assistant",
    "s7_research_notes",
}


def main(argv: list[str] | None = None) -> int:
    """Validate scope then delegate to ``agingbench.cli.main``."""
    argv = list(argv if argv is not None else sys.argv[1:])

    # Cheap scope check: reject suites/scenarios that aren't in lite.
    if argv and argv[0] in ("run", "compare", "list-suites"):
        scope_err = _validate_scope(argv)
        if scope_err:
            print(f"[agingbench-lite] {scope_err}", file=sys.stderr)
            print(
                "[agingbench-lite] For the full benchmark, install the parent "
                "`agingbench` package and use `agingbench run ...`.",
                file=sys.stderr,
            )
            return 2

    # Defer to the full CLI.
    from agingbench.cli import main as full_main

    # Set a sentinel so downstream code (telemetry, error reports) can
    # see this is the lite entrypoint.
    os.environ.setdefault("AGINGBENCH_LITE", "1")
    return full_main(argv)


def _validate_scope(argv: list[str]) -> str | None:
    """Return an error string if the args reference a non-lite suite/scenario."""
    for i, tok in enumerate(argv):
        if tok == "--suite" and i + 1 < len(argv):
            if argv[i + 1] not in _LITE_SUITES:
                return (
                    f"suite '{argv[i + 1]}' is not in the lite subset "
                    f"(allowed: {sorted(_LITE_SUITES)})"
                )
        if tok == "--scenario" and i + 1 < len(argv):
            if argv[i + 1] not in _LITE_SCENARIOS:
                return (
                    f"scenario '{argv[i + 1]}' is not in the lite subset "
                    f"(allowed: {sorted(_LITE_SCENARIOS)})"
                )
    return None


if __name__ == "__main__":
    sys.exit(main())
