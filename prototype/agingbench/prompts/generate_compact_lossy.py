#!/usr/bin/env python3
"""
generate_compact_lossy.py — produce a budget-adjusted compaction prompt.

Generates a static ``compact_lossy_<N>w.txt`` whose "at most N words" target
scales with the SUT's session count. Run once per (n_sessions, per_session)
configuration before launching a sweep. The generated .txt is then referenced
by the SUT YAML's ``compaction_prompt`` field exactly like the existing
``compact_lossy_400w.txt`` — no runtime templating, no policy changes, no
runner changes.

Examples
--------

    # Canonical 12-session run at 35 words/session → 420-word prompt
    python -m agingbench.prompts.generate_compact_lossy \\
        --n-sessions 12 --per-session 35

    # 24-session run at the same per-session rate → 840-word prompt
    python -m agingbench.prompts.generate_compact_lossy \\
        --n-sessions 24 --per-session 35

    # Custom output path
    python -m agingbench.prompts.generate_compact_lossy \\
        --n-sessions 12 --per-session 35 \\
        --out experiments/prompts/compact_lossy_420w.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The content of the prompt — same shape as compact_lossy.txt, with the
# literal word target as the only thing the generator substitutes. Kept
# deliberately simple and scenario-agnostic.
TEMPLATE = """You are a knowledge manager. Summarize the following project specification into a brief paragraph of at most {word_limit} words. Focus on the most important points. Be concise.

DOCUMENT:
{{text}}

SUMMARY:
"""


def render(word_limit: int) -> str:
    return TEMPLATE.format(word_limit=word_limit)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sessions", type=int, required=True,
                   help="Number of sessions in the sweep this prompt will run on.")
    p.add_argument("--per-session", type=int, default=30,
                   help="Words of budget per session. Default 30 is a "
                        "scenario-neutral mid-range: above the cross-scenario "
                        "mean (~23) and median (~25), close to S6's calibration "
                        "and slightly tight for S3 (which used 33-35 in its "
                        "3-seed sweep). Override per-scenario when stricter "
                        "calibration matters (S1 paper_batches: ~12, S4: ~12, "
                        "S2: ~25, S3: ~35).")
    p.add_argument("--out", type=str, default=None,
                   help="Output file path. Default: derived from word_limit as "
                        "agingbench/prompts/compact_lossy_<N>w.txt + mirrored "
                        "to experiments/prompts/.")
    p.add_argument("--mirror-experiments", action="store_true", default=True,
                   help="Also write to experiments/prompts/ (default ON; "
                        "kept in sync with the package-internal copy).")
    args = p.parse_args(argv)

    word_limit = args.n_sessions * args.per_session
    content = render(word_limit)

    pkg_dir = Path(__file__).resolve().parent
    project_root = pkg_dir.parent.parent  # agingbench/prompts → project root

    if args.out:
        targets = [Path(args.out).resolve()]
    else:
        default_name = f"compact_lossy_{word_limit}w.txt"
        targets = [pkg_dir / default_name]
        if args.mirror_experiments:
            targets.append(project_root / "experiments" / "prompts" / default_name)

    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(content)
        print(f"  wrote {t}  ({word_limit}-word budget, {args.n_sessions} sessions × {args.per_session}/session)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
