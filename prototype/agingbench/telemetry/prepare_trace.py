"""
Concatenate fragmented Claude Code conversation files into a single .jsonl
trace ready for upload to the website (or for direct use with
`trace_to_card_v11(...)`).

Why this exists
---------------
Claude Code stores each conversation as a separate `<uuid>.jsonl` file
under `~/.claude/projects/<encoded-cwd>/`. To see cross-session aging,
the events from all conversations must be merged into one trace, sorted
chronologically. This module does that with one call.

Other agent platforms (OpenAI Assistants, LangSmith, OpenHands) already
emit a single file per session via export, so they don't need this step.

Usage
-----
Python:

    from agingbench.telemetry import prepare_trace
    out = prepare_trace("~/.claude/projects/-my-project-dir")
    # → wrote ~/.claude/projects/-my-project-dir/agingbench_trace.jsonl

CLI (run as a module):

    python -m agingbench.telemetry.prepare_trace \\
        ~/.claude/projects/-my-project-dir \\
        -o ~/my_trace.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

PathLike = Union[str, Path]

# Common timestamp field names across agent platforms. Tried in order.
_TS_KEYS = ("timestamp", "time", "created_at", "ts")


def prepare_trace(
    source: Union[PathLike, Sequence[PathLike]],
    output: Optional[PathLike] = None,
    *,
    sort_by_timestamp: bool = True,
    glob_pattern: str = "*.jsonl",
) -> Path:
    """Concatenate JSONL trace files into a single uploadable trace.

    Parameters
    ----------
    source
        - A directory: glob `*.jsonl` (recursively if `glob_pattern` is
          `**/*.jsonl`).
        - A single file: no-op concatenation (still applies timestamp
          sort if requested).
        - A list of paths: concatenate the listed files in order.
    output
        Destination path. If None and source is a directory, writes
        `<source>/agingbench_trace.jsonl`. If None and source is a file,
        writes `<source>.prepared.jsonl`.
    sort_by_timestamp
        Sort events by their `timestamp` / `time` / `created_at` field
        if present. True by default (gives cleaner session boundaries).
    glob_pattern
        Used when source is a directory. Default `*.jsonl` (non-recursive).

    Returns
    -------
    Path
        Absolute path of the written trace file.
    """
    # 1. Resolve source list
    if isinstance(source, (list, tuple)):
        files = [Path(os.path.expanduser(str(p))) for p in source]
    else:
        src = Path(os.path.expanduser(str(source)))
        if src.is_dir():
            files = sorted(src.glob(glob_pattern))
        elif src.is_file():
            files = [src]
        else:
            raise FileNotFoundError(f"{src}: not a file or directory")

    if not files:
        raise ValueError(f"no .jsonl files found under {source!r}")

    # 2. Resolve output path
    if output is None:
        if isinstance(source, (list, tuple)):
            out = Path("agingbench_trace.jsonl").resolve()
        else:
            src = Path(os.path.expanduser(str(source)))
            if src.is_dir():
                out = src / "agingbench_trace.jsonl"
            else:
                out = src.with_suffix(".prepared.jsonl")
    else:
        out = Path(os.path.expanduser(str(output))).resolve()

    # 3. Read all events; pair each line with its extracted timestamp
    events: list[tuple[str, str, str]] = []  # (ts_key, source_file, raw_line)
    n_files = 0
    for f in files:
        if not f.is_file():
            continue
        n_files += 1
        try:
            text = f.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _extract_timestamp(obj) or ""
            events.append((ts, f.name, line))

    if not events:
        raise ValueError(
            f"no JSON events parsed from {n_files} files under {source!r}"
        )

    # 4. Sort by timestamp string (ISO-8601 sorts lexically as time)
    if sort_by_timestamp:
        events.sort(key=lambda x: (x[0] or ""))

    # 5. Write
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for _ts, _fn, raw_line in events:
            f.write(raw_line + "\n")

    return out


def _extract_timestamp(obj: dict) -> Optional[str]:
    """Pull a timestamp value out of an event dict.

    Handles flat (`timestamp`) and nested (`message.timestamp`) cases.
    """
    if not isinstance(obj, dict):
        return None
    for k in _TS_KEYS:
        v = obj.get(k)
        if isinstance(v, (str, int, float)):
            return str(v)
    # Nested under 'message' (Anthropic API style)
    msg = obj.get("message")
    if isinstance(msg, dict):
        for k in _TS_KEYS:
            v = msg.get(k)
            if isinstance(v, (str, int, float)):
                return str(v)
    return None


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agingbench.telemetry.prepare_trace",
        description=(
            "Concatenate Claude Code (or any other) JSONL conversation "
            "files into a single trace ready for upload to the AgingBench "
            "telemetry website."
        ),
    )
    parser.add_argument(
        "source",
        help=(
            "Directory of .jsonl files (e.g. ~/.claude/projects/<dir>) "
            "or a single .jsonl file."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output path. Default: <source>/agingbench_trace.jsonl "
            "for directories, <source>.prepared.jsonl for files."
        ),
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Don't sort events by timestamp (preserve file order).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories of source.",
    )
    args = parser.parse_args(argv)

    glob = "**/*.jsonl" if args.recursive else "*.jsonl"
    try:
        out = prepare_trace(
            args.source,
            output=args.output,
            sort_by_timestamp=not args.no_sort,
            glob_pattern=glob,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Count events for a friendly summary
    n_events = sum(1 for _ in out.open())
    print(f"wrote {out} ({n_events} events)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
