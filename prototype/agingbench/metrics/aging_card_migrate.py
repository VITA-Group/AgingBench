"""
agingbench/metrics/aging_card_migrate.py — Forward-migration utility for
AgingCard JSONs across schema versions.

Usage (CLI):
    python -m agingbench.metrics.aging_card_migrate <in.json> [--out <out.json>]
    Migrates the input card to the current AGING_CARD_SCHEMA_VERSION
    by walking the migration chain. Prints the migrated card to stdout
    if --out is omitted.

Migration chain
---------------
Each migration step is `migrate_v{old}_to_v{new}(card) -> card`. The
dispatcher walks the chain until the card's schema_version matches the
current target.

v1.0.0 is the initial release; no migrations exist yet. Future major
bumps register a step here. Minor and patch bumps do NOT require a
migration (the format is additive).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from agingbench.metrics.aging_card import AGING_CARD_SCHEMA_VERSION


# (from_version, to_version) -> callable
_MIGRATIONS: dict[tuple[str, str], Callable[[dict], dict]] = {}


def register(from_v: str, to_v: str):
    """Decorator to register a migration step."""
    def _wrap(fn):
        _MIGRATIONS[(from_v, to_v)] = fn
        return fn
    return _wrap


def migrate(card: dict, target_version: Optional[str] = None) -> dict:
    """Walk the migration chain until card['schema_version'] == target.

    If `target_version` is None, target the current
    AGING_CARD_SCHEMA_VERSION. Returns a NEW dict; does not mutate the input.
    """
    target = target_version or AGING_CARD_SCHEMA_VERSION
    current = dict(card)  # defensive copy
    seen = set()
    while current.get("schema_version") != target:
        sv = current.get("schema_version")
        if sv in seen:
            raise RuntimeError(f"migration loop detected at {sv!r}")
        seen.add(sv)
        step = _find_step_from(sv)
        if step is None:
            raise RuntimeError(
                f"no migration path from {sv!r} to {target!r}; "
                f"registered steps: {list(_MIGRATIONS.keys())}"
            )
        (from_v, to_v), fn = step
        current = fn(current)
        # The migration is responsible for stamping the new schema_version.
        if current.get("schema_version") == sv:
            raise RuntimeError(
                f"migration {from_v}->{to_v} did not update schema_version"
            )
    return current


def _find_step_from(sv: Optional[str]):
    """Pick a migration starting from `sv`. v1 has no steps; this is here
    so the chain logic is exercised by tests."""
    for (from_v, to_v), fn in _MIGRATIONS.items():
        if from_v == sv:
            return (from_v, to_v), fn
    return None


# Placeholder for future migrations. Pattern (example, do not enable):
#
# @register("1.0.0", "1.1.0")
# def migrate_v1_0_0_to_v1_1_0(card: dict) -> dict:
#     out = dict(card)
#     # ... transform additive fields ...
#     out["schema_version"] = "1.1.0"
#     return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="AgingCard JSON to migrate")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write migrated JSON here (default: stdout)")
    parser.add_argument("--target", default=None,
                        help="Target schema_version (default: current)")
    args = parser.parse_args(argv)

    with args.input.open("r") as f:
        card = json.load(f)

    try:
        migrated = migrate(card, target_version=args.target)
    except RuntimeError as e:
        print(f"[error] migration failed: {e}", file=sys.stderr)
        return 1

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(migrated, f, indent=2, sort_keys=True)
        print(
            f"migrated {args.input.name}: "
            f"{card.get('schema_version')!r} -> {migrated.get('schema_version')!r} "
            f"-> {args.out}",
        )
    else:
        json.dump(migrated, sys.stdout, indent=2, sort_keys=True)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
