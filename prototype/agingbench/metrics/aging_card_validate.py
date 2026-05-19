"""
agingbench/metrics/aging_card_validate.py — Schema validator for AgingCard JSONs.

Usage (CLI):
    python -m agingbench.metrics.aging_card_validate <path-to-card.json> [<path2.json> ...]
    Exit code 0 if all cards validate, 1 otherwise.

Programmatic:
    from agingbench.metrics.aging_card_validate import validate_card_dict
    errors = validate_card_dict(card_dict)  # [] if valid

We do a hand-rolled validation rather than depending on the `jsonschema`
package to keep the lite distribution dependency-free. If `jsonschema` is
available, we use it as a stricter cross-check.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional


SCHEMA_PATH = Path(__file__).parent / "aging_card_schema.json"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Fields the v1 schema requires at the top level (kept in sync with the
# `required` array in aging_card_schema.json).
REQUIRED_TOP_LEVEL = [
    "schema_version", "card_type", "generated_at", "run_id",
    "scenario", "scenario_version", "suite_id",
    "sut", "seed", "n_sessions", "pressure",
    "headline", "mechanism_metrics", "cost_and_efficiency",
    "checkpoints", "provenance", "warnings", "links",
]

REQUIRED_MECHANISMS = ["compression", "interference", "revision", "maintenance"]


def validate_card_dict(card: dict) -> list[str]:
    """Return a list of human-readable error strings (empty if valid)."""
    errors: list[str] = []

    if not isinstance(card, dict):
        return ["top-level: card must be a JSON object"]

    # Required top-level keys
    for key in REQUIRED_TOP_LEVEL:
        if key not in card:
            errors.append(f"missing required field: {key!r}")

    # schema_version SemVer
    sv = card.get("schema_version")
    if isinstance(sv, str):
        if not SEMVER_RE.match(sv):
            errors.append(f"schema_version {sv!r} does not match X.Y.Z")
    elif sv is not None:
        errors.append("schema_version must be a string")

    # card_type pin
    if card.get("card_type") not in (None, "agingbench.AgingCard"):
        errors.append(f"card_type must be 'agingbench.AgingCard', got {card.get('card_type')!r}")

    # sut block
    sut = card.get("sut")
    if not isinstance(sut, dict):
        errors.append("sut must be an object")
    elif "sut_id" not in sut:
        errors.append("sut missing required field: sut_id")

    # mechanism_metrics block
    mech = card.get("mechanism_metrics")
    if not isinstance(mech, dict):
        errors.append("mechanism_metrics must be an object")
    else:
        for m in REQUIRED_MECHANISMS:
            if m not in mech:
                errors.append(f"mechanism_metrics missing required key: {m!r}")
            elif not isinstance(mech[m], dict):
                errors.append(f"mechanism_metrics[{m!r}] must be an object")

    # checkpoints shape
    chk = card.get("checkpoints")
    if not isinstance(chk, list):
        errors.append("checkpoints must be an array")
    else:
        for i, entry in enumerate(chk):
            if not isinstance(entry, list) or len(entry) != 2:
                errors.append(f"checkpoints[{i}] must be a 2-element array")

    # warnings + links shape
    if "warnings" in card and not isinstance(card["warnings"], list):
        errors.append("warnings must be an array of strings")
    if "links" in card and not isinstance(card["links"], dict):
        errors.append("links must be an object")

    # Optional cross-check with the jsonschema lib if installed.
    try:
        import jsonschema  # type: ignore
    except ImportError:
        jsonschema = None  # noqa: F841

    if "jsonschema" in sys.modules and SCHEMA_PATH.is_file():
        try:
            import jsonschema as _js  # type: ignore
            with SCHEMA_PATH.open("r") as f:
                schema = json.load(f)
            validator = _js.Draft202012Validator(schema)
            for err in validator.iter_errors(card):
                errors.append(f"jsonschema: {err.message} (at {list(err.path)})")
        except Exception as e:  # pylint: disable=broad-except
            errors.append(f"jsonschema validation crashed: {e}")

    return errors


def validate_card_path(path: Path) -> list[str]:
    """Read a card from disk and validate it. Path errors are caller's problem."""
    if not path.is_file():
        return [f"not a file: {path}"]
    try:
        with path.open("r") as f:
            card = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return [f"could not load {path}: {e}"]
    return validate_card_dict(card)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print(
            "Usage: python -m agingbench.metrics.aging_card_validate <card.json> [...]",
            file=sys.stderr,
        )
        return 2

    n_ok = 0
    n_fail = 0
    for arg in argv:
        path = Path(arg)
        errors = validate_card_path(path)
        if errors:
            n_fail += 1
            print(f"FAIL {path}:")
            for e in errors:
                print(f"  - {e}")
        else:
            n_ok += 1
            print(f"OK   {path}")

    print(f"\nSummary: {n_ok} ok, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
