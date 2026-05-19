"""
examples/openai_evals_adapter.py — Convert an AgingCard JSON into OpenAI Evals
input format.

Usage:
    python examples/openai_evals_adapter.py \
        --card experiments/results/<run-dir>/aging_card.json \
        --out /tmp/openai_evals_input.jsonl

Status: v1 skeleton. Demonstrates the schema-translation pattern. The
exact OpenAI Evals payload shape depends on which eval flavor you're
running (classification, model-graded, programmatic-score, etc.); this
skeleton emits one summary record per AgingCard mechanism. Extend as
needed for your specific eval.

References:
- AgingCard schema: agingbench/metrics/aging_card_schema.json
- OpenAI Evals docs: https://github.com/openai/evals
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def aging_card_to_openai_evals(card: dict) -> list[dict]:
    """
    Translate one AgingCard into a list of OpenAI Evals records (JSONL rows).

    Strategy: emit one record per aging mechanism, so a downstream eval
    can report per-mechanism aging severity. Each record exposes:
      - input: a stable identifier (scenario + sut + seed + mechanism)
      - ideal: the no-aging baseline value for the mechanism
      - sample: the observed value from this run
      - metadata: provenance + cost info

    A custom OpenAI Evals scorer can then read these records and compute
    severity per mechanism however the eval owner wishes.
    """
    out = []
    base_meta = {
        "card_type": card.get("card_type"),
        "schema_version": card.get("schema_version"),
        "scenario": card.get("scenario"),
        "sut_id": (card.get("sut") or {}).get("sut_id"),
        "seed": card.get("seed"),
        "n_sessions": card.get("n_sessions"),
        "provenance": card.get("provenance"),
    }
    cost = card.get("cost_and_efficiency") or {}
    base_meta["cost_total_usd"] = cost.get("total_cost_usd")
    base_meta["latency_ms_p50"] = cost.get("latency_ms_p50")

    mech = card.get("mechanism_metrics") or {}
    for mech_name, mech_data in mech.items():
        if not isinstance(mech_data, dict):
            continue
        record = {
            "input": _build_input_id(card, mech_name),
            "ideal": _ideal_for_mechanism(mech_name),
            "sample": mech_data,
            "metadata": {**base_meta, "mechanism": mech_name},
        }
        out.append(record)
    return out


def _build_input_id(card: dict, mechanism: str) -> str:
    sut = card.get("sut") or {}
    return (
        f"{card.get('scenario','?')}::"
        f"{sut.get('sut_id','?')}::"
        f"seed{card.get('seed','?')}::"
        f"mech-{mechanism}"
    )


def _ideal_for_mechanism(mech_name: str) -> dict:
    """Per-mechanism baseline values (no aging). Used by eval scorers."""
    if mech_name == "compression":
        return {"score": 1.0}
    if mech_name == "interference":
        return {"resistance": 1.0}
    if mech_name == "revision":
        return {"version_accuracy": 1.0, "accumulator_abs_error": 0.0}
    if mech_name == "maintenance":
        return {"delta": 0.0}
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True, type=Path, help="Path to aging_card.json")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    args = parser.parse_args(argv)

    if not args.card.is_file():
        print(f"[error] AgingCard not found: {args.card}", file=sys.stderr)
        return 1
    with args.card.open("r") as f:
        card = json.load(f)

    records = aging_card_to_openai_evals(card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} OpenAI Evals records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
