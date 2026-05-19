"""
examples/langsmith_adapter.py — Translate an AgingCard JSON into LangSmith
dataset records.

Usage:
    python examples/langsmith_adapter.py \
        --card experiments/results/<run-dir>/aging_card.json \
        --out /tmp/langsmith_dataset.jsonl

Status: v1 skeleton. Demonstrates how to feed AgingCard data into the
LangSmith dataset/run abstractions. The actual LangSmith API requires
auth + a project; this adapter emits the JSONL records you'd `langsmith
upload`-ish. Adapt to your team's LangSmith deployment.

References:
- AgingCard schema: agingbench/metrics/aging_card_schema.json
- LangSmith docs: https://docs.smith.langchain.com/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def aging_card_to_langsmith_records(card: dict) -> list[dict]:
    """
    Translate one AgingCard into LangSmith dataset records (one per
    mechanism), so a LangSmith eval can plot mechanism-level aging
    across model versions / memory policies / time.
    """
    out = []
    base_inputs = {
        "scenario": card.get("scenario"),
        "sut_id": (card.get("sut") or {}).get("sut_id"),
        "model_id": (card.get("sut") or {}).get("model_id"),
        "memory_policy": (card.get("sut") or {}).get("memory_policy_type"),
        "seed": card.get("seed"),
        "n_sessions": card.get("n_sessions"),
    }
    base_meta = {
        "card_type": card.get("card_type"),
        "schema_version": card.get("schema_version"),
        "suite_id": card.get("suite_id"),
        "run_id": card.get("run_id"),
        "generated_at": card.get("generated_at"),
        "provenance": card.get("provenance"),
    }

    mech = card.get("mechanism_metrics") or {}
    for mech_name, mech_data in mech.items():
        if not isinstance(mech_data, dict):
            continue
        out.append({
            "inputs": {**base_inputs, "mechanism": mech_name},
            "outputs": mech_data,
            "reference": _ideal_for_mechanism(mech_name),
            "metadata": base_meta,
        })
    # Plus one record for the headline aging summary.
    out.append({
        "inputs": {**base_inputs, "mechanism": "_headline"},
        "outputs": card.get("headline") or {},
        "reference": {"m_final": 1.0, "decay_slope": 0.0, "aging_detected": False},
        "metadata": base_meta,
    })
    return out


def _ideal_for_mechanism(mech_name: str) -> dict:
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
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.card.is_file():
        print(f"[error] AgingCard not found: {args.card}", file=sys.stderr)
        return 1
    with args.card.open("r") as f:
        card = json.load(f)
    records = aging_card_to_langsmith_records(card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} LangSmith records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
