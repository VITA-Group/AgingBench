"""
examples/langfuse_adapter.py — Translate an AgingCard JSON into Langfuse /
OpenTelemetry trace-event format.

Usage:
    python examples/langfuse_adapter.py \
        --card experiments/results/<run-dir>/aging_card.json \
        --out /tmp/langfuse_events.jsonl

Status: v1 skeleton. Demonstrates the trace-style emission pattern. A
real Langfuse deployment would batch these via the Langfuse SDK + project
keys; this adapter just produces the per-event JSONL you'd publish.

References:
- AgingCard schema: agingbench/metrics/aging_card_schema.json
- Langfuse docs: https://langfuse.com/docs
- OTLP spec: https://opentelemetry.io/docs/specs/otlp/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SPAN_KIND_AGING_RUN = "agingbench.run"
SPAN_KIND_MECHANISM = "agingbench.mechanism"


def aging_card_to_langfuse_events(card: dict) -> list[dict]:
    """
    Translate one AgingCard into Langfuse-style events.

    Strategy: emit one parent span per AgingCard "run" + child spans per
    aging mechanism + child spans per cost/efficiency block. The parent
    span ID is derived from `run_id` so multiple runs are correlatable in
    the trace view.
    """
    out = []
    run_id = card.get("run_id", "?")
    sut = card.get("sut") or {}

    # Parent run span.
    out.append({
        "id": run_id,
        "parent_id": None,
        "name": f"AgingBench {card.get('scenario','?')} ({sut.get('sut_id','?')})",
        "kind": SPAN_KIND_AGING_RUN,
        "start_time": card.get("generated_at"),
        "attributes": {
            "scenario": card.get("scenario"),
            "scenario_version": card.get("scenario_version"),
            "suite_id": card.get("suite_id"),
            "sut.sut_id": sut.get("sut_id"),
            "sut.model_provider": sut.get("model_provider"),
            "sut.model_id": sut.get("model_id"),
            "sut.memory_policy_type": sut.get("memory_policy_type"),
            "seed": card.get("seed"),
            "n_sessions": card.get("n_sessions"),
            "headline.m0": (card.get("headline") or {}).get("m0"),
            "headline.m_final": (card.get("headline") or {}).get("m_final"),
            "headline.half_life": (card.get("headline") or {}).get("half_life"),
            "headline.decay_slope": (card.get("headline") or {}).get("decay_slope"),
            "aging_detected": (card.get("headline") or {}).get("aging_detected"),
        },
        "metadata": {
            "agingbench_version": (card.get("provenance") or {}).get("agingbench_version"),
            "git_sha": (card.get("provenance") or {}).get("git_sha"),
            "schema_version": card.get("schema_version"),
        },
    })

    # Per-mechanism child spans.
    mech = card.get("mechanism_metrics") or {}
    for i, (mech_name, mech_data) in enumerate(mech.items()):
        if not isinstance(mech_data, dict):
            continue
        out.append({
            "id": f"{run_id}.mech.{mech_name}",
            "parent_id": run_id,
            "name": f"mechanism: {mech_name}",
            "kind": SPAN_KIND_MECHANISM,
            "attributes": {
                f"mech.{mech_name}.{k}": v for k, v in mech_data.items()
            },
        })

    # Cost/efficiency child span (one).
    cost = card.get("cost_and_efficiency") or {}
    if cost:
        out.append({
            "id": f"{run_id}.cost",
            "parent_id": run_id,
            "name": "cost_and_efficiency",
            "kind": "agingbench.cost",
            "attributes": {f"cost.{k}": v for k, v in cost.items()},
        })

    return out


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
    events = aging_card_to_langfuse_events(card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    print(f"Wrote {len(events)} Langfuse events to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
