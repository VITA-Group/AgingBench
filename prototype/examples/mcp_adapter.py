"""
examples/mcp_adapter.py — Translate an AgingCard JSON into MCP-style
tool / memory event records.

Usage:
    python examples/mcp_adapter.py \
        --card experiments/results/<run-dir>/aging_card.json \
        --out /tmp/mcp_events.jsonl

Status: v1 skeleton. Demonstrates the event-emission pattern an MCP
server (or MCP-aware observability backend) could ingest to receive
AgingBench aging signals alongside the agent's tool/memory traffic.
Real MCP integration uses the framed JSON-RPC protocol over stdio or
WebSocket; this adapter just produces the per-event JSON records you'd
publish to that channel. Extend with real `mcp` package transports.

References:
- AgingCard schema: agingbench/metrics/aging_card_schema.json
- MCP spec: https://modelcontextprotocol.io/specification
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk

Why this exists in v1.1:
- post_release.md called out MCP-style tool/memory events as one of the
  four integrations the result format should plug into naturally.
- AgingCards carry per-mechanism scores that map cleanly onto MCP's
  notion of named tool resources; emitting them as MCP events lets an
  MCP-aware backend correlate AgingBench's longitudinal scores with
  the agent's live tool-call stream in a single ingestion pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# MCP-style event names: namespaced under agingbench/* so MCP backends
# can filter on prefix without ambiguity against the agent's own events.
EVENT_RUN_SUMMARY   = "agingbench/run_summary"
EVENT_MECHANISM     = "agingbench/mechanism"
EVENT_COST          = "agingbench/cost"
EVENT_CHECKPOINT    = "agingbench/checkpoint"


def aging_card_to_mcp_events(card: dict) -> list[dict]:
    """Translate one AgingCard into a stream of MCP-style event records.

    Strategy:
      1. One `run_summary` event carrying the high-level run identity +
         headline aging-curve statistics.
      2. One `mechanism` event per aging mechanism (compression /
         interference / revision / maintenance) with its sub-fields.
      3. One `cost` event with the cost_and_efficiency block.
      4. One `checkpoint` event per session (allows time-series correlation
         with the agent's tool-call traffic for the same session).

    Event records follow the JSON-RPC notification shape:
        { "jsonrpc": "2.0", "method": "<event>", "params": {...} }
    so any MCP-aware backend can consume them by name without further
    structural inference.
    """
    out: list[dict] = []
    run_id = card.get("run_id", "?")
    sut = card.get("sut") or {}
    headline = card.get("headline") or {}

    # 1. run_summary
    out.append(_notification(EVENT_RUN_SUMMARY, {
        "run_id": run_id,
        "schema_version": card.get("schema_version"),
        "scenario": card.get("scenario"),
        "scenario_version": card.get("scenario_version"),
        "suite_id": card.get("suite_id"),
        "generated_at": card.get("generated_at"),
        "seed": card.get("seed"),
        "n_sessions": card.get("n_sessions"),
        "sut": {
            "sut_id": sut.get("sut_id"),
            "model_provider": sut.get("model_provider"),
            "model_id": sut.get("model_id"),
            "memory_policy_type": sut.get("memory_policy_type"),
        },
        "headline": {
            "metric_name": headline.get("metric_name"),
            "m0": headline.get("m0"),
            "m_final": headline.get("m_final"),
            "half_life": headline.get("half_life"),
            "decay_slope": headline.get("decay_slope"),
            "aging_detected": headline.get("aging_detected"),
        },
        "provenance": card.get("provenance") or {},
    }))

    # 2. per-mechanism events
    mech = card.get("mechanism_metrics") or {}
    for mech_name, mech_data in mech.items():
        if not isinstance(mech_data, dict):
            continue
        out.append(_notification(EVENT_MECHANISM, {
            "run_id": run_id,
            "mechanism": mech_name,
            **{k: v for k, v in mech_data.items() if k != "trajectory"},
            # Trajectory can be large; surface it under its own key so
            # subscribers that don't want it can skip easily.
            "has_trajectory": "trajectory" in mech_data,
            "trajectory": mech_data.get("trajectory"),
        }))

    # 3. cost event
    cost = card.get("cost_and_efficiency") or {}
    if cost:
        out.append(_notification(EVENT_COST, {
            "run_id": run_id,
            **cost,
        }))

    # 4. one checkpoint event per session — enables time-series joins
    #    with the agent's MCP tool-call stream by session index.
    for cp in card.get("checkpoints") or []:
        if not isinstance(cp, (list, tuple)) or len(cp) < 2:
            continue
        out.append(_notification(EVENT_CHECKPOINT, {
            "run_id": run_id,
            "session": cp[0],
            "m": cp[1],
        }))

    return out


def _notification(method: str, params: dict) -> dict:
    """Wrap a payload in the JSON-RPC 2.0 notification shape MCP uses."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True, type=Path,
                        help="Path to an aging_card.json")
    parser.add_argument("--out", required=True, type=Path,
                        help="Where to write the MCP event JSONL")
    args = parser.parse_args(argv)

    if not args.card.is_file():
        print(f"[error] AgingCard not found: {args.card}", file=sys.stderr)
        return 1
    with args.card.open("r") as f:
        card = json.load(f)
    events = aging_card_to_mcp_events(card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    print(f"Wrote {len(events)} MCP events to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
