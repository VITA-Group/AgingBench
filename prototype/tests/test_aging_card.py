"""Tests for AgingCard JSON output.

`aging_card.json` is a pure post-processor over the existing
`metrics.json` / `dependency_metrics.json` outputs: it reads the
inputs and writes a new file, never modifying either. These tests
pin that contract.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agingbench.metrics import (
    AGING_CARD_SCHEMA_VERSION,
    build_aging_card,
    build_and_write_aging_card,
)


FIXTURE_METRICS = {
    "scenario": "s1_research_literature",
    "sut_id": "haiku45_lossy_compress",
    "metric_group": "G2",
    "m0": 0.95,
    "m_final": 0.42,
    "half_life": 3.5,
    "decay_slope": -0.0752,
    "hazard_proxy": 0.22,
    "n_checkpoints": 10,
    "n_sessions": 10,
    "headline_metric": "keyword_recall",
    "aging_detected": True,
    "checkpoints": [[i, 0.95 - i * 0.05] for i in range(10)],
    "session_results": [
        {"session": i, "score": 0.95 - i * 0.05, "tokens": 1024}
        for i in range(10)
    ],
}

FIXTURE_DEP_METRICS = {
    "interference_resistance": 0.62,
    "version_accuracy": 0.51,
    "forget_accuracy": 0.7,
    "accumulator_metrics": {
        "mean_error": 2.4,
        "compounding_detected": True,
    },
}

FIXTURE_SUT = {
    "sut_id": "haiku45_lossy_compress",
    "seed": 42,
    "model": {"provider": "litellm", "model": "claude-haiku-4-5-20251001"},
    "memory_policy": {"type": "summarize_store"},
}


def test_build_aging_card_returns_dict():
    """Sanity: build_aging_card produces a dict, not None."""
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    assert isinstance(card, dict)


def test_aging_card_has_required_v1_fields():
    """v1.0.0 card must include all required top-level fields."""
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    required = {
        "schema_version", "card_type", "generated_at", "run_id",
        "scenario", "scenario_version", "suite_id",
        "sut", "seed", "n_sessions", "pressure",
        "headline", "mechanism_metrics", "cost_and_efficiency",
        "checkpoints", "provenance", "warnings", "links",
    }
    missing = required - set(card.keys())
    assert not missing, f"AgingCard missing required v1.0.0 fields: {missing}"


def test_aging_card_schema_version_pinned_to_1_0_0():
    """v1 ships schema_version 1.0.0; future bumps require explicit migration."""
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    assert card["schema_version"] == "1.0.0"
    assert AGING_CARD_SCHEMA_VERSION == "1.0.0"


def test_aging_card_card_type_pinned():
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    assert card["card_type"] == "agingbench.AgingCard"


def test_metrics_json_untouched(tmp_path: Path):
    """build_aging_card and write_aging_card must NOT mutate the inputs.

    Non-interference gate: existing scenario outputs (metrics.json,
    dependency_metrics.json) are read-only inputs for the post-processor.
    """
    metrics_in = copy.deepcopy(FIXTURE_METRICS)
    dep_in = copy.deepcopy(FIXTURE_DEP_METRICS)
    sut_in = copy.deepcopy(FIXTURE_SUT)

    card = build_aging_card(
        metrics=metrics_in,
        sut_cfg=sut_in,
        dependency_metrics=dep_in,
    )

    # Inputs unchanged.
    assert metrics_in == FIXTURE_METRICS, "build_aging_card mutated metrics input"
    assert dep_in == FIXTURE_DEP_METRICS, "build_aging_card mutated dependency_metrics input"
    assert sut_in == FIXTURE_SUT, "build_aging_card mutated sut_cfg input"
    assert isinstance(card, dict)


def test_build_and_write_does_not_overwrite_metrics_json(tmp_path: Path):
    """End-to-end: writing aging_card.json must not touch metrics.json on disk."""
    # Lay down fixture metrics.json and dependency_metrics.json
    metrics_path = tmp_path / "metrics.json"
    dep_path = tmp_path / "dependency_metrics.json"
    metrics_path.write_text(json.dumps(FIXTURE_METRICS, sort_keys=True))
    dep_path.write_text(json.dumps(FIXTURE_DEP_METRICS, sort_keys=True))

    metrics_bytes_before = metrics_path.read_bytes()
    dep_bytes_before = dep_path.read_bytes()

    card_path = build_and_write_aging_card(tmp_path, sut_cfg=FIXTURE_SUT)
    assert card_path is not None
    assert card_path.is_file()
    assert card_path.name == "aging_card.json"

    metrics_bytes_after = metrics_path.read_bytes()
    dep_bytes_after = dep_path.read_bytes()
    assert metrics_bytes_before == metrics_bytes_after, (
        "build_and_write_aging_card mutated metrics.json. Non-interference broken."
    )
    assert dep_bytes_before == dep_bytes_after, (
        "build_and_write_aging_card mutated dependency_metrics.json. Non-interference broken."
    )


def test_build_and_write_returns_none_when_metrics_missing(tmp_path: Path):
    """If metrics.json is absent, the helper returns None and writes nothing."""
    result = build_and_write_aging_card(tmp_path, sut_cfg=FIXTURE_SUT)
    assert result is None
    # No aging_card.json should have been written.
    assert not (tmp_path / "aging_card.json").exists()


def test_aging_card_headline_block_passes_through():
    """Headline block surfaces the same numbers as the underlying metrics.json."""
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    h = card["headline"]
    assert h["m0"] == 0.95
    assert h["m_final"] == 0.42
    assert h["half_life"] == 3.5
    assert h["aging_detected"] is True
    assert h["metric_name"] == "keyword_recall"


def test_aging_card_mechanism_metrics_block():
    """Mechanism block surfaces all four mechanisms with available data."""
    card = build_aging_card(
        metrics=FIXTURE_METRICS,
        sut_cfg=FIXTURE_SUT,
        dependency_metrics=FIXTURE_DEP_METRICS,
    )
    mech = card["mechanism_metrics"]
    assert set(mech.keys()) == {"compression", "interference", "revision", "maintenance"}
    assert mech["interference"]["resistance"] == 0.62
    assert mech["revision"]["version_accuracy"] == 0.51
    assert mech["revision"]["accumulator_abs_error"] == 2.4
    assert mech["compression"]["score"] == 0.42


def test_aging_card_sut_block():
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    sut = card["sut"]
    assert sut["sut_id"] == "haiku45_lossy_compress"
    assert sut["model_provider"] == "litellm"
    assert sut["model_id"] == "claude-haiku-4-5-20251001"
    assert sut["memory_policy_type"] == "summarize_store"


def test_aging_card_pressure_block_from_to_dict_protocol():
    """If pressure has a to_dict() method, build_aging_card extracts it."""
    class _FakePressure:
        def to_dict(self):
            return {"preset": "medium", "tokens_per_session": 2000}

    card = build_aging_card(
        metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT, pressure=_FakePressure()
    )
    assert card["pressure"]["preset"] == "medium"
    assert card["pressure"]["tokens_per_session"] == 2000


def test_aging_card_serializable_to_json():
    """The card must round-trip through json.dumps/loads."""
    card = build_aging_card(
        metrics=FIXTURE_METRICS,
        sut_cfg=FIXTURE_SUT,
        dependency_metrics=FIXTURE_DEP_METRICS,
    )
    serialized = json.dumps(card)
    deserialized = json.loads(serialized)
    assert deserialized == card


# ----------------------------------------------------------------------
# v0.3 cost-block fix: aggregate from trace.jsonl when metrics.json
# lacks the cost fields. Previously cost_and_efficiency was always null
# in emitted cards because the runner-side aggregation path was missing.
# These tests pin the new fallback path so it doesn't regress.
# ----------------------------------------------------------------------

def _make_trace_jsonl(tmp_path: Path, llm_calls: list[dict]) -> Path:
    """Write a minimal trace.jsonl with the given llm_call event payloads."""
    p = tmp_path / "trace.jsonl"
    with p.open("w") as f:
        f.write(json.dumps({"event": "run_start", "scenario": "s1"}) + "\n")
        for rec in llm_calls:
            base = {"event": "llm_call"}
            base.update(rec)
            f.write(json.dumps(base) + "\n")
        f.write(json.dumps({"event": "run_end"}) + "\n")
    return p


def test_cost_block_aggregates_tokens_from_trace_jsonl(tmp_path):
    """When metrics.json lacks cost fields, trace.jsonl is the source of truth.

    The fix introduces a trace_path parameter to build_aging_card; tokens
    and call counts must populate even with no session_results array.
    """
    metrics = dict(FIXTURE_METRICS)
    metrics.pop("session_results", None)  # force the trace path
    trace_path = _make_trace_jsonl(tmp_path, [
        {"gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 50},
        {"gen_ai.usage.input_tokens": 200, "gen_ai.usage.output_tokens": 80},
        {"gen_ai.usage.input_tokens": 150, "gen_ai.usage.output_tokens": 60},
    ])
    card = build_aging_card(metrics=metrics, trace_path=trace_path)
    cost = card["cost_and_efficiency"]
    assert cost["total_input_tokens"] == 450
    assert cost["total_output_tokens"] == 190
    assert cost["total_calls"] == 3
    assert cost["tokens_per_session_mean"] is not None and cost["tokens_per_session_mean"] > 0


def test_cost_block_latency_percentiles_from_duration_field(tmp_path):
    """When llm_call events carry duration_ms, p50/p95 latency populates."""
    trace_path = _make_trace_jsonl(tmp_path, [
        {"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 5,
         "gen_ai.usage.duration_ms": 100.0},
        {"gen_ai.usage.input_tokens": 20, "gen_ai.usage.output_tokens": 10,
         "gen_ai.usage.duration_ms": 200.0},
        {"gen_ai.usage.input_tokens": 30, "gen_ai.usage.output_tokens": 15,
         "gen_ai.usage.duration_ms": 1000.0},
    ])
    card = build_aging_card(metrics=FIXTURE_METRICS, trace_path=trace_path)
    cost = card["cost_and_efficiency"]
    assert cost["latency_ms_p50"] is not None
    assert cost["latency_ms_p95"] is not None
    # Tail percentile should be the slow call (1000 ms), within interpolation tolerance.
    assert cost["latency_ms_p95"] >= 900


def test_cost_block_cost_usd_from_per_call_field(tmp_path):
    """When llm_call events carry cost_usd, the card's total_cost_usd sums them."""
    trace_path = _make_trace_jsonl(tmp_path, [
        {"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 5,
         "gen_ai.usage.cost_usd": 0.001},
        {"gen_ai.usage.input_tokens": 20, "gen_ai.usage.output_tokens": 10,
         "gen_ai.usage.cost_usd": 0.0025},
    ])
    card = build_aging_card(metrics=FIXTURE_METRICS, trace_path=trace_path)
    cost = card["cost_and_efficiency"]
    assert cost["total_cost_usd"] == pytest.approx(0.0035, abs=1e-6)


def test_cost_block_handles_missing_trace_gracefully(tmp_path):
    """If trace_path is given but the file doesn't exist, card still builds."""
    nonexistent = tmp_path / "no_such.jsonl"
    card = build_aging_card(metrics=FIXTURE_METRICS, trace_path=nonexistent)
    # Should not crash and should produce a structurally valid cost block.
    cost = card["cost_and_efficiency"]
    assert "total_input_tokens" in cost
    assert "total_calls" in cost


def test_build_and_write_aging_card_auto_discovers_trace(tmp_path):
    """build_and_write_aging_card auto-finds trace.jsonl in the run dir."""
    # Set up a run directory with metrics.json and trace.jsonl
    metrics = dict(FIXTURE_METRICS)
    metrics.pop("session_results", None)
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    _make_trace_jsonl(tmp_path, [
        {"gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 50},
        {"gen_ai.usage.input_tokens": 200, "gen_ai.usage.output_tokens": 80},
    ])
    out_path = build_and_write_aging_card(tmp_path, sut_cfg=FIXTURE_SUT)
    assert out_path is not None
    with out_path.open() as f:
        card = json.load(f)
    cost = card["cost_and_efficiency"]
    assert cost["total_input_tokens"] == 300
    assert cost["total_output_tokens"] == 130
    assert cost["total_calls"] == 2
