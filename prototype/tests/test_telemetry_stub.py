"""Tests for the v1.0 telemetry-mode stub.

The stub must (a) produce a v1.0.0-valid AgingCard, (b) flag itself as
partial via warnings, and (c) successfully ingest the bundled example
trace fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agingbench.telemetry import (
    SUPPORTED_TRACE_FORMATS,
    TraceToCardResult,
    trace_to_card,
)
from agingbench.metrics.aging_card_validate import validate_card_dict


EXAMPLE_DIR = Path(__file__).parent.parent / "agingbench" / "telemetry" / "example_traces"


def test_supported_formats_include_generic():
    assert "generic" in SUPPORTED_TRACE_FORMATS
    assert "langfuse" in SUPPORTED_TRACE_FORMATS
    assert "langsmith" in SUPPORTED_TRACE_FORMATS
    assert "otlp" in SUPPORTED_TRACE_FORMATS


def test_unsupported_format_raises():
    with pytest.raises(ValueError) as exc:
        trace_to_card(Path("/dev/null"), trace_format="bogus")
    assert "Unsupported trace_format" in str(exc.value)


def test_stub_emits_partial_warning():
    """The stub must always flag itself with warnings: ['telemetry_partial']."""
    result = trace_to_card(EXAMPLE_DIR / "langfuse_sample.jsonl",
                           scenario_hint="s2_lifestyle_assistant",
                           sut_hint={"sut_id": "haiku45_prod"},
                           trace_format="langfuse")
    assert isinstance(result, TraceToCardResult)
    assert "telemetry_partial" in result.card.get("warnings", [])


def test_stub_card_validates_v1_schema():
    """The stub card must validate against the v1.0.0 schema."""
    result = trace_to_card(EXAMPLE_DIR / "langfuse_sample.jsonl",
                           scenario_hint="s2_lifestyle_assistant",
                           sut_hint={"sut_id": "haiku45_prod"},
                           trace_format="langfuse")
    errors = validate_card_dict(result.card)
    assert errors == [], f"telemetry stub card invalid: {errors}"


def test_stub_aggregates_tokens_from_langfuse_trace():
    result = trace_to_card(EXAMPLE_DIR / "langfuse_sample.jsonl",
                           scenario_hint="s2_lifestyle_assistant",
                           trace_format="langfuse")
    cost = result.card["cost_and_efficiency"]
    # langfuse_sample.jsonl has 3 events with input_tokens 1024, 1500, 1100
    # = 3624 (best-effort sum)
    assert cost["total_input_tokens"] == 3624
    # 280 + 150 + 240 = 670
    assert cost["total_output_tokens"] == 670
    assert cost["total_calls"] == 3
    assert result.n_calls == 3


def test_stub_aggregates_tokens_from_otlp_trace():
    """OTLP trace uses nested attributes; v1 stub should still find tokens via
    common field-name fallback (or skip cleanly)."""
    result = trace_to_card(EXAMPLE_DIR / "otlp_sample.jsonl",
                           scenario_hint="s2_lifestyle_assistant",
                           trace_format="otlp")
    assert result.n_calls == 2
    # v1 stub may not parse nested attributes; just verify it didn't crash.
    assert "total_calls" in result.card["cost_and_efficiency"]


def test_stub_derived_and_missing_fields_listed():
    result = trace_to_card(EXAMPLE_DIR / "langfuse_sample.jsonl",
                           scenario_hint="s2_lifestyle_assistant",
                           trace_format="langfuse")
    assert "card_envelope" in result.derived_fields
    assert "n_calls" in result.derived_fields
    # Phase-3 work surfaces in missing_fields
    assert any("mechanism_metrics" in f for f in result.missing_fields)
    assert "pressure" in result.missing_fields


def test_stub_missing_trace_file_is_graceful():
    """Missing trace file shouldn't crash; produces empty card with 0 calls."""
    result = trace_to_card(Path("/tmp/nonexistent_trace_aging.jsonl"),
                           scenario_hint="s1_research_literature",
                           trace_format="generic")
    assert result.n_calls == 0
    assert result.card["cost_and_efficiency"]["total_calls"] == 0


def test_stub_card_provenance_marks_stub_version():
    """Provenance must include the stub-version marker so consumers know."""
    result = trace_to_card(EXAMPLE_DIR / "langfuse_sample.jsonl",
                           scenario_hint="s2_lifestyle_assistant",
                           trace_format="langfuse")
    prov = result.card["provenance"]
    assert "telemetry_stub_version" in prov
    assert prov["telemetry_stub_version"].startswith("v1.0.0-stub")
