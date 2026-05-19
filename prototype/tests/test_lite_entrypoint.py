"""Tests for the agingbench-lite console-script entry point."""
from __future__ import annotations

import pytest


from agingbench._lite_entrypoint import _validate_scope, _LITE_SCENARIOS, _LITE_SUITES


def test_lite_scenarios_set_includes_s1_s2_s7():
    """Lite must include the three v1 scenarios for the lite suite."""
    assert "s1_research_literature" in _LITE_SCENARIOS
    assert "s2_lifestyle_assistant" in _LITE_SCENARIOS
    assert "s7_research_notes" in _LITE_SCENARIOS


def test_lite_rejects_heavy_scenarios():
    """Scenarios outside the lite subset must be rejected."""
    rejected = ["s3_knowledge_base", "s4_software_engineering",
                "s5_self_planning", "s6_naturalistic", "s8_swe_bench"]
    for sid in rejected:
        msg = _validate_scope(["run", "--scenario", sid])
        assert msg is not None and "not in the lite subset" in msg


def test_lite_accepts_lite_subset():
    """Lite-subset invocations must pass scope validation."""
    for sid in _LITE_SCENARIOS:
        assert _validate_scope(["run", "--scenario", sid]) is None


def test_lite_accepts_lite_and_core_suites():
    """Only lite + core suites permitted at the lite entrypoint."""
    for sid in _LITE_SUITES:
        assert _validate_scope(["run", "--suite", sid]) is None


def test_lite_rejects_unknown_suite():
    """Unknown suites should be rejected with a clear message."""
    msg = _validate_scope(["run", "--suite", "api_models"])
    assert msg is not None
    assert "not in the lite subset" in msg


def test_lite_empty_args_returns_no_error():
    """Empty arg list (no --suite / --scenario flags) is fine; full CLI handles."""
    assert _validate_scope([]) is None
    assert _validate_scope(["run"]) is None
    assert _validate_scope(["list-suites"]) is None
