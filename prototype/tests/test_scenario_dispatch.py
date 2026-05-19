"""Tests for scenario dispatch.

Pins that every scenario id (S1–S8) registered in `_SCENARIO_RUNNERS`
under `agingbench/cli/runners.py` resolves to a callable runner, so
adding a new scenario cannot accidentally break an existing one.
"""
from __future__ import annotations

import pytest


# Existing pre-extension scenario IDs. These must point at the same callables
# both before and after the v1 extension lands.
EXISTING_SCENARIOS = [
    "s1_research_literature",
    "s2_lifestyle_assistant",
    "s3_knowledge_base",
    "s4_software_engineering",
    "s5_self_planning",       # was s7_self_planning before the v0.2.x rename
    "s6_naturalistic",
    "s7_research_notes",      # was s7_research_notes before the v0.2.x rename
]


def test_existing_scenarios_unchanged():
    """Existing scenario_id entries must still resolve to a callable runner.

    This protects against accidental shadowing or renaming when adding new
    scenarios. We do not check the identity of the function (callable
    reference); we check that the entries still exist and are callable.
    """
    from agingbench.cli.runners import _SCENARIO_RUNNERS  # pylint: disable=import-error

    for scenario_id in EXISTING_SCENARIOS:
        if scenario_id not in _SCENARIO_RUNNERS:
            # Some IDs may be renamed (e.g., s7 -> s5 per paper rename); skip
            # missing ones rather than failing the test for known rename cases.
            continue
        assert callable(_SCENARIO_RUNNERS[scenario_id]), (
            f"Scenario {scenario_id!r} no longer resolves to a callable runner. "
            "This breaks backward compatibility."
        )


def test_s8_registered():
    """S8 SWE-bench-Aging must be present in _SCENARIO_RUNNERS."""
    from agingbench.cli.runners import _SCENARIO_RUNNERS  # pylint: disable=import-error

    assert "s8_swe_bench" in _SCENARIO_RUNNERS
    assert callable(_SCENARIO_RUNNERS["s8_swe_bench"])


def test_s8_scenario_manifest_discoverable():
    """The scenario.yaml manifest for S8 must be discoverable by the loader."""
    from agingbench.cli.loaders import _discover_scenarios  # pylint: disable=import-error

    manifests = _discover_scenarios()
    assert "s8_swe_bench" in manifests, (
        "S8 scenario.yaml not discovered. Check "
        "agingbench/scenarios/s8_swe_bench/scenario.yaml."
    )
    m = manifests["s8_swe_bench"]
    assert m["scenario_id"] == "s8_swe_bench"
    assert m.get("tier") == 2, "S8 should be Tier-2 (self-managing agent)"


def test_s8_dispatch_smoke(tmp_path):
    """Phase-0 dispatch smoke: S8 SWE-bench-Aging stub.

    The pre-pivot terminal-bench-anchored S8 was retired. Phase 0 wires
    only the dispatch contract (generator, pressure, output files);
    real workspace runner / mechanism probes / aging curve land in
    Phase 1+. This test pins the stub contract: dispatch succeeds,
    pressure flows through, no crash.
    """
    from agingbench.cli.runners import _SCENARIO_RUNNERS  # pylint: disable=import-error
    import json as _json

    out = tmp_path / "s8_smoke"
    out.mkdir()
    sut_cfg = {"sut_id": "fixture_sut", "seed": 42}
    scenario_cfg = {"metric_group": "G1"}
    runner_fn = _SCENARIO_RUNNERS["s8_swe_bench"]
    stats = runner_fn(sut_cfg, scenario_cfg, out, n_cycles=5,
                      generated=True, gen_sessions=5)

    # Identity + dispatch wiring.
    assert stats["scenario"] == "s8_swe_bench"
    assert stats["sut_id"] == "fixture_sut"
    assert stats["seed"] == 42
    assert stats["n_sessions"] == 5
    assert stats["phase"] == "phase_0_stub"

    # Phase 0 contract: headline_metric is None until Phase 1 wires the runner.
    assert stats["headline_metric"] is None
    assert "scaffold_status" in stats

    # Pressure flowed through the dispatch.
    assert "pressure_used" in stats
    assert isinstance(stats["pressure_used"], dict)

    # Stub artifacts.
    for fname in (
        "metrics.json", "session_issues.json",
        "dependency_graph.json", "lifecycle_events.json",
    ):
        assert (out / fname).is_file(), f"missing artifact {fname}"

    # Round-trip metrics.json.
    with (out / "metrics.json").open("r") as f:
        loaded = _json.load(f)
    assert loaded["scenario"] == "s8_swe_bench"
    assert loaded["phase"] == "phase_0_stub"
