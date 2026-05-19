"""Verify all 8 scenario runners actually plumb PressureConfig through to
their generators.

The non-interference guarantee is that *any* scenario with a `pressure:`
block in its SUT yaml or scenario yaml gets that config honored, not
silently dropped. This was already the case for S1-S6 + S8, but S7
(renamed from s7plus) was instantiating S7Generator without pressure.
This test pins every runner.
"""
from __future__ import annotations

import inspect

import pytest

from agingbench.cli import runners
from agingbench.cli.loaders import _resolve_pressure
from agingbench.generators.pressure_config import PressureConfig


def _function_source(fn):
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return ""


PRESSURE_AWARE_RUNNERS = [
    "_run_s1", "_run_s2", "_run_s3", "_run_s4",
    "_run_s5", "_run_s6", "_run_s7", "_run_s8",
]


@pytest.mark.parametrize("runner_name", PRESSURE_AWARE_RUNNERS)
def test_runner_calls_resolve_pressure(runner_name):
    """Every standard scenario runner must call _resolve_pressure."""
    fn = getattr(runners, runner_name, None)
    assert fn is not None, f"runner {runner_name!r} not exported from cli.runners"
    src = _function_source(fn)
    assert "_resolve_pressure(sut_cfg" in src, (
        f"{runner_name} must call _resolve_pressure(sut_cfg, scenario_cfg) "
        "so SUT/scenario YAML pressure overrides are honored."
    )


def test_default_fallback_unchanged_after_runner_fixes():
    """The non-interference gate: empty inputs still produce medium()."""
    assert _resolve_pressure({}, {}).to_dict() == PressureConfig.medium().to_dict()


def test_s7_pressure_yaml_override_propagates():
    """Scenario-cfg pressure block must reach _resolve_pressure for S7."""
    scen = {"pressure": {"preset": "light"}}
    p = _resolve_pressure({}, scen)
    assert p.tokens_per_session == 500
    assert p.dependency_density == 0.3


def test_s8_generator_phase0_stub_contract():
    """S8 SWE-bench-Aging is in Phase-0 stub state.

    The pre-pivot terminal-bench DAG-induction tests don't apply to
    Phase 0 (no chain selected, no probes synthesised). This test
    pins the stub's contract: pressure flows through, generator returns
    the expected empty-stream shape, no crashes. Phase 1 re-introduces
    DAG-density and confusable-pair tests once the real chain is wired.
    """
    from agingbench.generators.s8_swe_bench_generator import S8SweBenchGenerator

    for preset in ("light", "medium", "heavy"):
        p = getattr(PressureConfig, preset)()
        out = S8SweBenchGenerator(seed=42, pressure=p).generate(5)
        assert out["phase"] == "phase_0_stub"
        assert out["session_issues"] == []
        assert out["lifecycle_events"] == []
        assert out["dependency_graph"]["dependency_edges"] == []
        assert out["dependency_graph"]["interference_pairs"] == []
        # Pressure dial values flow through the output for audit.
        assert out["pressure_used"]["dependency_density"] == p.dependency_density
        assert out["pressure_used"]["n_confusable_pairs"] == p.n_confusable_pairs
