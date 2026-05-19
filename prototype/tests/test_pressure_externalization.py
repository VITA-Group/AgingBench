"""Tests for PressureConfig externalization.

PressureConfig is resolved from (in order): SUT YAML, scenario YAML,
suite manifest, then a built-in `medium` default. These tests pin
that the default-fallback path returns exactly the prior built-in
`PressureConfig.medium()` so no existing scenario drifts when the
upstream YAML omits a `pressure:` block.
"""
from __future__ import annotations

import pytest

from agingbench.generators.pressure_config import PressureConfig


def test_pressureconfig_medium_is_stable():
    """Sanity: PressureConfig.medium() returns a stable, deterministic config.

    This is a precondition for the fallback-equals-medium test: if medium()
    is not deterministic, equality checks below are meaningless.
    """
    a = PressureConfig.medium()
    b = PressureConfig.medium()
    assert a.to_dict() == b.to_dict(), "PressureConfig.medium() is non-deterministic"


def test_default_fallback_equals_medium():
    """When no `pressure:` key is present at any layer, _resolve_pressure must
    return the equivalent of PressureConfig.medium().

    This is THE non-interference gate for the 2c refactor. If this fails, the
    extension has changed the behavior of existing scenarios that relied on
    the hard-coded PressureConfig.medium() default.
    """
    from agingbench.cli.loaders import _resolve_pressure  # pylint: disable=import-error

    resolved = _resolve_pressure(sut_cfg={}, scenario_cfg={}, manifest={})
    expected = PressureConfig.medium()
    assert resolved.to_dict() == expected.to_dict(), (
        "Default _resolve_pressure path drifted from PressureConfig.medium(). "
        "This breaks backward compatibility with all existing scenarios."
    )


def test_default_fallback_with_none_layers():
    """All-None layers should also fall back to medium (defensive)."""
    from agingbench.cli.loaders import _resolve_pressure  # pylint: disable=import-error

    resolved = _resolve_pressure(sut_cfg=None, scenario_cfg=None, manifest=None)
    expected = PressureConfig.medium()
    assert resolved.to_dict() == expected.to_dict()


def test_pressure_light_override_via_scenario_cfg():
    """`pressure: light` in scenario_cfg should produce the light preset."""
    from agingbench.cli.loaders import _resolve_pressure  # pylint: disable=import-error

    resolved = _resolve_pressure(
        sut_cfg={}, scenario_cfg={"pressure": "light"}, manifest={}
    )
    assert resolved.to_dict() == PressureConfig.light().to_dict()


def test_pressure_preset_dict_form():
    """`pressure: {preset: heavy}` should produce the heavy preset."""
    from agingbench.cli.loaders import _resolve_pressure  # pylint: disable=import-error

    resolved = _resolve_pressure(
        sut_cfg={}, scenario_cfg={"pressure": {"preset": "heavy"}}, manifest={}
    )
    assert resolved.to_dict() == PressureConfig.heavy().to_dict()


def test_pressure_override_fields():
    """`pressure: {preset: light, n_confusable_pairs: 7}` should apply per-field override on light."""
    from agingbench.cli.loaders import _resolve_pressure  # pylint: disable=import-error

    resolved = _resolve_pressure(
        sut_cfg={},
        scenario_cfg={"pressure": {"preset": "light", "n_confusable_pairs": 7}},
        manifest={},
    )
    expected = PressureConfig.light().to_dict()
    expected["n_confusable_pairs"] = 7
    assert resolved.to_dict() == expected


def test_sut_cfg_takes_precedence_over_scenario_cfg():
    """SUT YAML override beats scenario YAML default."""
    from agingbench.cli.loaders import _resolve_pressure  # pylint: disable=import-error

    resolved = _resolve_pressure(
        sut_cfg={"pressure": "heavy"},
        scenario_cfg={"pressure": "light"},
        manifest={"pressure": "none"},
    )
    assert resolved.to_dict() == PressureConfig.heavy().to_dict()
