"""Tests for CLI flag defaults around AgingCard emission.

The `--card` flag opts in to writing `aging_card.json`. Default is
OFF so that existing CLI scripts (`agingbench run ...` without
`--card`) continue to produce only `metrics.json` and
`dependency_metrics.json`, with no `aging_card.json` side-effect.
"""
from __future__ import annotations

import pytest


def _find_card_action():
    """Locate the --card argparse Action in the run-subcommand parser."""
    from agingbench.cli import _build_parser  # pylint: disable=import-error
    parser = _build_parser()
    # The run subcommand parser owns --card. Walk argparse subparsers to find it.
    for action in parser._actions:  # pylint: disable=protected-access
        if hasattr(action, "choices") and isinstance(action.choices, dict):
            for sub_name, sub_parser in action.choices.items():
                if sub_name != "run":
                    continue
                for sub_action in sub_parser._actions:  # pylint: disable=protected-access
                    if "--card" in sub_action.option_strings:
                        return sub_action
    return None


def test_card_flag_present_on_run_subcommand():
    """The --card flag must exist on the `run` subcommand parser."""
    action = _find_card_action()
    assert action is not None, "CLI 'run' subcommand is missing --card flag"


def test_card_flag_default_off():
    """The --card flag must default to False in v1.

    Per the plan: v1 ships with --card opt-in. Promotion to opt-in (default
    on) is deferred to v1.1 after one release cycle, when downstream CI
    pipelines have had time to expect aging_card.json.
    """
    action = _find_card_action()
    assert action is not None
    assert action.default is False, (
        "--card defaults to True; v1 requires --card default=False so existing "
        "CI scripts that don't expect aging_card.json are unaffected."
    )


def test_card_flag_is_store_true():
    """The flag should be a store_true action (no value required)."""
    action = _find_card_action()
    assert action is not None
    assert action.const is True or getattr(action, "_store_true", False) or repr(action).find("_StoreTrueAction") != -1


def test_cmd_run_signature_accepts_emit_card():
    """cmd_run() must accept the `emit_card` kwarg that wires through `--card`."""
    from inspect import signature
    from agingbench.cli import cmd_run  # pylint: disable=import-error

    sig = signature(cmd_run)
    assert "emit_card" in sig.parameters, (
        "cmd_run() missing emit_card parameter; --card flag wiring incomplete."
    )
    assert sig.parameters["emit_card"].default is False, (
        "emit_card kwarg must default to False for backward compatibility."
    )
