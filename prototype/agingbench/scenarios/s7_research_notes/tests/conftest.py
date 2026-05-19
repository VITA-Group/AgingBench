"""conftest.py — S7+ pytest options and session-marker filtering."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--scenario-session", type=int, default=999,
        help="current scenario session; tests marked with a later session are skipped",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "session(n): test runs only when scenario session >= n",
    )


def pytest_collection_modifyitems(config, items):
    cur = int(config.getoption("--scenario-session", default=999))
    for item in items:
        m = item.get_closest_marker("session")
        s = int(m.args[0]) if m else 0
        if s > cur:
            item.add_marker(pytest.mark.skip(reason=f"session {s} not reached (cur={cur})"))
