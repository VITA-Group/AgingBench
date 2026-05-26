"""Unit tests for Cursor Agent CLI adapters (credential-free)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agingbench.core.cursor_adapter import CursorAdapter
from agingbench.core.adapters.cursor_agent_adapter import CursorAgentAdapter
from agingbench.scenarios.s8_swe_bench.agent import build_s8_agent_from_sut


def _success_json(**overrides):
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 1234,
        "result": "done",
        "session_id": "abc-123",
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_cursor_adapter_parses_json_and_tracks_session(tmp_path):
    adapter = CursorAdapter(model="composer-2", cwd=tmp_path, cli_path="agent")
    with patch("subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout=_success_json(), stderr="")
        resp = adapter.send_message("hello")
    assert resp.text == "done"
    assert resp.session_id == "abc-123"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5
    assert adapter._session_id == "abc-123"


def test_cursor_adapter_resume_passes_session_id(tmp_path):
    adapter = CursorAdapter(model="composer-2", cwd=tmp_path, cli_path="agent")
    adapter._session_id = "abc-123"
    with patch("subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout=_success_json(), stderr="")
        adapter.send_message("follow up", resume=True)
        cmd = run.call_args[0][0]
    assert "--resume" in cmd
    assert "abc-123" in cmd


def test_cursor_adapter_reset_clears_session(tmp_path):
    adapter = CursorAdapter(model="composer-2", cwd=tmp_path, cli_path="agent")
    adapter._session_id = "abc-123"
    adapter.reset_session()
    assert adapter._session_id is None


def test_cursor_agent_adapter_reset_session(tmp_path):
    adapter = CursorAgentAdapter(model="composer-2", cwd=tmp_path, system_prompt=None)
    adapter.adapter._session_id = "abc-123"
    adapter.reset_session()
    assert adapter.adapter._session_id is None


def test_factory_picks_cursor_kind(tmp_path):
    sut = {"agent": {"adapter": "cursor", "model": "composer-2", "cli_path": "agent"}}
    runner = build_s8_agent_from_sut(sut, host_workspace_root=tmp_path)
    assert runner.adapter_kind == "cursor"
    assert runner.model == "composer-2"
    assert runner.cli_path == "agent"


def test_s8_build_adapter_cursor(tmp_path):
    sut = {"agent": {"adapter": "cursor", "model": "composer-2"}}
    runner = build_s8_agent_from_sut(sut, host_workspace_root=tmp_path)
    built = runner._build_adapter(tmp_path / "sess")
    assert isinstance(built, CursorAgentAdapter)
