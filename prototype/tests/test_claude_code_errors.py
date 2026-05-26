"""Tests for Claude Code adapter fatal-error handling."""

import json
import subprocess

import pytest

from agingbench.core.claude_code_adapter import (
    ClaudeCodeError,
    ClaudeCodeAdapter,
    _raise_if_fatal_cli_error,
)


def test_raise_on_nonzero_exit_with_json_error():
    stdout = json.dumps({
        "type": "result",
        "is_error": True,
        "api_error_status": 429,
        "result": "Rate limit exceeded. Try again later.",
    })
    with pytest.raises(ClaudeCodeError) as exc:
        _raise_if_fatal_cli_error(exit_code=1, stdout=stdout, stderr="")
    assert exc.value.api_status == 429
    assert "rate limit" in str(exc.value).lower()


def test_raise_on_is_error_in_parsed_output():
    adapter = ClaudeCodeAdapter(model="claude-sonnet-4-6", cwd=".")
    stdout = json.dumps({
        "type": "result",
        "is_error": True,
        "api_error_status": 402,
        "result": "You have exceeded your usage limit.",
    })
    with pytest.raises(ClaudeCodeError):
        adapter._parse_json_output(stdout)


def test_send_subprocess_propagates_usage_limit(monkeypatch, tmp_path):
    adapter = ClaudeCodeAdapter(model="claude-opus-4-7", cwd=tmp_path)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout=json.dumps({
                "is_error": True,
                "api_error_status": 429,
                "result": "usage limit reached",
            }),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ClaudeCodeError):
        adapter.send_message("hello")
