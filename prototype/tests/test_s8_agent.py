"""Phase 3 — S8 agent layer tests.

All tests here are credential-free: they exercise the prompt construction,
the artifact-collection logic, and the adapter-selection factory. The
end-to-end path that actually invokes Claude Code or OpenHands is
exercised separately under S8_LIVE_AGENT_SMOKE=1 (different test module).
"""
from __future__ import annotations

import pytest

from agingbench.scenarios.s8_swe_bench.agent import (
    S8AgentRequest,
    S8AgentResponse,
    S8AgentRunner,
    _build_user_prompt,
    _extract_block,
    build_s8_agent_from_sut,
)


# ---- helper extraction ---------------------------------------------------

def test_extract_block_basic():
    text = "<NOTES>\nfoo\n</NOTES>\n<DIFF>\nbar\n</DIFF>"
    assert _extract_block(text, "NOTES") == "foo"
    assert _extract_block(text, "DIFF") == "bar"


def test_extract_block_missing_returns_none():
    assert _extract_block("no tags here", "NOTES") is None


def test_extract_block_unclosed_returns_remainder():
    assert _extract_block("<NOTES>\npartial", "NOTES") == "partial"


# ---- prompt construction -------------------------------------------------

def test_build_user_prompt_includes_session_and_issue(tmp_path):
    req = S8AgentRequest(
        session_idx=3,
        instance_id="sphinx-doc__sphinx-7454",
        issue_text="Inconsistent handling of None by autodoc_typehints.",
        chain_role="foundation",
        chain_summary="Establishes baseline behaviour for autodoc_typehints",
        prior_notes="prior session note about typehints",
        host_workspace_dir=tmp_path / "sess",
        persistent_notes_path=tmp_path / "memory" / ".aging" / "notes.md",
    )
    prompt = _build_user_prompt(req)
    assert "Session 3" in prompt
    assert "sphinx-doc__sphinx-7454" in prompt
    assert "autodoc_typehints" in prompt
    assert "foundation" in prompt
    assert "prior session note about typehints" in prompt


def test_build_user_prompt_handles_empty_prior_notes(tmp_path):
    req = S8AgentRequest(
        session_idx=0,
        instance_id="sphinx-doc__sphinx-7454",
        issue_text="task",
        chain_role="foundation",
        chain_summary="",
        prior_notes="",
        host_workspace_dir=tmp_path,
        persistent_notes_path=tmp_path / ".aging" / "notes.md",
    )
    prompt = _build_user_prompt(req)
    assert "first session" in prompt


# ---- factory selection ---------------------------------------------------

def test_factory_picks_claude_code_kind(tmp_path):
    sut = {"agent": {"adapter": "claude_code", "model": "claude-haiku-4-5-20251001"}}
    runner = build_s8_agent_from_sut(sut, host_workspace_root=tmp_path)
    assert runner.adapter_kind == "claude_code"
    assert runner.model == "claude-haiku-4-5-20251001"


def test_factory_picks_openhands_kind(tmp_path):
    sut = {"agent": {"adapter": "openhands", "model": "gpt-4o-mini",
                     "max_turns": 10, "api_key_env": "OPENAI_API_KEY"}}
    runner = build_s8_agent_from_sut(sut, host_workspace_root=tmp_path)
    assert runner.adapter_kind == "openhands"
    assert runner.model == "gpt-4o-mini"
    assert runner.max_turns == 10
    assert runner.api_key_env == "OPENAI_API_KEY"


def test_factory_falls_back_to_litellm(tmp_path):
    sut = {}
    runner = build_s8_agent_from_sut(sut, host_workspace_root=tmp_path)
    assert runner.adapter_kind == "litellm"


def test_unknown_adapter_kind_raises_at_session_time(tmp_path):
    runner = S8AgentRunner(
        adapter_kind="not_a_real_adapter",
        model="x", host_workspace_root=tmp_path,
    )
    with pytest.raises(ValueError, match="Unsupported S8 agent adapter_kind"):
        runner._build_adapter(tmp_path)


# ---- artifact collection (no LLM) -----------------------------------------

def test_litellm_extract_writes_notes_and_diff_to_cwd(tmp_path):
    """Verify the LiteLLM bridge correctly parses + writes artifacts.

    We monkey-patch litellm.completion so no API call happens.
    """
    from agingbench.scenarios.s8_swe_bench.agent import _LiteLLMAgentBridge
    import litellm
    from types import SimpleNamespace

    fixed_response = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=(
            "<NOTES>\n## session 0\nlearned about autodoc_typehints\n</NOTES>\n"
            "<DIFF>\n--- a/foo.py\n+++ b/foo.py\n@@\n-old\n+new\n</DIFF>\n"
        )))
    ])
    real_completion = litellm.completion
    litellm.completion = lambda **kw: fixed_response
    try:
        bridge = _LiteLLMAgentBridge(model="claude-haiku-4-5-20251001", cwd=tmp_path)
        bridge.send_message("dummy prompt")
    finally:
        litellm.completion = real_completion

    notes = (tmp_path / ".aging" / "notes.md").read_text()
    diff = (tmp_path / "solution.diff").read_text()
    assert "autodoc_typehints" in notes
    assert "--- a/foo.py" in diff
