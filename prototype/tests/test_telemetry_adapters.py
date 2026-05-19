"""Tests for the per-format trace adapters: edge cases on the generic
adapter, interop with Langfuse / OTLP fixtures, and the two new
agent-platform adapters (OpenAI Assistants, OpenHands)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agingbench.telemetry import list_supported_formats, trace_to_card_v11
from agingbench.telemetry.adapters import (
    generic, langfuse_v1, otlp_v1, openai_assistants, openhands,
)


_FIXTURE_DIR = (Path(__file__).resolve().parent.parent
                / "agingbench" / "telemetry" / "example_traces")


# ---------------------------------------------------------------- registry

def test_registry_lists_all_seven_formats():
    fmts = set(list_supported_formats())
    assert {"claude_code", "generic", "langfuse", "langsmith", "otlp",
            "openai_assistants", "openhands"} <= fmts


# ---------------------------------------------------------------- generic adapter regressions

def test_generic_routes_content_to_prompt_for_user_role():
    rec = generic.normalize({
        "session_id": "s1", "role": "user",
        "content": "Hello world", "timestamp": "2026-05-14T10:00:00Z",
    })
    assert rec is not None
    assert rec.role == "user"
    assert rec.prompt_preview == "Hello world"
    assert rec.response_preview is None


def test_generic_routes_content_to_response_for_assistant_role():
    rec = generic.normalize({
        "session_id": "s1", "role": "assistant",
        "content": "Hi back", "timestamp": "2026-05-14T10:00:00Z",
    })
    assert rec is not None
    assert rec.role == "agent"   # assistant → agent
    assert rec.response_preview == "Hi back"
    assert rec.prompt_preview is None


def test_generic_accepts_tool_only_event():
    rec = generic.normalize({
        "session_id": "s1",
        "tool_calls": [{"name": "bash", "arguments": {"cmd": "ls"}}],
        "timestamp": "2026-05-14T10:00:00Z",
    })
    assert rec is not None
    assert len(rec.tool_calls) == 1
    assert rec.tool_calls[0].name == "bash"


def test_generic_coerces_numeric_session_id_to_string():
    rec = generic.normalize({
        "session_id": 12345, "prompt": "x",
        "timestamp": "2026-05-14T10:00:00Z",
    })
    assert rec is not None
    assert rec.session_id == "12345"
    assert isinstance(rec.session_id, str)


def test_generic_rejects_metadata_only_event():
    rec = generic.normalize({
        "session_id": "s1", "timestamp": "2026-05-14T10:00:00Z", "foo": "bar",
    })
    assert rec is None


# ---------------------------------------------------------------- langfuse fixes

def test_langfuse_handles_snake_case_and_falls_back_to_trace_id():
    """Real-world Langfuse REST exports use snake_case; sessionId is often
    absent and the trace_id is the natural session grouping."""
    rec = langfuse_v1.normalize({
        "id": "span_001", "name": "answer",
        "trace_id": "t_2026_05_11_a",
        "usage": {"input_tokens": 1500, "output_tokens": 150},
        "model": "claude-haiku-4-5",
        "timestamp": "2026-05-11T10:32:00Z",
    })
    assert rec is not None
    assert rec.session_id == "t_2026_05_11_a"
    assert rec.input_tokens == 1500
    assert rec.output_tokens == 150


def test_langfuse_sample_fixture_normalises_with_session_grouping():
    rows = [json.loads(l) for l in
            (_FIXTURE_DIR / "langfuse_sample.jsonl").read_text().splitlines()
            if l.strip()]
    out = [r for r in (langfuse_v1.normalize(ev) for ev in rows) if r is not None]
    assert len(out) == 3
    assert {r.session_id for r in out} == {"t_2026_05_11_a", "t_2026_05_11_b"}


# ---------------------------------------------------------------- otlp legacy namespace

def test_otlp_recognises_legacy_llm_namespace():
    rec = otlp_v1.normalize({
        "span_id": "s001", "trace_id": "t001",
        "operation_name": "llm.compaction",
        "attributes": {
            "llm.input_tokens": 800,
            "llm.output_tokens": 200,
            "llm.model": "gpt-4o-mini",
        },
        "start_time": "2026-05-11T10:30:00Z",
    })
    assert rec is not None
    assert rec.input_tokens == 800
    assert rec.output_tokens == 200
    assert rec.model_id == "gpt-4o-mini"
    assert rec.session_id == "t001"   # falls back to trace_id


def test_otlp_recognises_modern_gen_ai_namespace():
    rec = otlp_v1.normalize({
        "spanId": "s002", "traceId": "t002",
        "attributes": {
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-haiku-4-5",
            "gen_ai.usage.input_tokens": 1024,
            "gen_ai.usage.output_tokens": 256,
        },
        "startTime": "2026-05-11T10:30:00Z",
    })
    assert rec is not None
    assert rec.model_id == "claude-haiku-4-5"
    assert rec.input_tokens == 1024


def test_otlp_rejects_non_llm_span():
    rec = otlp_v1.normalize({
        "spanId": "s003", "operation_name": "http.request",
        "attributes": {"http.method": "GET", "http.status_code": 200},
    })
    assert rec is None


# ---------------------------------------------------------------- openai_assistants

def test_assistants_thread_message_user():
    rec = openai_assistants.normalize({
        "id": "msg_x", "object": "thread.message",
        "thread_id": "thread_a", "role": "user",
        "content": [{"type": "text", "text": {"value": "Hello"}}],
        "created_at": 1715000000,
    })
    assert rec is not None
    assert rec.session_id == "thread_a"
    assert rec.role == "user"
    assert rec.prompt_preview == "Hello"


def test_assistants_thread_run_carries_token_usage():
    rec = openai_assistants.normalize({
        "id": "run_x", "object": "thread.run",
        "thread_id": "thread_a", "model": "gpt-4o-mini",
        "started_at": 1715000001, "completed_at": 1715000005,
        "usage": {"prompt_tokens": 1500, "completion_tokens": 220},
    })
    assert rec is not None
    assert rec.input_tokens == 1500
    assert rec.output_tokens == 220
    assert rec.model_id == "gpt-4o-mini"
    assert rec.duration_ms == 4000.0


def test_assistants_run_step_extracts_function_tool_call():
    rec = openai_assistants.normalize({
        "id": "step_x", "object": "thread.run.step",
        "thread_id": "thread_a", "run_id": "run_x",
        "type": "tool_calls",
        "step_details": {"tool_calls": [
            {"type": "function",
             "function": {"name": "fetch_sales",
                          "arguments": "{\"quarter\": \"Q3\"}"}},
        ]},
    })
    assert rec is not None
    assert len(rec.tool_calls) == 1
    assert rec.tool_calls[0].name == "fetch_sales"
    assert rec.tool_calls[0].args == {"quarter": "Q3"}


def test_assistants_run_step_extracts_code_interpreter():
    rec = openai_assistants.normalize({
        "id": "step_y", "object": "thread.run.step",
        "thread_id": "thread_a", "run_id": "run_x",
        "type": "tool_calls",
        "step_details": {"tool_calls": [
            {"type": "code_interpreter",
             "code_interpreter": {"input": "df.sum()", "outputs": []}},
        ]},
    })
    assert rec is not None
    assert rec.tool_calls[0].name == "code_interpreter"
    assert "df.sum()" in rec.tool_calls[0].args["input"]


def test_assistants_skips_thread_metadata_object():
    assert openai_assistants.normalize({
        "id": "thread_a", "object": "thread", "created_at": 1715000000,
    }) is None


def test_assistants_e2e_sample_two_threads():
    result = trace_to_card_v11(
        trace_jsonl=str(_FIXTURE_DIR / "openai_assistants_sample.jsonl"),
        trace_format="openai_assistants",
        profile="generic",
    )
    assert result.n_records == 8
    assert result.n_sessions == 2
    cost = result.card.get("cost_and_efficiency") or {}
    assert cost.get("total_input_tokens") == 3200
    assert cost.get("total_output_tokens") == 400


# ---------------------------------------------------------------- openhands

def test_openhands_user_message():
    rec = openhands.normalize({
        "id": 0, "conversation_id": "conv_a",
        "timestamp": "2026-05-14T10:00:00Z",
        "source": "user", "message": "Refactor the auth module",
    })
    assert rec is not None
    assert rec.session_id == "conv_a"
    assert rec.role == "user"
    assert "Refactor" in rec.prompt_preview


def test_openhands_agent_action_becomes_tool_call():
    rec = openhands.normalize({
        "id": 1, "conversation_id": "conv_a",
        "timestamp": "2026-05-14T10:00:02Z",
        "source": "agent", "action": "edit",
        "args": {"path": "src/auth.py", "command": "str_replace"},
    })
    assert rec is not None
    assert rec.role == "agent"
    assert len(rec.tool_calls) == 1
    assert rec.tool_calls[0].name == "edit"
    assert rec.tool_calls[0].args["path"] == "src/auth.py"


def test_openhands_environment_observation_becomes_tool_role():
    rec = openhands.normalize({
        "id": 2, "conversation_id": "conv_a",
        "source": "environment", "observation": "edit",
        "message": "Successfully edited src/auth.py",
        "timestamp": "2026-05-14T10:00:03Z",
    })
    assert rec is not None
    assert rec.role == "tool"
    assert "Successfully" in rec.response_preview


def test_openhands_llm_metrics_carries_tokens_and_model():
    rec = openhands.normalize({
        "id": 6, "conversation_id": "conv_a",
        "source": "agent",
        "llm_metrics": {"prompt_tokens": 1820, "completion_tokens": 245,
                        "model": "claude-sonnet-4-5", "cost": 0.018},
        "timestamp": "2026-05-14T10:00:09Z",
    })
    assert rec is not None
    assert rec.input_tokens == 1820
    assert rec.output_tokens == 245
    assert rec.model_id == "claude-sonnet-4-5"
    assert rec.cost_usd == pytest.approx(0.018)


def test_openhands_action_message_becomes_agent_response():
    rec = openhands.normalize({
        "id": 5, "conversation_id": "conv_a",
        "source": "agent", "action": "message",
        "message": "Refactored auth.py",
        "timestamp": "2026-05-14T10:00:08Z",
    })
    assert rec is not None
    assert rec.role == "agent"
    assert rec.response_preview == "Refactored auth.py"


def test_openhands_e2e_sample_two_conversations():
    result = trace_to_card_v11(
        trace_jsonl=str(_FIXTURE_DIR / "openhands_sample.jsonl"),
        trace_format="openhands",
        profile="generic",
    )
    assert result.n_records == 13
    assert result.n_sessions == 2
    cost = result.card.get("cost_and_efficiency") or {}
    assert cost.get("total_input_tokens") == 3920
    assert cost.get("total_output_tokens") == 555
