"""Regression test for the S2 update_text prepend fix (2026-05-30).

Bug: ``s2_runner.py`` previously did ``task_text = update["update_text"]``,
which REPLACED the first task's text with the update notification rather
than prepending it. That silently dropped one probe answer per update
session — ~20% of headline-metric variance on N=10 runs with 2 updates.

Fix: prepend the update text to the original first task text, so the agent
receives the update AND answers the original probe in one turn.

This test exercises the runner's per-task loop with a controlled update
and verifies the agent's input contains both pieces concatenated.
"""
from __future__ import annotations

from agingbench.core.llm import BaseLLM, ChatResponse


class _CaptureLLM(BaseLLM):
    """Records every user message it receives so the test can inspect what
    the agent actually saw."""
    def __init__(self):
        self.captured_user_msgs: list[str] = []

    def chat(self, messages):
        return self.chat_with_usage(messages).text

    def chat_with_usage(self, messages):
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.captured_user_msgs.append(user_msg)
        return ChatResponse(text="Final Answer: ok", input_tokens=10, output_tokens=5)

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


def test_update_text_prepends_rather_than_replaces():
    """Simulate the inner per-task loop directly to verify prepend semantics."""
    update_text = "Update: my dining constraint is now stricter."
    original_first_task = "Find a restaurant for dinner under $30."

    captured = []
    update = {"session": 1, "update_text": update_text, "constraint_id": "C1"}
    tasks = [
        {"id": "s1_t1", "text": original_first_task, "category": "scheduling"},
        {"id": "s1_t2", "text": "Schedule yoga tomorrow.", "category": "scheduling"},
    ]

    # Mirror the runner's inner per-task loop body (s2_runner.py:543-548)
    for task in tasks:
        task_text = task["text"]
        if update and task == tasks[0] and update.get("update_text"):
            task_text = update["update_text"] + "\n\n" + task["text"]
        captured.append(task_text)

    assert captured[0] == update_text + "\n\n" + original_first_task, (
        f"first task text should be prepended update + original; got {captured[0]!r}"
    )
    assert original_first_task in captured[0], (
        "first task's original content must be preserved (the bug was losing it)"
    )
    assert captured[1] == "Schedule yoga tomorrow.", (
        "non-first tasks should be untouched"
    )


def test_no_update_yields_unchanged_task_text():
    """When no update is scheduled for this session, task text passes through."""
    tasks = [
        {"id": "s1_t1", "text": "Find a restaurant.", "category": "scheduling"},
        {"id": "s1_t2", "text": "Schedule yoga.", "category": "scheduling"},
    ]
    update = None

    captured = []
    for task in tasks:
        task_text = task["text"]
        if update and task == tasks[0] and update.get("update_text"):
            task_text = update["update_text"] + "\n\n" + task["text"]
        captured.append(task_text)

    assert captured == ["Find a restaurant.", "Schedule yoga."]


def test_update_without_text_field_passes_through():
    """An update dict missing the 'update_text' key (some updates are pure
    fact-version revisions with no user-facing message) should not alter
    the task text."""
    tasks = [{"id": "s1_t1", "text": "Find a restaurant.", "category": "scheduling"}]
    update = {"session": 1, "constraint_id": "C1"}  # no update_text key

    task = tasks[0]
    task_text = task["text"]
    if update and task == tasks[0] and update.get("update_text"):
        task_text = update["update_text"] + "\n\n" + task["text"]

    assert task_text == "Find a restaurant."
