"""
agingbench/core/adapters/openhands_bridge.py — Subprocess bridge for OpenHands.

Runs inside the isolated `openhands` conda env (NOT agingbench env). Invoked by
OpenHandsAdapter via subprocess with a JSON request on stdin; writes a JSON
response to stdout.

Protocol (one-shot per invocation):
  stdin:  {
    "task": str,                  # user message
    "workspace_dir": str,          # persistent agent workspace
    "persistence_dir": str,        # conversation state directory
    "model": str,                  # litellm model string (e.g. "gpt-4o-mini")
    "api_key": str,                # provider API key
    "conversation_id": str|null,   # resume existing conversation if set
    "system_prompt": str|null,     # optional system prompt prefix
    "max_iterations": int,         # upper bound on agent turns per run
    "condenser": str|null,         # "default" (preset) | "none" | "llm_summary"
    "condenser_max_size": int|null,   # llm_summary trigger size (events); default 80
    "condenser_keep_first": int|null, # llm_summary head events kept verbatim; default 4
  }
  stdout: {
    "text": str,                   # final assistant text
    "input_tokens": int,
    "output_tokens": int,
    "cost_usd": float,
    "conversation_id": str,        # UUID; caller stores for resumption
    "files_changed": [str],        # paths modified in workspace_dir (relative)
    "iterations": int,             # agent turns consumed
    "condenser": {...},            # active condenser: kind/class/max_size/keep_first
    "condensations": int,          # how many times the condenser fired this run
    "error": str|null,
  }

Memory-compression control (Level A): the OpenHands agent presets bake in an
LLMSummarizingCondenser(max_size=80, keep_first=4). The `condenser` request
fields let the benchmark control that in-context compaction as a designed
factor (off / summarize-at-budget) instead of inheriting it silently, and the
`condensations` count reports whether it actually fired so an inert run is not
mistaken for a controlled one.

This bridge intentionally does one turn per process. Resumption is via
`persistence_dir` + `conversation_id` (OpenHands' built-in state persistence).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")


def _snapshot_workspace(ws: Path) -> dict[str, float]:
    """Return {relpath: mtime} for all files in workspace."""
    out: dict[str, float] = {}
    if not ws.exists():
        return out
    for f in ws.rglob("*"):
        if f.is_file():
            try:
                out[str(f.relative_to(ws))] = f.stat().st_mtime
            except (OSError, ValueError):
                continue
    return out


def _diff_workspace(before: dict[str, float], after: dict[str, float]) -> list[str]:
    changed: list[str] = []
    for p, mt in after.items():
        if p not in before or before[p] != mt:
            changed.append(p)
    return sorted(changed)


def _is_reasoning_model(model: str) -> bool:
    """Detect reasoning models that need the gpt-5 preset (ApplyPatchTool,
    gpt5 condenser) rather than the default preset."""
    m = model.lower()
    return any(tag in m for tag in ("gpt-5", "gpt5", "o1", "o3", "o4-mini"))


def _run(req: dict) -> dict:
    from openhands.sdk import Conversation, LLM, Message, TextContent
    from openhands.tools.preset.default import get_default_agent
    from openhands.tools.preset.gpt5 import get_gpt5_agent

    task: str = req["task"]
    workspace_dir = Path(req["workspace_dir"]).resolve()
    persistence_dir = Path(req["persistence_dir"]).resolve()
    model: str = req["model"]
    api_key: str = req["api_key"]
    conversation_id_str = req.get("conversation_id")
    system_prompt = req.get("system_prompt")
    max_iterations = int(req.get("max_iterations", 30))
    reasoning_effort = req.get("reasoning_effort")  # "low" | "medium" | "high" | None
    preset_override = req.get("preset")  # "default" | "gpt5" | None (auto)
    # Level-A memory-compression control. "default" keeps the preset's baked-in
    # LLMSummarizingCondenser(80, 4); "none" disables compaction; "llm_summary"
    # uses a controlled budget. None/missing == "default" (preserves prior runs).
    condenser_kind = (req.get("condenser") or "default").lower()

    workspace_dir.mkdir(parents=True, exist_ok=True)
    persistence_dir.mkdir(parents=True, exist_ok=True)
    # Pre-seed common memory subdir so the agent's file_editor.create can write
    # inside it without the directory-vs-file confusion that gpt-4o-mini hits
    # when asked to "create notes/x.md" in an empty workspace.
    (workspace_dir / "notes").mkdir(parents=True, exist_ok=True)

    before_snapshot = _snapshot_workspace(workspace_dir)

    use_gpt5_preset = (
        preset_override == "gpt5"
        if preset_override
        else _is_reasoning_model(model)
    )

    llm_kwargs: dict = dict(
        usage_id="agingbench",
        model=model,
        api_key=api_key,
        max_output_tokens=1024,
    )
    # Reasoning-model-specific fields. `reasoning_effort` is tied to the actual
    # MODEL (not the tool preset) — forcing `preset: gpt5` on a non-reasoning
    # model (e.g. gpt-4o-mini) to get ApplyPatchTool must NOT send
    # `reasoning_effort`, which those models reject. Reasoning models also
    # ignore `temperature`, so only set it on non-reasoning models.
    if _is_reasoning_model(model):
        llm_kwargs["reasoning_effort"] = reasoning_effort or "low"
    else:
        llm_kwargs["temperature"] = 0.0
    llm = LLM(**llm_kwargs)

    # gpt5 preset: TerminalTool + ApplyPatchTool + TaskTrackerTool (reasoning-
    # model-friendly edit surface). default preset: TerminalTool +
    # FileEditorTool + TaskTrackerTool.
    if use_gpt5_preset:
        agent = get_gpt5_agent(llm=llm, cli_mode=True)
    else:
        agent = get_default_agent(llm=llm, cli_mode=True)

    if system_prompt:
        # Preserve agent's existing system_prompt_kwargs ({"cli_mode": True}).
        existing = dict(agent.system_prompt_kwargs or {})
        existing["extra_system_message"] = system_prompt
        agent = agent.model_copy(update={"system_prompt_kwargs": existing})

    # ---- Level-A: control the in-context condenser (memory compression) ----
    # The Agent.condenser field is swappable; override the preset's baked-in
    # LLMSummarizingCondenser when the benchmark asks for a specific regime.
    condenser_info: dict = {"kind": condenser_kind}
    if condenser_kind == "none":
        from openhands.sdk.context.condenser import NoOpCondenser
        agent = agent.model_copy(update={"condenser": NoOpCondenser()})
    elif condenser_kind == "llm_summary":
        from openhands.sdk.context.condenser import LLMSummarizingCondenser
        c_max = int(req.get("condenser_max_size") or 80)
        c_keep = int(req.get("condenser_keep_first") or 4)
        # Constraint (SDK validator): condensation needs at least one event in
        # the tail half, i.e. max_size//2 - keep_first - 1 >= 1, so
        # keep_first <= max_size//2 - 2. Clamp instead of crashing on a
        # too-aggressive YAML, and record that we did. (max_size < 6 cannot
        # satisfy this for any keep_first; the SDK then raises a clear error.)
        c_keep_max = max(0, c_max // 2 - 2)
        if c_keep > c_keep_max:
            condenser_info["keep_first_clamped_from"] = c_keep
            c_keep = c_keep_max
        cond_llm = llm.model_copy(update={"usage_id": "condenser"})
        agent = agent.model_copy(update={"condenser": LLMSummarizingCondenser(
            llm=cond_llm, max_size=c_max, keep_first=c_keep,
        )})
        condenser_info["requested_max_size"] = c_max
        condenser_info["requested_keep_first"] = c_keep
    elif condenser_kind in ("default", "preset", ""):
        pass  # keep the preset's baked-in condenser unchanged
    else:
        condenser_info["warning"] = (
            f"unknown condenser '{condenser_kind}'; kept preset default"
        )
    # Record what is actually active so the caller can verify the regime.
    _active = getattr(agent, "condenser", None)
    condenser_info["active_class"] = type(_active).__name__ if _active else None
    condenser_info["active_max_size"] = getattr(_active, "max_size", None)
    condenser_info["active_keep_first"] = getattr(_active, "keep_first", None)

    conv_id: uuid.UUID | None = (
        uuid.UUID(conversation_id_str) if conversation_id_str else None
    )

    events: list = []
    token_usage = {"input": 0, "output": 0, "cost": 0.0}

    def event_cb(ev):
        events.append(ev)

    conv = Conversation(
        agent=agent,
        workspace=str(workspace_dir),
        persistence_dir=str(persistence_dir),
        conversation_id=conv_id,
        callbacks=[event_cb],
        max_iteration_per_run=max_iterations,
        stuck_detection=True,
        delete_on_close=False,
        visualizer=None,  # suppress default stdout visualizer; stdout is our JSON channel
    )

    conv.send_message(Message(role="user", content=[TextContent(text=task)]))
    conv.run()

    # Aggregate response text from:
    #   (a) assistant MessageEvents (plain text replies), and
    #   (b) ActionEvent arguments for the `finish` tool (final-answer payload)
    assistant_texts: list[str] = []
    for ev in events:
        cls = type(ev).__name__
        if cls == "MessageEvent":
            msg = getattr(ev, "llm_message", None)
            if msg is None or getattr(msg, "role", "") != "assistant":
                continue
            for c in getattr(msg, "content", []):
                txt = getattr(c, "text", None)
                if txt:
                    assistant_texts.append(txt)
        elif cls == "ActionEvent":
            # Finish tool carries the final answer in its message argument.
            tool_name = getattr(ev, "tool_name", "") or ""
            if tool_name.lower() in ("finish", "finishtool"):
                action = getattr(ev, "action", None)
                if action is not None:
                    msg_arg = getattr(action, "message", None)
                    if msg_arg:
                        assistant_texts.append(str(msg_arg))

    # Token and cost aggregation from conversation stats
    try:
        stats = conv.conversation_stats
        usage = stats.get_combined_metrics().accumulated_token_usage
        token_usage["input"] = int(getattr(usage, "prompt_tokens", 0))
        token_usage["output"] = int(getattr(usage, "completion_tokens", 0))
        token_usage["cost"] = float(
            getattr(stats.get_combined_metrics(), "accumulated_cost", 0.0)
        )
    except Exception:
        pass

    after_snapshot = _snapshot_workspace(workspace_dir)
    files_changed = _diff_workspace(before_snapshot, after_snapshot)

    conv_id_out = str(conv.state.id) if hasattr(conv, "state") and hasattr(conv.state, "id") else (
        str(conv_id) if conv_id else str(uuid.uuid4())
    )
    # `iterations` is surfaced to the benchmark as `num_turns` — it should
    # reflect total agent ACTIVITY (tool calls + distinct assistant messages),
    # not just the 2 MessageEvents that bracket any run (user prompt + final
    # assistant text). We count: ActionEvents (each tool invocation) +
    # assistant MessageEvents beyond the first (additional replies during
    # tool loops). This makes the metric comparable to Claude Code's num_turns.
    n_actions = sum(1 for e in events if type(e).__name__ == "ActionEvent")
    n_assistant_msgs = sum(
        1 for e in events
        if type(e).__name__ == "MessageEvent"
        and getattr(getattr(e, "llm_message", None), "role", "") == "assistant"
    )
    iterations = n_actions + n_assistant_msgs

    # How many times the condenser actually fired this run. Condensation emits a
    # dedicated event; match by class name so this is robust across SDK layouts
    # and reports 0 cleanly when compaction never triggered (e.g. NoOp, or the
    # event view never exceeded max_size).
    condensations = sum(
        1 for e in events if "Condensation" in type(e).__name__
    )
    condenser_info["fired"] = condensations

    conv.close()

    return {
        "text": "\n".join(assistant_texts).strip(),
        "input_tokens": token_usage["input"],
        "output_tokens": token_usage["output"],
        "cost_usd": token_usage["cost"],
        "conversation_id": conv_id_out,
        "files_changed": files_changed,
        "iterations": iterations,
        "condenser": condenser_info,
        "condensations": condensations,
        "error": None,
    }


def main():
    # OpenHands SDK components sometimes write to stdout (e.g., banner, log
    # messages that slip past the visualizer suppression). We pin stdout to
    # stderr for the duration of the run and only print our JSON response on
    # the real stdout at the end, so the caller can json.loads() cleanly.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        try:
            req = json.loads(sys.stdin.read())
        except Exception as e:
            print(json.dumps({"error": f"bad request: {e}", "text": ""}), file=real_stdout)
            return
        try:
            resp = _run(req)
        except Exception:
            resp = {
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "conversation_id": req.get("conversation_id") or "",
                "files_changed": [],
                "iterations": 0,
                "error": traceback.format_exc(),
            }
        real_stdout.write(json.dumps(resp))
        real_stdout.flush()
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    main()
