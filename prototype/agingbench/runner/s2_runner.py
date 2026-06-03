"""
agingbench/runner/s2_runner.py — S2 ScenarioRunner.

Implements the S2 session state machine for the Personal Finance & Lifestyle
Assistant scenario. Mechanisms covered: Compression + Revision (see the
scenario README for the full metric stack). Headline aging signal:
constraint_precision(t).

Session loop
------------
For each session 0..9:
  1. Read M_t (or profile_text if session 0 / no_memory)
  2. Apply any constraint updates scheduled for this session
  3. Run 5 lifestyle tasks via ReferenceAgent (ReAct with check_constraints tool)
  4. Run 10 held-out eval probes against agent (read-only, does NOT modify M_t)
  5. Score CVR(t) and constraint_precision(t); plus lag_recall and compounding
  6. Build session interaction history and write to memory: M_{t+1} = U(M_t, H_t)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from .base import BaseRunner, RunResult
from .trace import TraceLogger
from ..metrics.aging import AgingCurve, compute_half_life, compute_decay_slope
from ..core.memory.base import MemoryPolicy
from ..core.memory.append_only import AppendOnlyPolicy
from ..core.agent import AgentInterface, ReferenceAgent
from ..core.tools import ToolSpec, ToolRegistry
from ..scenarios.s2_lifestyle_assistant.tools import (
    load_profile,
    check_constraints as _check_constraints_fn,
    TOOL_SPEC as _S2_TOOL_SPEC,
)
from ..scenarios.s2_lifestyle_assistant.validator import (
    load_eval_probes,
    load_session_facts,
    load_compounding_probes,
    score_session,
    score_recall,
    compute_lag_recall,
    compute_compounding_score,
    compute_compounding_fresh_score,
)


_S2_DIR = Path(__file__).parent.parent / "scenarios" / "s2_lifestyle_assistant"


def _load_session_tasks() -> list[dict]:
    with open(_S2_DIR / "session_tasks.json") as f:
        return json.load(f)["sessions"]


def _load_constraint_updates() -> list[dict]:
    with open(_S2_DIR / "constraint_updates.json") as f:
        return json.load(f)["updates"]


def _build_tool_registry(memory_reader, tool_kind: str = "read") -> ToolRegistry:
    """Tool registry for S2: memory-access tool (read or lookup) + check_constraints."""
    if tool_kind == "lookup":
        from ..core.tool_helpers import build_lookup_tool_registry
        registry = build_lookup_tool_registry(memory_reader)
    elif tool_kind == "read":
        from ..core.tool_helpers import build_default_tool_registry
        registry = build_default_tool_registry(memory_reader)
    else:
        raise ValueError(
            f"S2 tool_kind must be 'read' or 'lookup', got {tool_kind!r}"
        )

    def _tool_fn(args: dict):
        category = args.get("category", "")
        memory_text = memory_reader()
        return _check_constraints_fn(category, memory_text)

    registry.register(ToolSpec(
        name="check_constraints",
        version="1.0.0",
        description=_S2_TOOL_SPEC["description"],
        parameters=_S2_TOOL_SPEC["input_schema"],
        fn=_tool_fn,
    ))
    return registry


class S2Runner(BaseRunner):
    """
    ScenarioRunner for Scenario S2 — Personal Finance & Lifestyle Assistant.

    Session state machine
    ---------------------
    Session 0:
      memory = profile_text (initial user profile)
      run tasks via agent → interaction history
      run eval probes → CVR(0), constraint_precision(0)
      memory_policy.write(profile_text + interaction history)

    Sessions 1..9:
      [no_memory]         : memory = profile_text (control)
      [summarize/append]  : memory = memory_policy.read()
      if session has constraint update: inject update into task flow
      run tasks via agent → interaction history
      run eval probes → CVR(t), constraint_precision(t)
      memory_policy.write(memory + interaction history)
    """

    SCENARIO_ID = "s2_lifestyle_assistant"

    def __init__(
        self,
        memory_policy: MemoryPolicy,
        llm,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        self_plan: bool = False,
        agent_class: type[AgentInterface] = ReferenceAgent,
        generated_data: dict | None = None,
        tool_kind: str = "read",
        interaction_format: str = "with_tool_findings",
    ):
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        if self.llm is not None:
            self.llm.tracer = self.tracer
        self.sut_id = sut_id
        self.agent_class = agent_class
        self.self_plan = self_plan
        self.tool_kind = tool_kind
        if interaction_format not in ("thin", "with_tool_findings"):
            raise ValueError(
                f"interaction_format must be 'thin' or 'with_tool_findings', got {interaction_format!r}"
            )
        self.interaction_format = interaction_format

        # Load scenario data (from generator or curated files)
        if generated_data:
            self.profile = generated_data["source_profile"]
            self.profile_text = self.profile["profile_text"]
            self.sessions = generated_data["session_tasks"]["sessions"]
            self.constraint_updates = generated_data["constraint_updates"]["updates"]
            self.eval_probes = generated_data["eval_probes"]["probes"]
            self.session_facts = generated_data["session_facts"]["facts"]
            self.compounding_probes = generated_data["compounding_probes"]["probes"]
        else:
            self.profile = load_profile()
            self.profile_text = self.profile["profile_text"]
            self.sessions = _load_session_tasks()
            self.constraint_updates = _load_constraint_updates()
            self.eval_probes = load_eval_probes()
            self.session_facts = load_session_facts()
            self.compounding_probes = load_compounding_probes()

        # Build update schedule: {session_number: update_dict}
        self._update_schedule = {u["session"]: u for u in self.constraint_updates}

        # Index facts by session for quick lookup
        self._facts_by_session = {f["session"]: f for f in self.session_facts}

        # Track current effective profile text (with updates applied)
        self._current_profile_text = self.profile_text

        # Model info for tracing
        self._model_id = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
        self._provider = "local_hf" if hasattr(llm, "tok") else "litellm"

        # Additive post-session hooks. Default empty -> zero behavior change.
        # Each hook is called as hook(runner, session_idx) at the end of each
        # session, after memory write and metric scoring. Used by the typed-
        # state overlay (E1) and runtime controller (E2) experiments to drive
        # accumulator initialization, delta application, and threshold-trigger
        # action dispatch from outside the runner without modifying its core
        # logic. Callers register hooks via runner._post_session_hooks.append(fn).
        self._post_session_hooks: list = []
        # Snapshot of the most recently scored session_results entry, surfaced
        # for hooks to read. None when no session has yet completed.
        self._latest_session_record: Optional[dict] = None
        # Cumulative list of raw per-session interaction histories, surfaced
        # for hooks that retroactively re-summarize prior sessions (E2
        # controller's `retro_recompact` action). Populated each session after
        # _build_interaction_history is called. Default empty; existing runs
        # don't read it, so prior results reproduce bit-identically.
        self._raw_session_histories: list[str] = []

    def run(self, n_sessions: int = 10, seed: int = 42) -> dict:
        """
        Run the S2 loop for n_sessions.

        Returns dict with:
          - precision_curve: AgingCurve for constraint_precision(t) (headline)
          - cvr_curve: AgingCurve for constraint_adherence(t) = 1 - CVR(t)
          - lag_recall_curve: AgingCurve for lag_recall(t)
          - compounding_curve: AgingCurve for compounding_accuracy(t)
          - session_results: per-session scoring details
        """
        import random as _random
        # Seed runtime randomness so multi-seed runs produce different results.
        _random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        self.memory_policy.reset()
        self._current_profile_text = self.profile_text
        is_no_memory = type(self.memory_policy).__name__ == "NoMemoryPolicy"
        progress_on = os.getenv("AGINGBENCH_S2_PROGRESS", "1").lower() not in {
            "0", "false", "no", "off"
        }
        task_log_every = max(1, int(os.getenv("AGINGBENCH_S2_TASK_LOG_EVERY", "1")))
        probe_log_every = max(1, int(os.getenv("AGINGBENCH_S2_PROBE_LOG_EVERY", "2")))

        run_t0 = time.time()

        def _fmt_elapsed(start: float) -> str:
            delta = int(time.time() - start)
            m, s = divmod(delta, 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h:02d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        def _progress(msg: str, session_start: float | None = None) -> None:
            if not progress_on:
                return
            run_elapsed = _fmt_elapsed(run_t0)
            if session_start is None:
                print(f"  [S2][progress][run {run_elapsed}] {msg}", flush=True)
                return
            session_elapsed = _fmt_elapsed(session_start)
            print(
                f"  [S2][progress][run {run_elapsed} | session {session_elapsed}] {msg}",
                flush=True,
            )

        # Initialize memory with the user profile.
        if not is_no_memory:
            self.memory_policy.write(self.profile_text, llm=self.llm)

        exposures: list[int] = []
        cvr_scores: list[float] = []
        precision_scores: list[float] = []
        lag_recall_scores: list[float] = []
        compounding_scores: list[float] = []
        compounding_fresh_scores: list[float | None] = []
        # Persist probe outputs across sessions so each probe is asked ONCE
        # (at its cohort session) and cumulative scoring reuses the stored
        # answer. Without this, the dense fresh schedule would re-ask every
        # probe every session and blow up wall time O(n^2).
        compounding_outputs_persist: dict[str, str] = {}
        session_results: list[dict] = []

        run_span = self.tracer.log(
            "run_start",
            parent_span_id=None,
            sut_id=self.sut_id,
            scenario=self.SCENARIO_ID,
            seed=seed,
            n_sessions=n_sessions,
            policy=type(self.memory_policy).__name__,
            self_plan=self.self_plan,
            **{"gen_ai.request.model": self._model_id},
        )

        actual_sessions = min(n_sessions, len(self.sessions))
        _progress(
            f"starting run: sessions={actual_sessions}, policy={type(self.memory_policy).__name__}, "
            f"self_plan={self.self_plan}"
        )

        for session_idx in range(actual_sessions):
            session_t0 = time.time()
            session_data = self.sessions[session_idx]
            session_span = self.tracer.log(
                "session_start",
                parent_span_id=run_span,
                session=session_idx,
            )

            lag_targets = sum(1 for fact in self.session_facts if fact["session"] < session_idx)
            comp_targets = sum(
                1
                for cprobe in self.compounding_probes
                if cprobe["available_from_session"] <= session_idx
            )
            _progress(
                f"session {session_idx + 1}/{actual_sessions} start: "
                f"tasks={len(session_data.get('tasks', []))}, eval_probes={len(self.eval_probes)}, "
                f"lag_probes={lag_targets}, compounding_probes={comp_targets}",
                session_t0,
            )

            # ---- Determine current memory text ----
            # no_memory: profile-only, no cross-session carry-over.
            # baseline:  SUT memory_policy as configured.
            if is_no_memory:
                memory_text = self._current_profile_text
            else:
                memory_text = self.memory_policy.read() or self._current_profile_text

            # ---- Check for constraint update at this session ----
            update = self._update_schedule.get(session_idx)
            if update:
                self._apply_constraint_update(update)
                _progress(
                    f"session {session_idx + 1}: applied constraint update {update['constraint_id']}",
                    session_t0,
                )
                self.tracer.log(
                    "constraint_update",
                    parent_span_id=session_span,
                    session=session_idx,
                    constraint_id=update["constraint_id"],
                    update_type=update["type"],
                    new_rule=update["new_rule"],
                )

            # ---- Build agent with check_constraints tool ----
            def _memory_reader():
                if is_no_memory:
                    return self._current_profile_text
                return self.memory_policy.read() or self._current_profile_text

            tool_registry = _build_tool_registry(_memory_reader, tool_kind=self.tool_kind)
            agent = self.agent_class(
                llm=self.llm,
                memory_policy=self.memory_policy,
                tools=tool_registry,
                max_turns=8,
            )

            # ---- Run session tasks ----
            tasks = session_data["tasks"]
            task_outputs = []
            task_tool_calls: list[list[dict]] = []  # parallel to task_outputs
            all_trace_events = []
            # Capture accumulator probe outputs paired with their gold values
            # so dependency_scorer.score_accumulator can grade revision-aging
            # for the budget tracking (Ledger-QA flagship pattern). Previously
            # the runner discarded which task was an accumulator probe, so the
            # accumulator track was generated and executed but never scored.
            accumulator_probe_results: list[dict] = []

            if self.self_plan:
                # Self-planned mode: give the agent all tasks at once and let
                # it decide how to approach them (order, tool usage, planning).
                task_outputs, all_trace_events = self._run_self_planned_session(
                    agent, tasks, session_idx, session_span, update,
                )
                # In self-plan mode the agent processes tasks in a single
                # turn, so we cannot easily attribute outputs to specific
                # accumulator probes. Skip accumulator scoring in that mode.
            else:
                # Runner-controlled mode: feed tasks one by one.
                for task_i, task in enumerate(tasks, start=1):
                    task_text = task["text"]
                    # For constraint update tasks, prepend the update instruction
                    # to the first task's text so the agent both acknowledges the
                    # update AND answers the original probe in one turn. Previously
                    # this REPLACED the first task's text, silently dropping the
                    # probe answer on every update session (~20% of headline-metric
                    # variance on N=10 runs).
                    if update and task == tasks[0] and update.get("update_text"):
                        task_text = update["update_text"] + "\n\n" + task["text"]

                    if task_i == 1 or task_i % task_log_every == 0 or task_i == len(tasks):
                        _progress(
                            f"session {session_idx + 1}: running task {task_i}/{len(tasks)} ({task['id']})",
                            session_t0,
                        )

                    result = agent.run_session(task_text, session_id=session_idx)

                    task_outputs.append(result["output"])
                    task_tool_calls.append(result.get("tool_calls", []))

                    # If this task is an accumulator probe, record the gold
                    # value alongside the agent's response so the scorer can
                    # extract a numeric answer and compute mean error.
                    if task.get("category") == "accumulator_probe":
                        accumulator_probe_results.append({
                            "session": session_idx,
                            "task_id": task.get("id"),
                            "question": task_text,
                            "response_text": result["output"],
                            "gold_value": task.get("gold_value"),
                            "eval_keywords": task.get("eval_keywords", []),
                        })

                    # Collect trace events for tool usage tracking
                    for tc in result.get("tool_calls", []):
                        all_trace_events.append({
                            "tool_name": tc["tool"],
                            "input": tc.get("input"),
                        })

                    self.tracer.log(
                        "task_completed",
                        parent_span_id=session_span,
                        session=session_idx,
                        task_id=task["id"],
                        constraints_tested=task.get("constraints_tested", []),
                        n_tool_calls=len(result.get("tool_calls", [])),
                        turns=result.get("turns", 0),
                    )

                    if task_i == 1 or task_i % task_log_every == 0 or task_i == len(tasks):
                        _progress(
                            f"session {session_idx + 1}: finished task {task_i}/{len(tasks)} "
                            f"(turns={result.get('turns', 0)}, tools={len(result.get('tool_calls', []))})",
                            session_t0,
                        )

            # ---- Run eval probes (read-only scoring) ----
            probe_outputs = []
            total_eval_probes = len(self.eval_probes)
            _progress(
                f"session {session_idx + 1}: starting eval probes ({total_eval_probes})",
                session_t0,
            )
            for probe_i, probe in enumerate(self.eval_probes, start=1):
                probe_result = agent.run_session(probe["text"], session_id=session_idx)
                probe_outputs.append(probe_result["output"])
                # Track tool calls from eval probes too
                for tc in probe_result.get("tool_calls", []):
                    all_trace_events.append({
                        "tool_name": tc["tool"],
                        "input": tc.get("input"),
                    })

                if (
                    probe_i == 1
                    or probe_i % probe_log_every == 0
                    or probe_i == total_eval_probes
                ):
                    _progress(
                        f"session {session_idx + 1}: eval probe {probe_i}/{total_eval_probes} complete",
                        session_t0,
                    )

            # ---- Lag recall probes (probe facts from prior sessions) ----
            lag_recall_outputs: dict[str, str] = {}
            if session_idx > 0:
                lag_facts = [fact for fact in self.session_facts if fact["session"] < session_idx]
                total_lag = len(lag_facts)
                lag_log_every = max(1, total_lag // 5)
                _progress(
                    f"session {session_idx + 1}: starting lag recall probes ({total_lag})",
                    session_t0,
                )
                for lag_i, fact in enumerate(lag_facts, start=1):
                    if fact["session"] >= session_idx:
                        continue  # only probe facts from earlier sessions
                    recall_result = agent.run_session(
                        fact["recall_question"], session_id=session_idx
                    )
                    lag_recall_outputs[fact["id"]] = recall_result["output"]

                    if lag_i == 1 or lag_i % lag_log_every == 0 or lag_i == total_lag:
                        _progress(
                            f"session {session_idx + 1}: lag probe {lag_i}/{total_lag} complete",
                            session_t0,
                        )

                lag_result = compute_lag_recall(
                    session_idx, self.session_facts, lag_recall_outputs
                )
            else:
                lag_result = {"recall_by_lag": {}, "recall_details": [], "overall_recall": 1.0}

            lag_recall_scores.append(lag_result["overall_recall"])

            # ---- Compounding probes (multi-session context synthesis) ----
            # Ask each probe exactly once — at its cohort session. Reuse the
            # stored output in later sessions' cumulative scoring. The agent's
            # memory has already evolved past the probe when asked in a later
            # session, so "re-asking" doesn't give more signal than asking at
            # the cohort moment (and costs O(n) per session instead of O(1)).
            eligible_compounding = [
                cprobe
                for cprobe in self.compounding_probes
                if cprobe["available_from_session"] <= session_idx
            ]
            to_ask = [
                cprobe for cprobe in eligible_compounding
                if cprobe["id"] not in compounding_outputs_persist
            ]
            total_comp = len(to_ask)
            comp_log_every = max(1, total_comp // 4) if total_comp else 1
            if total_comp:
                _progress(
                    f"session {session_idx + 1}: starting compounding probes ({total_comp} new)",
                    session_t0,
                )
            for comp_i, cprobe in enumerate(to_ask, start=1):
                cp_result = agent.run_session(
                    cprobe["text"], session_id=session_idx
                )
                compounding_outputs_persist[cprobe["id"]] = cp_result["output"]
                if comp_i == 1 or comp_i % comp_log_every == 0 or comp_i == total_comp:
                    _progress(
                        f"session {session_idx + 1}: compounding probe {comp_i}/{total_comp} complete",
                        session_t0,
                    )
            compounding_outputs = compounding_outputs_persist

            compounding_result = compute_compounding_score(
                session_idx, self.compounding_probes, compounding_outputs
            )
            compounding_scores.append(compounding_result["compounding_accuracy"])
            # Fresh (non-cumulative) compounding: scores only the probe
            # introduced at this session, so the curve reports rate-of-decay
            # rather than the binary cliff of the cumulative metric.
            compounding_fresh_result = compute_compounding_fresh_score(
                session_idx, self.compounding_probes, compounding_outputs
            )
            compounding_fresh_scores.append(
                compounding_fresh_result["compounding_fresh_accuracy"]
            )

            # ---- Score session (CVR + precision) ----
            session_score = score_session(
                agent_outputs=probe_outputs,
                probes=self.eval_probes,
                session_idx=session_idx,
            )

            cvr = session_score["cvr"]
            cp = session_score.get("constraint_precision", 1.0)

            exposures.append(session_idx)
            cvr_scores.append(cvr)
            precision_scores.append(cp)

            # Concatenate all task outputs (truncated) so dependency_scorer's
            # forget_accuracy can scan for invalidated keywords. Without this,
            # forget_accuracy was silently saturated at 1.0 because no
            # scenario-side text was reachable from session_results.
            task_outputs_text = " ".join(
                str(o)[:2000] for o in task_outputs if o
            )

            # Token-cap diagnostics: track response tokens to detect
            # max_new_tokens truncation confounds.
            from agingbench.metrics.aging import count_response_tokens
            task_token_counts = [
                count_response_tokens(self.llm, str(o)) for o in task_outputs if o
            ]
            valid_tokens = [t for t in task_token_counts if t >= 0]

            session_results.append({
                "session": session_idx,
                "cvr": cvr,
                "constraint_precision": cp,
                "n_violations": session_score["n_violations"],
                "violated_constraints": session_score["violated_constraints"],
                "precision_per_probe": session_score.get("precision_per_probe", []),
                "lag_recall": lag_result["overall_recall"],
                "lag_by_distance": lag_result["recall_by_lag"],
                "compounding_accuracy": compounding_result["compounding_accuracy"],
                "compounding_fresh_accuracy": compounding_fresh_result["compounding_fresh_accuracy"],
                # Revision-aging scoring inputs (consumed by dependency_scorer)
                "accumulator_probes": accumulator_probe_results,
                "task_outputs_text": task_outputs_text,
                "response_tokens_per_task": task_token_counts,
                "response_tokens_max": max(valid_tokens) if valid_tokens else 0,
            })
            # Expose latest session record for additive post-session hooks
            # (E1 typed-state, E2 controller). No-op when no hooks registered.
            self._latest_session_record = session_results[-1]

            self.tracer.log(
                "session_scored",
                parent_span_id=session_span,
                session=session_idx,
                cvr=cvr,
                lag_recall=lag_result["overall_recall"],
                lag_by_distance=lag_result["recall_by_lag"],
                compounding_accuracy=compounding_result["compounding_accuracy"],
                n_violations=session_score["n_violations"],
                violated_constraints=session_score["violated_constraints"],
                t_writes=getattr(self.memory_policy, "n_writes", 0),
            )

            violated_str = (
                f"  violated={session_score['violated_constraints']}"
                if session_score["violated_constraints"]
                else ""
            )
            lag_str = f"  lag_recall={lag_result['overall_recall']:.2f}" if session_idx > 0 else ""
            comp_str = (
                f"  comp={compounding_result['n_passed']}/{compounding_result['n_available']}"
                if compounding_result["n_available"] > 0
                else ""
            )
            print(
                f"  [S2] Session {session_idx:2d}  CVR={cvr:.3f}  "
                f"precision={cp:.3f}  "
                f"({session_score['n_violations']}/{session_score['n_probes']} probes)"
                f"{violated_str}{lag_str}{comp_str}"
            )
            _progress(
                f"session {session_idx + 1}/{actual_sessions} scored: "
                f"precision={cp:.3f}, cvr={cvr:.3f}, lag={lag_result['overall_recall']:.3f}",
                session_t0,
            )

            # ---- Write interaction history to memory ----
            if not is_no_memory:
                interaction_history = self._build_interaction_history(
                    session_idx, tasks, task_outputs, update,
                    task_tool_calls=task_tool_calls,
                )
                # Preserve raw history for retroactive-recompact hooks (E2 A4c).
                # No-op for any run that doesn't register that hook.
                self._raw_session_histories.append(interaction_history)

                # Policy-aware write. AppendOnly is episodic — one session =
                # one entry — so we pass just the new interaction_history.
                # Summarize-style policies (and anything else) get "current
                # memory + new" so they can re-compress against the full
                # history. Before this split, AppendOnly was receiving the
                # cumulative snapshot too, turning it into an O(N^2)
                # snapshot store with heavy redundancy (see lesion_queue
                # audit 2026-04-23).
                if isinstance(self.memory_policy, AppendOnlyPolicy):
                    # Episodic write: the new content == interaction_history.
                    written_payload = interaction_history
                    self.memory_policy.write(interaction_history, llm=self.llm)
                else:
                    # summarize_store and similar: write() internally concats
                    # with prior memory, so pass only the new content.
                    if self.memory_policy.read():
                        written_payload = interaction_history
                    else:
                        written_payload = self._current_profile_text + "\n\n" + interaction_history
                    self.memory_policy.write(written_payload, llm=self.llm)

                compressed = self.memory_policy.read()
                in_tok = getattr(self.memory_policy, "last_input_tokens", 0)
                out_tok = getattr(self.memory_policy, "last_output_tokens", 0)
                if session_results:
                    session_results[-1]["memory_write_tokens"] = out_tok

                self.tracer.log_llm_call(
                    parent_span_id=session_span,
                    model=self._model_id,
                    provider=self._provider,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    input_preview=written_payload,
                    output_preview=compressed or "",
                    thought=getattr(self.llm, "last_thought", ""),
                    cycle=session_idx,
                )
                _progress(
                    f"session {session_idx + 1}: memory write complete "
                    f"(in_tok={in_tok}, out_tok={out_tok})",
                    session_t0,
                )

            # ---- Post-session hooks (typed-state overlay, runtime controller) ----
            # No-op when self._post_session_hooks is empty (default).
            for hook in self._post_session_hooks:
                try:
                    hook(self, session_idx)
                except Exception as e:
                    print(f"  [S2][hook-error] post-session hook raised: {e!r}", flush=True)

            self.tracer.log(
                "session_end",
                parent_span_id=session_span,
                session=session_idx,
                cvr=cvr,
            )
            _progress(f"session {session_idx + 1}/{actual_sessions} end", session_t0)

        # ---- Build aging curves ----
        # CVR is a violation rate (0 = good, 1 = bad), but AgingCurve expects
        # higher = better. We invert: constraint_adherence = 1 - CVR
        adherence_scores = [1.0 - c for c in cvr_scores]

        cvr_curve = AgingCurve(
            exposures=exposures,
            scores=adherence_scores,
            scenario=self.SCENARIO_ID,
            sut_id=self.sut_id,
        )

        # Lag recall curve — overall recall rate per session (should decay monotonically)
        lag_recall_curve = AgingCurve(
            exposures=exposures,
            scores=lag_recall_scores,
            scenario=self.SCENARIO_ID,
            sut_id=self.sut_id,
        )

        # Constraint precision curve — primary S2 aging metric (monotonically decaying)
        precision_curve = AgingCurve(
            exposures=exposures,
            scores=precision_scores,
            scenario=self.SCENARIO_ID,
            sut_id=self.sut_id,
        )

        # Compounding accuracy curve — should decay monotonically as more deps are lost
        compounding_curve = AgingCurve(
            exposures=exposures,
            scores=compounding_scores,
            scenario=self.SCENARIO_ID,
            sut_id=self.sut_id,
        )

        self.tracer.log(
            "run_end",
            parent_span_id=run_span,
            cvr_curve=list(zip(exposures, cvr_scores)),
            adherence_curve=list(zip(exposures, adherence_scores)),
            precision_curve=list(zip(exposures, precision_scores)),
            lag_recall_curve=list(zip(exposures, lag_recall_scores)),
            compounding_curve=list(zip(exposures, compounding_scores)),
            half_life=compute_half_life(precision_curve),
            slope=round(compute_decay_slope(precision_curve), 5),
            m0=precision_scores[0] if precision_scores else None,
            m_final=precision_scores[-1] if precision_scores else None,
        )

        _progress(
            f"run complete: sessions={actual_sessions}, "
            f"m_final={precision_scores[-1] if precision_scores else 0:.3f}"
        )

        return {
            "cvr_curve": cvr_curve,
            "precision_curve": precision_curve,
            "lag_recall_curve": lag_recall_curve,
            "compounding_curve": compounding_curve,
            "compounding_fresh_raw": list(zip(exposures, compounding_fresh_scores)),
            "session_results": session_results,
        }

    def _apply_constraint_update(self, update: dict) -> None:
        """Apply a mid-lifetime constraint update to the effective profile."""
        old_rule = update["old_rule"]
        new_rule = update["new_rule"]
        # Update the profile text in-place
        if old_rule in self._current_profile_text:
            self._current_profile_text = self._current_profile_text.replace(
                old_rule, new_rule
            )
        else:
            # Fallback: append the update
            self._current_profile_text += f"\n\nUPDATE: {new_rule}"
        print(f"  [S2] Constraint update applied: {update['constraint_id']} ({update['type']})")

    # ---------------------------------------------------------------- self-plan

    _SELF_PLAN_PROMPT = """You are a personal finance and lifestyle assistant. You have {n_tasks} tasks to handle this session.

Review all tasks below, then decide your approach:
- You should check the user's constraints and preferences BEFORE making recommendations.
- You may use your tools at any point (check_constraints, search_memory).
- You may handle tasks in any order if it's more efficient.
- For each task, provide a clear response respecting all user constraints.

{update_note}
Tasks for this session:
{task_list}

Think step by step. For each task, state which task you're addressing, then give your response. Format your final output as:

=== Task 1 ===
[your response to the first task listed above]

=== Task 2 ===
[your response to the second task listed above]

(and so on for all {n_tasks} tasks)"""

    def _run_self_planned_session(
        self,
        agent: ReferenceAgent,
        tasks: list[dict],
        session_idx: int,
        session_span: str,
        update: Optional[dict],
    ) -> tuple[list[str], list[dict]]:
        """
        Self-planned session: give the agent all tasks at once.

        The agent sees the full task list and decides its own approach —
        which tools to call first, what order to handle tasks, etc.

        Returns (task_outputs, trace_events) matching the runner-controlled format.
        """
        update_note = ""
        if update and update.get("update_text"):
            update_note = f"IMPORTANT UPDATE: {update['update_text']}\n"

        task_list = "\n".join(
            f"{i+1}. {t['text']}" for i, t in enumerate(tasks)
        )

        prompt = self._SELF_PLAN_PROMPT.format(
            n_tasks=len(tasks),
            update_note=update_note,
            task_list=task_list,
        )

        # Give the agent more turns for multi-task planning
        old_max = agent.max_turns
        agent.max_turns = max(old_max, len(tasks) * 3)

        result = agent.run_session(prompt, session_id=session_idx)

        agent.max_turns = old_max

        # Parse per-task outputs from the combined response
        full_output = result.get("output", "")
        task_outputs = self._parse_planned_outputs(full_output, len(tasks))

        # Collect trace events
        trace_events = []
        for tc in result.get("tool_calls", []):
            trace_events.append({
                "tool_name": tc["tool"],
                "input": tc.get("input"),
            })

        # Log as a single planned session
        self.tracer.log(
            "self_planned_session",
            parent_span_id=session_span,
            session=session_idx,
            n_tasks=len(tasks),
            n_tool_calls=len(result.get("tool_calls", [])),
            turns=result.get("turns", 0),
            planned=True,
        )

        return task_outputs, trace_events

    @staticmethod
    def _parse_planned_outputs(full_output: str, n_tasks: int) -> list[str]:
        """
        Parse per-task responses from a self-planned session output.

        Looks for '=== Task N ===' delimiters. Falls back to splitting
        evenly or returning the full output for each task.
        """
        import re
        # Try to split by === Task N === markers
        parts = re.split(r'===\s*Task\s*\d+\s*===', full_output)
        # First element is preamble (planning text), skip it
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) >= n_tasks:
            return parts[:n_tasks]

        # Fallback: try numbered list (1. ... 2. ... etc.)
        numbered = re.split(r'\n\d+\.\s+', full_output)
        numbered = [p.strip() for p in numbered if p.strip()]
        if len(numbered) >= n_tasks:
            return numbered[:n_tasks]

        # Last resort: return full output for each task
        return [full_output] * n_tasks

    def _build_interaction_history(
        self,
        session_idx: int,
        tasks: list[dict],
        task_outputs: list[str],
        update: Optional[dict],
        task_tool_calls: Optional[list[list[dict]]] = None,
    ) -> str:
        """Build a text summary of the session interaction for memory write.

        When ``self.interaction_format == "with_tool_findings"`` (default),
        a ``Tool findings:`` line summarises the structured results of
        ``check_constraints`` calls (skipping read_memory/lookup_memory
        which are redundant with the system prompt content). Opt-in
        ``"thin"`` produces a Task/Response-only shape — the pre-cleanup
        format, retained for reproducing earlier baselines.
        """
        lines = [f"--- Session {session_idx} ---"]
        if update:
            lines.append(
                f"IMPORTANT UPDATE from user: {update.get('update_text', update['new_rule'])}"
            )
            lines.append(
                f"(Changed constraint {update['constraint_id']}: {update['new_rule']})"
            )
        session_fact = self._facts_by_session.get(session_idx)
        if session_fact:
            lines.append(f"Note: {session_fact['text']}")
        include_findings = (
            self.interaction_format == "with_tool_findings"
            and task_tool_calls is not None
        )
        for i, (task, output) in enumerate(zip(tasks, task_outputs)):
            lines.append(f"Task: {task['text']}")
            if include_findings:
                tcs = task_tool_calls[i] if i < len(task_tool_calls) else []
                findings = self._summarise_tool_findings(tcs)
                if findings:
                    lines.append(f"Tool findings: {findings}")
            lines.append(f"Response: {output}")
        return "\n".join(lines)

    @staticmethod
    def _summarise_tool_findings(tool_calls: list[dict]) -> str:
        """Compress tool results into one line of structured facts.

        Includes ``check_constraints`` findings (the scenario-specific
        compute tool); skips ``read_memory`` / ``lookup_memory`` /
        ``search_memory`` since those return content already in the
        system prompt. Limits output length to keep the compactor's
        input bounded.
        """
        out = []
        for tc in tool_calls:
            name = tc.get("tool", "")
            result = tc.get("result")
            inp = tc.get("input", {}) or {}
            if name == "check_constraints" and isinstance(result, dict):
                cat = inp.get("category") or result.get("category", "?")
                found = result.get("constraints_found", []) or []
                if not found:
                    out.append(f"check_constraints({cat})→none")
                    continue
                ids = [c.get("id", "?") for c in found if isinstance(c, dict)]
                # Include the FIRST rule excerpt to give the compactor an
                # actual fact to anchor on, not just a constraint id.
                first_rule = ""
                for c in found:
                    if isinstance(c, dict) and c.get("rule"):
                        first_rule = str(c["rule"])[:80]
                        break
                ids_str = ",".join(ids[:3]) if ids else "?"
                if first_rule:
                    out.append(f"check_constraints({cat})→{ids_str}: {first_rule}")
                else:
                    out.append(f"check_constraints({cat})→{ids_str}")
        return "; ".join(out)[:600]
