"""
agingbench/runner/s6_runner_unified.py — S6 runner using FullReactAgent.

Parallel-track POC for the React-faithful unification plan
(see ``agingbench_react_unification_plan.md``). Implements the same scoring
contract as ``S6Runner`` but swaps two things:

  1. ``ReferenceAgent`` (memory injected into system prompt) →
     ``FullReactAgent`` (memory accessible only via tools).
  2. Ad-hoc ``_build_tool_registry(_memory_reader)`` callback →
     ``build_search_memory_tool(memory_policy, top_k=3)`` which exposes a
     proper search primitive over the policy's stored content.

What stays identical
--------------------
- AgingCurve construction (task_curve, recall_curve)
- partition_recall headline (recall_compression)
- recall_matrix row build
- Maintenance event handling (workspace_flush / recompact via event.apply)
- interaction_text builder (imported from S6Runner)
- Memory write semantics (single write of new content)
- BaseRunner interface — runner is dispatchable like any other S6 variant

What's new
----------
- ``session_results[*]["turn_stats"]`` — per-session list of per-round dicts
  with ``input_tokens``, ``output_tokens``, ``latency_s``, ``had_tool_call``,
  ``had_reasoning``, ``forced``. Captures from FullReactAgent's enriched
  return dict. Makes "loop is dead" visible per-run.
- ``session_results[*]["loop_utilization"]`` — small summary computed from
  turn_stats: ``turns_median``, ``tool_calls_total``, ``exhausted_count``,
  ``had_reasoning_share``.

What's intentionally dropped (for POC clarity)
----------------------------------------------
- DiagnosticMixin (oracle-probe error partitioning). The diagnostic path in the
  original S6Runner runs probes under oracle_retrieval / oracle_context
  conditions for error attribution; it isn't needed for the unification POC.
  Re-add as a separate variant if the diagnostic split is still wanted under
  tool-only retrieval.
- ``agent_class`` constructor parameter. Hard-coded to FullReactAgent here
  because the whole point is to validate the unified-agent design.

Original S6Runner is NOT modified — this is a side-by-side variant. Both
runners produce the same ``metrics.json`` shape (recall_curve, task_curve,
recall_matrix, session_results), so existing analyzers work on both.
"""

from __future__ import annotations

import os
import time
import statistics
from typing import Optional

from .base import BaseRunner
from .trace import TraceLogger
from ..metrics.aging import AgingCurve, compute_half_life, compute_decay_slope
from ..core.memory.base import MemoryPolicy
from ..core.memory.append_only import AppendOnlyPolicy
from ..core.full_react_agent import FullReactAgent, build_search_memory_tool
from ..core.tools import ToolRegistry
from ..scenarios.s6_naturalistic.validator import (
    load_session_tasks,
    load_system_prompt,
    score_task,
    score_recall_probe,
    build_recall_matrix_entry,
    partition_recall,
)


def _loop_utilization_from_turn_stats(turn_stats: list[dict]) -> dict:
    """Compute a small summary of how much the loop fired this session."""
    if not turn_stats:
        return {
            "turns_median": 0,
            "tool_calls_total": 0,
            "exhausted_count": 0,
            "had_reasoning_share": 0.0,
        }
    rounds = [s.get("round", 0) for s in turn_stats]
    return {
        "turns_median": int(statistics.median(rounds)) if rounds else 0,
        "tool_calls_total": sum(1 for s in turn_stats if s.get("had_tool_call")),
        "exhausted_count": sum(1 for s in turn_stats if s.get("forced")),
        "had_reasoning_share": (
            sum(1 for s in turn_stats if s.get("had_reasoning")) / len(turn_stats)
        ),
    }


class S6UnifiedRunner(BaseRunner):
    """S6 (Naturalistic Aging) using the unified FullReactAgent framework.

    Same scenario data, same scoring, same memory policy interface. The only
    behavioural difference from ``S6Runner`` is that the agent's memory is
    accessed via tools (``search_memory(query)``) rather than dumped into the
    system prompt. This is the POC validation for the unification plan; if
    score divergence vs ``S6Runner`` is small under matched seeds, the unified
    design preserves the headline measurement.
    """

    SCENARIO_ID = "s6_naturalistic"

    def __init__(
        self,
        memory_policy: MemoryPolicy,
        llm,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        generated_data: Optional[dict] = None,
        maintenance_events: Optional[list] = None,
        agent_max_turns: int = 10,
        search_memory_top_k: int = 3,
    ):
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        if self.llm is not None:
            self.llm.tracer = self.tracer
        self.sut_id = sut_id
        self.maintenance_events = maintenance_events or []
        self.agent_max_turns = agent_max_turns
        self.search_memory_top_k = search_memory_top_k

        if generated_data:
            self.sessions = generated_data["session_tasks"]["sessions"]
            self.system_prompt_base = generated_data["session_tasks"]["system_prompt"]
        else:
            self.sessions = load_session_tasks()
            self.system_prompt_base = load_system_prompt()

        self._model_id = (
            getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
        )
        self._provider = "local_hf" if hasattr(llm, "tok") else "litellm"

    # ----------------------------------------------------- helpers (copied)
    # _build_interaction_text and _compute_lag_curves mirror S6Runner exactly;
    # we copy them rather than import to keep this runner truly standalone.

    def _build_interaction_text(
        self,
        session_idx: int,
        session_data: dict,
        task_output: str,
    ) -> str:
        lines = [
            f"--- Research Session {session_idx} ({session_data['domain']}) ---",
        ]
        env_data = session_data.get("environment_data", "")
        if env_data:
            lines.append(f"Data source: {session_data['domain']}")
            lines.append(env_data)
        lines.append(f"Task: {session_data['task']['text']}")
        lines.append(f"Finding: {task_output}")
        return "\n".join(lines)

    @staticmethod
    def _compute_lag_curves(
        recall_matrix: dict[int, dict[int, float]],
    ) -> dict[int, list[tuple[int, float]]]:
        lag_curves: dict[int, list[tuple[int, float]]] = {}
        for eval_time, row in sorted(recall_matrix.items()):
            for origin, rate in row.items():
                lag = eval_time - origin
                lag_curves.setdefault(lag, []).append((eval_time, rate))
        return lag_curves

    # --------------------------------------------------------------- main loop

    def run(self, n_sessions: int = 15, seed: int = 42) -> dict:
        import random as _random
        _random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        self.memory_policy.reset()
        is_no_memory = type(self.memory_policy).__name__ == "NoMemoryPolicy"
        progress_on = os.getenv("AGINGBENCH_S6_PROGRESS", "1").lower() not in {
            "0", "false", "no", "off"
        }
        probe_log_every = max(1, int(os.getenv("AGINGBENCH_S6_PROBE_LOG_EVERY", "5")))

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
                print(f"  [S6/unified][progress][run {run_elapsed}] {msg}", flush=True)
                return
            session_elapsed = _fmt_elapsed(session_start)
            print(
                f"  [S6/unified][progress][run {run_elapsed} | session {session_elapsed}] {msg}",
                flush=True,
            )

        exposures: list[int] = []
        task_scores: list[float] = []
        recall_scores: list[float] = []
        recall_exposures: list[int] = []
        recall_matrix: dict[int, dict[int, float]] = {}
        session_results: list[dict] = []

        run_span = self.tracer.log(
            "run_start",
            parent_span_id=None,
            sut_id=self.sut_id,
            scenario=self.SCENARIO_ID,
            seed=seed,
            n_sessions=n_sessions,
            policy=type(self.memory_policy).__name__,
            agent="full_react_agent",
            agent_max_turns=self.agent_max_turns,
            **{"gen_ai.request.model": self._model_id},
        )

        actual_sessions = min(n_sessions, len(self.sessions))
        _progress(
            f"starting unified run: sessions={actual_sessions}, "
            f"policy={type(self.memory_policy).__name__}, "
            f"agent=FullReactAgent(max_turns={self.agent_max_turns}), "
            f"search_memory_top_k={self.search_memory_top_k}"
        )

        for session_idx in range(actual_sessions):
            session_t0 = time.time()
            session_data = self.sessions[session_idx]

            planned_probes = sum(
                len(self.sessions[s].get("recall_probes", [])) for s in range(session_idx)
            )
            _progress(
                f"session {session_idx + 1}/{actual_sessions} start: "
                f"domain={session_data['domain']}, planned_recall_probes={planned_probes}",
                session_t0,
            )

            # ---- Maintenance events ----
            for event in self.maintenance_events:
                if event.session == session_idx and not is_no_memory:
                    shock_type = event.apply(self.memory_policy, llm=self.llm)
                    self.tracer.log(
                        "life_event",
                        parent_span_id=run_span,
                        session=session_idx,
                        event_type=shock_type,
                        params=event.params,
                    )

            session_span = self.tracer.log(
                "session_start",
                parent_span_id=run_span,
                session=session_idx,
                domain=session_data["domain"],
                is_cross_reference=session_data.get("is_cross_reference", False),
            )

            # ---- Build agent ----
            # Key difference vs S6Runner: tools-only memory access. The agent's
            # system prompt does not embed memory_policy.read(); instead it
            # documents that memory is accessible via search_memory(query).
            tool_registry = ToolRegistry()
            if not is_no_memory:
                tool_registry.register(
                    build_search_memory_tool(
                        self.memory_policy, top_k=self.search_memory_top_k
                    )
                )

            agent = FullReactAgent(
                llm=self.llm,
                memory_policy=self.memory_policy,
                tools=tool_registry,
                max_turns=self.agent_max_turns,
            )
            # Inject the S6 scenario persona (loaded from session_tasks.json)
            # into the agent's system prompt. FullReactAgent's REACT_SYSTEM
            # has only {tool_descriptions}, so we prepend the persona as a
            # leading paragraph. Fixes the prior dead-code state.
            if getattr(self, "system_prompt_base", ""):
                from ..core.full_react_agent import REACT_SYSTEM as FR_REACT_SYSTEM
                agent.system_template = (
                    self.system_prompt_base.strip() + "\n\n" + FR_REACT_SYSTEM
                )

            # ---- Build task prompt ----
            env_data = session_data.get("environment_data", "")
            task_text = session_data["task"]["text"]
            if env_data:
                full_task = (
                    f"Here is data from a research source:\n\n"
                    f"{env_data}\n\n"
                    f"Task: {task_text}"
                )
            else:
                full_task = task_text

            # ---- Run primary task ----
            _progress(f"session {session_idx + 1}: primary task start", session_t0)
            task_result = agent.run_session(full_task, session_id=session_idx)
            task_output = task_result["output"]
            task_turn_stats = task_result.get("turn_stats", [])
            task_exhausted = bool(task_result.get("exhausted"))
            _progress(
                f"session {session_idx + 1}: primary task done "
                f"(turns={task_result.get('turns', 0)}, "
                f"tools={len(task_result.get('tool_calls', []))}, "
                f"exhausted={task_exhausted})",
                session_t0,
            )

            task_eval = score_task(task_output, session_data)
            self.tracer.log(
                "task_completed",
                parent_span_id=session_span,
                session=session_idx,
                task_score=task_eval["task_score"],
                keywords_found=task_eval["keywords_found"],
                keywords_missing=task_eval["keywords_missing"],
                turns=task_result.get("turns", 0),
                n_tool_calls=len(task_result.get("tool_calls", [])),
            )

            # ---- Recall probes for past sessions ----
            all_probes = []
            for s in range(session_idx):
                for probe in self.sessions[s].get("recall_probes", []):
                    inv_at = probe.get("invalidated_at_session")
                    if inv_at is not None and session_idx >= inv_at:
                        continue
                    all_probes.append(probe)

            probe_results = []
            interference_probe_results = []
            # Accumulate per-probe turn_stats so we can summarise loop usage
            # across all probes for this session.
            all_probe_turn_stats: list[dict] = []
            total_probes = len(all_probes)
            if total_probes:
                _progress(
                    f"session {session_idx + 1}: recall probes start ({total_probes})",
                    session_t0,
                )
            for probe_i, probe in enumerate(all_probes, start=1):
                probe_result = agent.run_session(
                    probe["question"], session_id=session_idx
                )
                all_probe_turn_stats.extend(probe_result.get("turn_stats", []))

                scored = score_recall_probe(
                    probe_result["output"], probe, at_session=session_idx
                )
                _mech_fail = (
                    bool(probe_result.get("exhausted"))
                    or not (probe_result.get("output") or "").strip()
                )
                scored["mechanics_failure"] = _mech_fail
                scored["is_revised"] = bool(probe.get("keywords_history"))
                scored["agent_answer"] = probe_result["output"]
                scored["probe_type"] = probe.get("probe_type")
                probe_results.append(scored)

                if probe.get("probe_type") == "interference_binding":
                    interference_probe_results.append({
                        "session": session_idx,
                        "task_id": probe.get("probe_id"),
                        "question": probe["question"],
                        "response_text": probe_result["output"],
                        "gold_value": probe.get("gold_value"),
                        "distractor_value": probe.get("distractor_value"),
                        "mechanics_failure": _mech_fail,
                    })

                if (
                    probe_i == 1
                    or probe_i % probe_log_every == 0
                    or probe_i == total_probes
                ):
                    _progress(
                        f"session {session_idx + 1}: recall probe {probe_i}/{total_probes} complete",
                        session_t0,
                    )

            # ---- Build recall matrix + headline partition ----
            matrix_row: dict[int, float] = {}
            _rp = partition_recall(probe_results)
            recall_all = _rp["recall_all"]
            recall_excl_binding = _rp["recall_excl_binding"]
            recall_compression = _rp["recall_compression"]
            n_mech_fail = _rp["n_mechanics_failures"]

            if probe_results:
                matrix_row = build_recall_matrix_entry(
                    session_idx, self.sessions, probe_results
                )
                recall_matrix[session_idx] = matrix_row
            else:
                recall_matrix[session_idx] = {}

            overall_recall = recall_all if recall_all is not None else 0.0

            exposures.append(session_idx)
            task_scores.append(task_eval["task_score"])
            if recall_compression is not None:
                recall_exposures.append(session_idx)
                recall_scores.append(recall_compression)

            # ---- Loop-utilization summary (new field, unique to unified runner) ----
            combined_turn_stats = list(task_turn_stats) + list(all_probe_turn_stats)
            loop_util = _loop_utilization_from_turn_stats(combined_turn_stats)

            # Truncated task_outputs_text for downstream forget_accuracy scoring
            probe_text = " ".join(
                str(p.get("agent_answer", ""))[:300] for p in probe_results
            )
            task_outputs_text = (str(task_output)[:1000] + " " + probe_text).strip()

            from agingbench.metrics.aging import count_response_tokens
            task_tokens = count_response_tokens(self.llm, task_output)
            probe_token_counts = [
                count_response_tokens(self.llm, p.get("agent_answer", ""))
                for p in probe_results
            ]
            response_tokens_session = [task_tokens] + probe_token_counts
            response_tokens_session = [t for t in response_tokens_session if t >= 0]

            session_results.append({
                "session": session_idx,
                "domain": session_data["domain"],
                "is_cross_reference": session_data.get("is_cross_reference", False),
                "task_score": task_eval["task_score"],
                "recall_compression": recall_compression,
                "recall_rate": round(overall_recall, 4),
                "recall_rate_all": recall_all,
                "recall_rate_excl_binding": recall_excl_binding,
                "mechanics_failure_rate": (
                    round(n_mech_fail / len(probe_results), 4)
                    if probe_results else None
                ),
                "n_mechanics_failures": n_mech_fail,
                "recall_matrix_row": matrix_row if probe_results else {},
                "n_probes_total": len(probe_results),
                "n_probes_recalled": sum(p["recalled"] for p in probe_results),
                "probe_details": probe_results,
                "interference_probes": interference_probe_results,
                "task_keywords_found": task_eval["keywords_found"],
                "task_keywords_missing": task_eval["keywords_missing"],
                "task_outputs_text": task_outputs_text,
                "response_tokens_task": task_tokens,
                "response_tokens_probes": probe_token_counts,
                "response_tokens_max": (
                    max(response_tokens_session) if response_tokens_session else 0
                ),
                # ---- Unified-framework additions ----
                "turn_stats": combined_turn_stats,
                "loop_utilization": loop_util,
                "task_exhausted": task_exhausted,
            })

            self.tracer.log(
                "session_scored",
                parent_span_id=session_span,
                session=session_idx,
                task_score=task_eval["task_score"],
                recall_rate=round(overall_recall, 4),
                n_probes=len(probe_results),
                n_recalled=sum(p["recalled"] for p in probe_results),
                recall_matrix_row=matrix_row if probe_results else {},
                t_writes=getattr(self.memory_policy, "n_writes", 0),
                loop_utilization=loop_util,
            )

            xref_tag = " [XREF]" if session_data.get("is_cross_reference") else ""
            print(
                f"  [S6/unified] Session {session_idx:2d} "
                f"({session_data['domain']:15s}){xref_tag}  "
                f"task={task_eval['task_score']:.3f}  "
                f"recall={overall_recall:.3f}  "
                f"({sum(p['recalled'] for p in probe_results)}"
                f"/{len(probe_results)} probes)  "
                f"turns_med={loop_util['turns_median']} "
                f"tools={loop_util['tool_calls_total']}"
            )

            # ---- Write interaction to memory ----
            interaction_text = self._build_interaction_text(
                session_idx, session_data, task_output
            )
            if not is_no_memory:
                _progress(f"session {session_idx + 1}: memory write start", session_t0)
                is_growing = (
                    type(self.memory_policy).__name__ == "GrowingHistoryStorePolicy"
                )
                if is_growing or isinstance(self.memory_policy, AppendOnlyPolicy):
                    self.memory_policy.write(interaction_text, llm=self.llm)
                else:
                    # summarize_store and similar: write() internally concats with
                    # prior memory, so pass only the new content (post-fix from
                    # the runner-policy double-write audit, May 2026).
                    self.memory_policy.write(interaction_text, llm=self.llm)

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
                    input_preview=interaction_text,
                    output_preview=compressed or "",
                    thought=getattr(self.llm, "last_thought", ""),
                    cycle=session_idx,
                )
                _progress(
                    f"session {session_idx + 1}: memory write done "
                    f"(in_tok={in_tok}, out_tok={out_tok})",
                    session_t0,
                )

            self.tracer.log(
                "session_end", parent_span_id=session_span, session=session_idx,
            )

        # ---- Build aging curves ----
        task_curve = AgingCurve(
            exposures=exposures, scores=task_scores,
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        recall_curve = AgingCurve(
            exposures=recall_exposures, scores=recall_scores,
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        lag_curves = self._compute_lag_curves(recall_matrix)

        self.tracer.log(
            "run_end",
            parent_span_id=run_span,
            task_curve=list(zip(exposures, task_scores)),
            recall_curve=list(zip(recall_exposures, recall_scores)),
            recall_matrix={str(k): v for k, v in recall_matrix.items()},
            half_life=compute_half_life(recall_curve),
            slope=round(compute_decay_slope(recall_curve), 5),
        )
        _progress(
            f"unified run complete: "
            f"m_final={recall_curve.scores[-1] if recall_curve.scores else 0:.3f}"
        )

        # ---- Aggregate loop_utilization across sessions for run-level summary ----
        per_sess_loop = [sr.get("loop_utilization", {}) for sr in session_results]
        run_loop_summary = {
            "turns_median_overall": int(statistics.median(
                [lu.get("turns_median", 0) for lu in per_sess_loop]
            )) if per_sess_loop else 0,
            "tool_calls_per_session": (
                sum(lu.get("tool_calls_total", 0) for lu in per_sess_loop)
                / max(len(per_sess_loop), 1)
            ),
            "exhausted_session_share": (
                sum(1 for sr in session_results if sr.get("task_exhausted"))
                / max(len(session_results), 1)
            ),
        }

        return {
            "task_curve": task_curve,
            "recall_curve": recall_curve,
            "task_raw": list(zip(exposures, task_scores)),
            "recall_raw": list(zip(recall_exposures, recall_scores)),
            "recall_matrix": recall_matrix,
            "lag_curves": lag_curves,
            "session_results": session_results,
            # Loop telemetry as a first-class run-level field. If this number
            # stays at turns_median_overall=1 and tool_calls_per_session≈0,
            # the unified design didn't actually fire the loop and we have a
            # bug in the scenario kit (no useful tools registered).
            "loop_utilization": run_loop_summary,
        }
