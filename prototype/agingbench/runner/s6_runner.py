"""
agingbench/runner/s6_runner.py — S6 Naturalistic Aging Runner.

Implements Track B: naturalistic aging on WebArena-derived multi-domain
workflows.  Memory carryover is rational — later sessions reference facts
from earlier ones.  The aging signal is measured via a recall matrix.

Session loop
------------
For each session 0..14:
  1. Read M_t (memory from past sessions).
  2. Build task prompt (environment_data + task question).
  3. Run primary task via ReferenceAgent → task_output.
  4. Score primary task (keyword match against reference_answer).
  5. Collect ALL recall probes from sessions 0..current_session.
  6. Run each recall probe via agent → probe_output.
  7. Score recall probes (keyword match) and build recall_matrix row.
  8. Write session interaction to memory: M_{t+1} = U(M_t, H_t).

Output metrics
--------------
  - task_m(t): primary task accuracy at each session.
  - recall_m(t): mean recall rate across all probes at session t.
  - recall_matrix[t][s]: recall of session s's facts at evaluation time t.
  - fresh_recall(t): recall of current session's own probes (just learned).
  - aged_recall(t, lag): recall of facts from `lag` sessions ago.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .base import BaseRunner, RunResult
from .trace import TraceLogger
from .diagnostic_mixin import DiagnosticMixin
from ..metrics.aging import AgingCurve, compute_half_life, compute_decay_slope
from ..core.memory.base import MemoryPolicy
from ..core.memory.append_only import AppendOnlyPolicy
from ..core.agent import AgentInterface, ReferenceAgent
from ..core.tools import ToolSpec, ToolRegistry
from ..scenarios.s6_naturalistic.validator import (
    load_session_tasks,
    load_system_prompt,
    score_task,
    score_recall_probe,
    build_recall_matrix_entry,
    partition_recall,
)


def _build_tool_registry(memory_reader) -> ToolRegistry:
    """Tool registry for S6: shared ``read_memory`` only."""
    from ..core.tool_helpers import build_default_tool_registry
    return build_default_tool_registry(memory_reader)


class S6Runner(BaseRunner, DiagnosticMixin):
    """
    ScenarioRunner for S6 — Naturalistic Aging (WebArena-derived).

    Measures longitudinal recall degradation across multi-domain workflows
    where memory carryover is rational and tasks are from a real benchmark.

    Supports P1/P2/P3 diagnostic error partitioning (§5.2) via the
    ``diagnose`` flag.  When active, recall probes are evaluated under
    all three conditions (baseline, oracle retrieval, oracle context)
    in a single run.
    """

    SCENARIO_ID = "s6_naturalistic"

    def __init__(
        self,
        memory_policy: MemoryPolicy,
        llm,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        diagnose: bool = False,
        agent_class: type[AgentInterface] = ReferenceAgent,
        generated_data: dict | None = None,
        maintenance_events: list | None = None,
    ):
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        if self.llm is not None:
            self.llm.tracer = self.tracer
        self.sut_id = sut_id
        self.diagnose = diagnose
        self.agent_class = agent_class
        self.maintenance_events = maintenance_events or []

        # Load scenario data (from generator or curated files)
        if generated_data:
            self.sessions = generated_data["session_tasks"]["sessions"]
            self.system_prompt_base = generated_data["session_tasks"]["system_prompt"]
        else:
            self.sessions = load_session_tasks()
            self.system_prompt_base = load_system_prompt()

        # Model info for tracing
        self._model_id = (
            getattr(llm, "model_id", None)
            or getattr(llm, "model", "unknown")
        )
        self._provider = "local_hf" if hasattr(llm, "tok") else "litellm"

    def run(self, n_sessions: int = 15, seed: int = 42) -> dict:
        """
        Run the S6 naturalistic aging loop.

        Returns dict with:
          - task_curve: AgingCurve for task_m(t)
          - recall_curve: AgingCurve for recall_m(t)
          - recall_matrix: {eval_time: {origin_session: recall_rate}}
          - session_results: per-session scoring details
        """
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
                print(f"  [S6][progress][run {run_elapsed}] {msg}", flush=True)
                return
            session_elapsed = _fmt_elapsed(session_start)
            print(
                f"  [S6][progress][run {run_elapsed} | session {session_elapsed}] {msg}",
                flush=True,
            )

        exposures: list[int] = []
        task_scores: list[float] = []
        recall_scores: list[float] = []
        # Recall (compression) curve gets its own exposure axis: sessions with
        # no measurable stable-fact pool are skipped, not counted as 1.0.
        recall_exposures: list[int] = []
        recall_matrix: dict[int, dict[int, float]] = {}
        session_results: list[dict] = []

        # P1/P2/P3 diagnostic partitioning accumulators (only when diagnose=True)
        diagnostic_partitions: list[dict] = []

        run_span = self.tracer.log(
            "run_start",
            parent_span_id=None,
            sut_id=self.sut_id,
            scenario=self.SCENARIO_ID,
            seed=seed,
            n_sessions=n_sessions,
            policy=type(self.memory_policy).__name__,
            diagnose=self.diagnose,
            **{"gen_ai.request.model": self._model_id},
        )

        actual_sessions = min(n_sessions, len(self.sessions))
        _progress(
            f"starting run: sessions={actual_sessions}, policy={type(self.memory_policy).__name__}, "
            f"diagnose={self.diagnose}"
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

            # ---- Apply maintenance events scheduled for this session ----
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

            # ---- Build agent with native memory (P1 baseline) ----
            def _memory_reader():
                if is_no_memory:
                    return ""
                return self.memory_policy.read() or ""

            tool_registry = _build_tool_registry(_memory_reader)
            agent = self.agent_class(
                llm=self.llm,
                memory_policy=self.memory_policy,
                tools=tool_registry,
                max_turns=8,
            )
            # S6 ReferenceAgent uses the same legacy REACT_SYSTEM as S1-S4 for
            # cross-scenario consistency on the headline benchmark — no
            # scenario-specific persona is injected here. The
            # ``self.system_prompt_base`` field is still loaded by __init__
            # (it remains available to callers that want it via
            # ``runner.system_prompt_base``) but is intentionally NOT applied
            # to the agent's system prompt; doing so would diverge S6 from
            # S1-S4 framing.

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
                # Cross-reference session — no new data
                full_task = task_text

            # Log memory state before task
            memory_before = _memory_reader()
            # ---- Run primary task ----
            _progress(f"session {session_idx + 1}: primary task start", session_t0)
            task_result = agent.run_session(full_task, session_id=session_idx)
            task_output = task_result["output"]
            _progress(
                f"session {session_idx + 1}: primary task done "
                f"(turns={task_result.get('turns', 0)}, tools={len(task_result.get('tool_calls', []))})",
                session_t0,
            )

            # Log agent output
            # Score primary task
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

            # ---- Run recall probes for PAST sessions (0..t-1) ----
            # We only probe past sessions because the current session's data
            # hasn't been written to memory yet (the agent is stateless across
            # run_session calls).  This means the minimum meaningful lag is 1.
            all_probes = []
            for s in range(session_idx):
                for probe in self.sessions[s].get("recall_probes", []):
                    # Skip probes whose target fact was invalidated at or before
                    # this session: a retracted fact is no longer a valid recall
                    # target, so it must leave the recall_rate pool (otherwise
                    # the headline rewards citing the retracted value). It is
                    # still scored in the sessions before the retraction.
                    inv_at = probe.get("invalidated_at_session")
                    if inv_at is not None and session_idx >= inv_at:
                        continue
                    all_probes.append(probe)

            probe_results = []
            # Forced binding probes (confusable names): capture response with
            # gold + distractor so score_interference_binding can classify
            # correct/confused/miss. S6 re-asks prior probes each session, so
            # this yields a lag/density gradient automatically.
            interference_probe_results = []
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
                # at_session = current eval time, so a probe re-asked before a
                # revision is scored against the pre-revision value and one
                # re-asked after against the post-revision value (time-correct).
                scored = score_recall_probe(probe_result["output"], probe, at_session=session_idx)
                # Mechanics failure: agent exhausted its turn budget or returned
                # no usable text — an availability failure, NOT a recall outcome;
                # excluded from recall + binding below.
                _mech_fail = (bool(probe_result.get("exhausted"))
                              or not (probe_result.get("output") or "").strip())
                scored["mechanics_failure"] = _mech_fail
                # Revised facts carry keywords_history -> they belong to the
                # revision axis (version_accuracy), not the compression headline.
                scored["is_revised"] = bool(probe.get("keywords_history"))
                # Carry the raw probe answer so forget_accuracy's text scan and
                # the per-probe token diagnostics below see it.
                scored["agent_answer"] = probe_result["output"]
                # Tag the probe type so the recall headline can be sliced with
                # vs without the confusable subset (comparability across configs).
                scored["probe_type"] = probe.get("probe_type")
                probe_results.append(scored)
                # Forced binding probes (confusable names) also count toward the
                # recall headline (a "wrong John" answer is a recall failure),
                # AND are decomposed here into confusion vs omission for
                # score_interference_binding. Same probe, two views; S6 re-asks
                # them every session so the per-session series is the density
                # gradient.
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

            # ---- P2/P3 diagnostic probes (when --diagnose is active) ----
            if self.diagnose and all_probes:
                _progress(
                    f"session {session_idx + 1}: diagnostic P2/P3 probes start ({total_probes})",
                    session_t0,
                )
                diag_result = self.run_diagnostic_probes(
                    probes=all_probes,
                    session_idx=session_idx,
                    agent=agent,
                    score_fn=score_recall_probe,
                    gold_facts=self._build_gold_facts(session_idx),
                    p1_results=probe_results,  # reuse P1 scores already computed
                )
                diagnostic_partitions.append(diag_result["partition"])
                _progress(
                    f"session {session_idx + 1}: diagnostic done "
                    f"P1={diag_result['partition']['acc_p1']:.3f} "
                    f"P2={diag_result['partition']['acc_p2']:.3f} "
                    f"P3={diag_result['partition']['acc_p3']:.3f}",
                    session_t0,
                )

            # Build recall matrix row + the de-blended recall metrics. The
            # four-mechanism partition (compression / interference / revision +
            # mechanics-failure exclusion) lives in validator.partition_recall
            # so it is unit-testable; see that helper for the rationale.
            matrix_row = {}
            _rp = partition_recall(probe_results)
            recall_all = _rp["recall_all"]
            recall_excl_binding = _rp["recall_excl_binding"]
            recall_compression = _rp["recall_compression"]   # HEADLINE
            n_mech_fail = _rp["n_mechanics_failures"]
            if probe_results:
                matrix_row = build_recall_matrix_entry(
                    session_idx, self.sessions, probe_results
                )
                recall_matrix[session_idx] = matrix_row
            else:
                recall_matrix[session_idx] = {}

            # Float display value for logs/print (0.0 when no measurable pool).
            overall_recall = recall_all if recall_all is not None else 0.0

            # Record scores. task_curve spans every session; the recall
            # (compression) curve only includes sessions with a measurable
            # STABLE-fact pool — an empty pool is NOT scored as perfect recall.
            exposures.append(session_idx)
            task_scores.append(task_eval["task_score"])
            if recall_compression is not None:
                recall_exposures.append(session_idx)
                recall_scores.append(recall_compression)

            # Concatenate task + probe outputs (truncated) for forget_accuracy.
            probe_text = " ".join(
                str(p.get("agent_answer", ""))[:300] for p in probe_results
            )
            task_outputs_text = (str(task_output)[:1000] + " " + probe_text).strip()

            # Token-cap diagnostics: track max response size per session so
            # post-hoc analysis can detect whether aging is confounded by
            # max_new_tokens truncation.
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
                "is_cross_reference": session_data.get(
                    "is_cross_reference", False
                ),
                "task_score": task_eval["task_score"],
                # HEADLINE compression metric: stable-fact recall (binding +
                # revised probes excluded; mechanics failures excluded).
                "recall_compression": recall_compression,
                # All-probe live recall (kept for back-compat / blended view).
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
                "n_probes_recalled": sum(
                    p["recalled"] for p in probe_results
                ),
                "probe_details": probe_results,
                "interference_probes": interference_probe_results,
                "task_keywords_found": task_eval["keywords_found"],
                "task_keywords_missing": task_eval["keywords_missing"],
                "task_outputs_text": task_outputs_text,
                "response_tokens_task": task_tokens,
                "response_tokens_probes": probe_token_counts,
                "response_tokens_max": (max(response_tokens_session)
                                       if response_tokens_session else 0),
            })

            self.tracer.log(
                "session_scored",
                parent_span_id=session_span,
                session=session_idx,
                task_score=task_eval["task_score"],
                recall_rate=round(overall_recall, 4),
                n_probes=len(probe_results),
                n_recalled=sum(p["recalled"] for p in probe_results),
                recall_matrix_row=(
                    matrix_row if probe_results else {}
                ),
                t_writes=getattr(self.memory_policy, "n_writes", 0),
            )

            # Print progress
            xref_tag = " [XREF]" if session_data.get(
                "is_cross_reference"
            ) else ""
            diag_tag = ""
            if self.diagnose and diagnostic_partitions:
                dp = diagnostic_partitions[-1]
                diag_tag = (
                    f"  W={dp['write_error']:.3f} R={dp['read_error']:.3f} "
                    f"U={dp['utilization_error']:.3f}"
                )
            print(
                f"  [S6] Session {session_idx:2d} ({session_data['domain']:15s})"
                f"{xref_tag}  "
                f"task={task_eval['task_score']:.3f}  "
                f"recall={overall_recall:.3f}  "
                f"({sum(p['recalled'] for p in probe_results)}"
                f"/{len(probe_results)} probes)"
                f"{diag_tag}"
            )

            # ---- Write interaction to memory ----
            interaction_text = self._build_interaction_text(
                session_idx, session_data, task_output
            )

            # Always write via native policy (P1 baseline path).
            if not is_no_memory:
                _progress(f"session {session_idx + 1}: memory write start", session_t0)
                is_growing = type(self.memory_policy).__name__ == "GrowingHistoryStorePolicy"
                if is_growing or isinstance(self.memory_policy, AppendOnlyPolicy):
                    self.memory_policy.write(interaction_text, llm=self.llm)
                else:
                    # summarize_store and similar: write() internally concats
                    # with prior memory, so pass only the new content.
                    self.memory_policy.write(interaction_text, llm=self.llm)

                compressed = self.memory_policy.read()
                in_tok = getattr(self.memory_policy, "last_input_tokens", 0)
                out_tok = getattr(
                    self.memory_policy, "last_output_tokens", 0
                )
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
                    f"session {session_idx + 1}: memory write done (in_tok={in_tok}, out_tok={out_tok})",
                    session_t0,
                )

            self.tracer.log(
                "session_end",
                parent_span_id=session_span,
                session=session_idx,
            )
            _progress(
                f"session {session_idx + 1}/{actual_sessions} end: "
                f"task={task_eval['task_score']:.3f}, recall={overall_recall:.3f}",
                session_t0,
            )

        # ---- Build aging curves ----
        task_curve = AgingCurve(
            exposures=exposures,
            scores=task_scores,
            scenario=self.SCENARIO_ID,
            sut_id=self.sut_id,
        )
        # recall_scores is sparse: only sessions with a measurable STABLE-fact
        # pool contribute, so it pairs with recall_exposures (NOT exposures).
        recall_curve = AgingCurve(
            exposures=recall_exposures,
            scores=recall_scores,
            scenario=self.SCENARIO_ID,
            sut_id=self.sut_id,
        )

        # ---- Compute lag-based recall curves ----
        lag_curves = self._compute_lag_curves(recall_matrix)

        self.tracer.log(
            "run_end",
            parent_span_id=run_span,
            task_curve=list(zip(exposures, task_scores)),
            recall_curve=list(zip(recall_exposures, recall_scores)),
            recall_matrix={
                str(k): v for k, v in recall_matrix.items()
            },
            half_life=compute_half_life(recall_curve),
            slope=round(compute_decay_slope(recall_curve), 5),
        )

        # Close trajectory log
        _progress(f"run complete: m_final={recall_curve.scores[-1] if recall_curve.scores else 0:.3f}")

        result = {
            "task_curve": task_curve,
            "recall_curve": recall_curve,
            "task_raw": list(zip(exposures, task_scores)),
            "recall_raw": list(zip(recall_exposures, recall_scores)),
            "recall_matrix": recall_matrix,
            "lag_curves": lag_curves,
            "session_results": session_results,
        }

        # Compute and attach full error partition when --diagnose was used.
        if self.diagnose and diagnostic_partitions:
            from ..diagnostics.partitioner import partition_errors
            p1_scores = {d["session"]: d["acc_p1"] for d in diagnostic_partitions}
            p2_scores = {d["session"]: d["acc_p2"] for d in diagnostic_partitions}
            p3_scores = {d["session"]: d["acc_p3"] for d in diagnostic_partitions}
            n_probes = {d["session"]: d["n_probes"] for d in diagnostic_partitions}
            result["diagnostic_partition"] = partition_errors(
                p1_scores, p2_scores, p3_scores, n_probes,
            )
            result["diagnostic_per_session"] = diagnostic_partitions

        return result

    def _build_gold_facts(self, up_to_session: int) -> str:
        """Build ground-truth fact sheet from all sessions before `up_to_session`.

        Used by P3 (oracle context): the LLM receives these facts directly,
        establishing the absolute reasoning ceiling of the utilization logic U.
        """
        lines = []
        for s in range(up_to_session):
            if s >= len(self.sessions):
                break
            sd = self.sessions[s]
            if sd.get("is_cross_reference"):
                continue
            ref_answer = sd.get("task", {}).get("reference_answer", "")
            if ref_answer:
                lines.append(f"Session {s} ({sd['domain']}): {ref_answer}")
            for probe in sd.get("recall_probes", []):
                canonical = probe.get("canonical_answer", "")
                if canonical:
                    lines.append(f"  Fact: {canonical}")
        return "\n".join(lines) if lines else ""

    def _build_interaction_text(
        self,
        session_idx: int,
        session_data: dict,
        task_output: str,
    ) -> str:
        """Build a text summary of the session for memory write.

        No runner-level truncation — the memory policy's compression handles
        information selection. Pre-truncation would destroy information before
        the policy ever sees it, creating artificial aging artifacts.
        """
        lines = [
            f"--- Research Session {session_idx} "
            f"({session_data['domain']}) ---",
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
        """
        Compute recall-by-lag curves from the recall matrix.

        For each lag value (0 = just learned, 1 = one session ago, etc.),
        produces a time series of recall rates.

        Returns {lag: [(eval_time, recall_rate), ...]}
        """
        lag_curves: dict[int, list[tuple[int, float]]] = {}

        for eval_time, row in sorted(recall_matrix.items()):
            for origin, rate in row.items():
                lag = eval_time - origin
                lag_curves.setdefault(lag, []).append((eval_time, rate))

        return lag_curves
