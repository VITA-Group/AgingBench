"""
agingbench/runner/s3_runner.py — S3 Project Knowledge Base Agent.

Implements the 12-session state machine for the Knowledge Base scenario.
Each session: agent reads transcript + M_{t-1}, answers 3 team queries,
writes interaction history to memory.

Produces G3 metrics: summarization_fidelity, memory_bloat, contradiction_rate.
Also tracks query_accuracy as a G1-aligned task performance signal.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .base import BaseRunner, RunResult
from .trace import TraceLogger
from ..metrics.aging import AgingCurve
from ..metrics.g3_metrics import compute_memory_bloat
from ..core.memory.base import MemoryPolicy
from ..core.memory.eval_proxy import EvalTextMemoryProxy
from ..core.agent import AgentInterface, ReferenceAgent
from ..core.tools import ToolRegistry, ToolSpec


class S3Runner(BaseRunner):
    SCENARIO_ID = "s3_knowledge_base"

    def __init__(
        self,
        memory_policy: MemoryPolicy,
        llm,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        oracle_mode: bool = False,
        oracle_retrieval: bool = False,
        agent_class: type[AgentInterface] = ReferenceAgent,
        generated_data: dict | None = None,
    ):
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        if self.llm is not None:
            self.llm.tracer = self.tracer
        self.sut_id = sut_id
        self.oracle_mode = oracle_mode
        self.oracle_retrieval = oracle_retrieval
        self.agent_class = agent_class

        self._model_id = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
        self._provider = "local_hf" if hasattr(llm, "tok") else "litellm"

        # Load scenario data (from generator or curated files)
        if generated_data:
            self.transcripts = generated_data["transcripts"]["sessions"]
            self.gold_decisions = generated_data["gold_timeline"]["decisions"]
            self.queries_by_session = generated_data["queries"]["sessions"]
            # Forced-choice interference binding probes (confusable pairs). The
            # generator emits one per injected pair, on a key separate from the
            # session queries so the `fidelity` headline + query path are
            # untouched. The runner asks them at a few post-injection lags and
            # feeds score_interference_binding. Absent for older/curated data.
            interf = generated_data.get("interference_probes", {})
            self.binding_probes_by_session = interf.get("sessions", [])
            self.binding_probe_lags = interf.get("probe_lags", [1, 3]) or [1, 3]
        else:
            data_dir = Path(__file__).parent.parent / "scenarios" / "s3_knowledge_base"
            with open(data_dir / "transcripts.json") as f:
                self.transcripts = json.load(f)["sessions"]
            with open(data_dir / "gold_timeline.json") as f:
                self.gold_decisions = json.load(f)["decisions"]
            with open(data_dir / "queries.json") as f:
                self.queries_by_session = json.load(f)["sessions"]
            # Curated S3 data has no interference binding probes.
            self.binding_probes_by_session = []
            self.binding_probe_lags = [1, 3]

    def _get_decisions_up_to(self, session: int) -> list[dict]:
        """Get all gold decisions introduced up to and including this session."""
        return [d for d in self.gold_decisions if d["session"] <= session]

    def _binding_probes_due_at(self, t: int, n_sessions: int) -> list[dict]:
        """Return interference binding probes to ask at session ``t``.

        Each pair's probe was emitted at its injection session ``s``; we re-ask
        it at ``s + lag`` for each configured lag (default [1, 3]). This bounds
        cost (no re-asking every probe every session) while guaranteeing at
        least one scored probe per injected pair: if a pair would never be
        reached because all of its lagged sessions fall past the run horizon,
        it is asked at the final session as a fallback so n_probes > 0.
        """
        due: list[dict] = []
        for entry in self.binding_probes_by_session:
            s = entry.get("session", -1)
            probes = entry.get("probes", []) or []
            if not probes:
                continue
            target_sessions = sorted({
                s + lag for lag in self.binding_probe_lags if lag >= 0
            })
            reachable = [ts for ts in target_sessions if ts < n_sessions]
            if reachable:
                if t in reachable:
                    due.extend(probes)
            else:
                # No lag lands inside the horizon — fall back to the last
                # session so each injected pair is scored at least once.
                if t == n_sessions - 1:
                    due.extend(probes)
        return due

    def _build_tools(self, memory_text: str) -> ToolRegistry:
        """Tool registry: shared ``read_memory`` only (no scenario-specific tools)."""
        from ..core.tool_helpers import build_default_tool_registry
        return build_default_tool_registry(lambda: memory_text)

    def run(self, n_sessions: int = 12, seed: int = 42) -> dict:
        """Run S3 for n_sessions; return dict with aging curves and session results."""
        from ..scenarios.s3_knowledge_base.validator import (
            score_query, compute_fidelity_detailed, compute_contradiction_rate,
            compute_contradiction_count, score_revision_aging,
            _present as _kw_boundary_present,
        )

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
        progress_on = os.getenv("AGINGBENCH_S3_PROGRESS", "1").lower() not in {
            "0", "false", "no", "off"
        }
        query_log_every = max(1, int(os.getenv("AGINGBENCH_S3_QUERY_LOG_EVERY", "1")))

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
                print(f"  [S3][progress][run {run_elapsed}] {msg}", flush=True)
                return
            session_elapsed = _fmt_elapsed(session_start)
            print(
                f"  [S3][progress][run {run_elapsed} | session {session_elapsed}] {msg}",
                flush=True,
            )

        # Accumulate all transcripts so far for oracle mode
        all_transcripts_text = ""

        is_append_only = type(self.memory_policy).__name__ == "AppendOnlyPolicy"

        fidelity_raw = []
        bloat_raw = []
        contradiction_raw = []
        query_acc_raw = []
        retrieval_precision_raw = []
        retrieval_recall_raw = []
        # Revision-aging trident raw series (count form for aging curves;
        # the rate form lives in session_results for per-session readability).
        rev_excess_count_raw = []
        stale_residue_count_raw = []
        session_results = []

        # ---- Full trajectory log (content-level, separate from trace) ----
        trajectory_path = self.tracer.path.parent / "trajectory.jsonl"
        traj_f = open(trajectory_path, "w", buffering=1)
        def _log_traj(event_type: str, **fields):
            import json as _json
            record = {"event": event_type, "timestamp": time.time(), **fields}
            traj_f.write(_json.dumps(record, ensure_ascii=False) + "\n")

        run_span = self.tracer.log(
            "run_start", parent_span_id=None,
            sut_id=self.sut_id, scenario=self.SCENARIO_ID,
            seed=seed, n_sessions=n_sessions,
            policy=type(self.memory_policy).__name__,
            oracle_mode=self.oracle_mode,
        )

        actual_sessions = min(n_sessions, len(self.transcripts))
        _progress(
            f"starting run: sessions={actual_sessions}, policy={type(self.memory_policy).__name__}, "
            f"oracle={self.oracle_mode}"
        )

        for t in range(actual_sessions):
            session_t0 = time.time()
            sess_span = self.tracer.log(
                "session_start", parent_span_id=run_span, session=t,
            )

            transcript = self.transcripts[t]
            transcript_text = transcript["transcript"]
            all_transcripts_text += f"\n\n--- Session {t}: {transcript['title']} ---\n{transcript_text}"

            # Read memory.
            # C0 (no_memory)        : empty
            # C3 (oracle_mode)      : full concatenated transcripts (agent skips writes below)
            # C2 (oracle_retrieval) : agent writes compressed memory below, but
            #                         READS full transcripts — isolates retrieval.
            # C1 (baseline)         : compressed memory via memory_policy
            if t == 0 or is_no_memory:
                memory_text = ""
            elif self.oracle_mode:
                memory_text = all_transcripts_text
            elif self.oracle_retrieval:
                memory_text = all_transcripts_text
            else:
                memory_text = self.memory_policy.read()

            # Build agent with search_memory tool
            tools = self._build_tools(memory_text)
            # C1-C4 control: system-prompt memory must match the mode-dependent
            # memory_text (also injected into the user context below), not the
            # raw SUT policy. In C1 memory_text == policy.read() → no-op for the
            # baseline; C2 (oracle_retrieval) / C3 (oracle_mode) are corrected.
            agent = self.agent_class(
                llm=self.llm,
                memory_policy=EvalTextMemoryProxy(self.memory_policy, memory_text),
                tools=tools,
                max_turns=4,
            )

            # Agent context: memory + new transcript
            context = ""
            if memory_text:
                context += f"Project Knowledge Base (accumulated decisions):\n{memory_text}\n\n"
            context += f"New Meeting Transcript — {transcript['title']}:\n{transcript_text}\n\n"

            # Log memory state before queries
            _log_traj("memory_snapshot", session=t, phase="before_task",
                      memory_text=memory_text, memory_tokens=len(memory_text.split()) if memory_text else 0)

            # Answer queries
            queries = self.queries_by_session[t]["queries"] if t < len(self.queries_by_session) else []
            _progress(
                f"session {t + 1}/{actual_sessions} start: queries={len(queries)}",
                session_t0,
            )
            query_responses = []
            query_scores = []

            for q_i, q in enumerate(queries, start=1):
                if q_i == 1 or q_i % query_log_every == 0 or q_i == len(queries):
                    _progress(
                        f"session {t + 1}: running query {q_i}/{len(queries)} ({q['query_id']})",
                        session_t0,
                    )
                prompt = (
                    f"{context}"
                    f"A team member asks: {q['question']}\n\n"
                    f"Search your knowledge base if needed, then answer with specific details "
                    f"(names, dates, dollar amounts, version numbers)."
                )
                result = agent.run_session(prompt, session_id=t)
                response = result["output"]
                query_responses.append(response)

                sc = score_query(response, q)
                query_scores.append(sc)

                _log_traj("agent_output", session=t, phase="query",
                          query_id=q["query_id"],
                          prompt=prompt[:500], output=response[:500],
                          score=sc, turns=result.get("turns", 0))

                self.tracer.log(
                    "query_answered", parent_span_id=sess_span,
                    session=t, query_id=q["query_id"],
                    score=sc, turns=result["turns"],
                )

                if q_i == 1 or q_i % query_log_every == 0 or q_i == len(queries):
                    _progress(
                        f"session {t + 1}: finished query {q_i}/{len(queries)} "
                        f"(turns={result.get('turns', 0)})",
                        session_t0,
                    )

            query_acc = sum(query_scores) / len(query_scores) if query_scores else 0.0

            # ---- Forced-choice interference binding probes ----
            # Ask each injected confusable pair's binding probe at a few
            # post-injection lags (default [1, 3]). These are SEPARATE from the
            # session queries (they do not touch `fidelity` or `query_accuracy`)
            # and populate session_result["interference_probes"] with the schema
            # score_interference_binding expects. Bounded cost: a pair is asked
            # only on its lagged sessions, not every session.
            interference_probe_results = []
            due_probes = self._binding_probes_due_at(t, actual_sessions)
            if due_probes:
                _progress(
                    f"session {t + 1}: interference binding probes start "
                    f"({len(due_probes)})",
                    session_t0,
                )
            for bp in due_probes:
                bp_prompt = (
                    f"{context}"
                    f"A team member asks: {bp['question']}\n\n"
                    f"Search your knowledge base if needed, then answer with the "
                    f"exact value only."
                )
                bp_result = agent.run_session(bp_prompt, session_id=t)
                bp_response = bp_result["output"]
                interference_probe_results.append({
                    "session": t,
                    "task_id": bp.get("probe_id"),
                    "question": bp["question"],
                    "response_text": bp_response,
                    "gold_value": bp.get("gold_value"),
                    "distractor_value": bp.get("distractor_value"),
                })
                _log_traj("interference_probe", session=t,
                          probe_id=bp.get("probe_id", ""),
                          question=bp["question"],
                          agent_answer=bp_response[:300],
                          gold_value=bp.get("gold_value"),
                          distractor_value=bp.get("distractor_value"))

            # Write interaction history to memory
            interaction = f"Session {t}: {transcript['title']}\n{transcript_text}\n"
            for q, r in zip(queries, query_responses):
                interaction += f"Q: {q['question']}\nA: {r}\n"

            out_tok = -1
            if not is_no_memory and not self.oracle_mode:
                _progress(f"session {t + 1}: memory write start", session_t0)
                self.memory_policy.write(interaction, llm=self.llm)

                compressed = self.memory_policy.read()
                in_tok = getattr(self.memory_policy, "last_input_tokens", 0)
                out_tok = getattr(self.memory_policy, "last_output_tokens", 0)

                _log_traj("compression", session=t,
                          input_text=interaction,
                          output_text=compressed or "",
                          input_tokens=in_tok, output_tokens=out_tok,
                          compression_ratio=round(
                              len(interaction.split()) / max(len((compressed or "").split()), 1), 2
                          ))

                self.tracer.log_llm_call(
                    parent_span_id=sess_span,
                    model=self._model_id, provider=self._provider,
                    input_tokens=in_tok, output_tokens=out_tok,
                    input_preview=interaction,
                    output_preview=compressed or "",
                    thought=getattr(self.llm, "last_thought", ""),
                    cycle=t,
                )
                _progress(
                    f"session {t + 1}: memory write done (in_tok={in_tok}, out_tok={out_tok})",
                    session_t0,
                )

            # Score G3 metrics against gold timeline
            current_memory = self.memory_policy.read() if not is_no_memory else all_transcripts_text
            decisions_so_far = self._get_decisions_up_to(t)

            fidelity_detail = compute_fidelity_detailed(
                current_memory, decisions_so_far, at_session=t
            )
            fidelity = fidelity_detail["fidelity"]
            contradiction = compute_contradiction_rate(current_memory, decisions_so_far)
            contradiction_count = compute_contradiction_count(current_memory, decisions_so_far)
            # Revision-aging trident: paired excess (cancels compression
            # baseline) + stale residue (compression-immune signal). Returned
            # dict carries rates + counts + coverage verdict.
            rev_aging = score_revision_aging(
                current_memory, decisions_so_far, at_session=t
            )
            bloat = compute_memory_bloat(current_memory)

            # Retrieval metrics (AppendOnly track: G3-M4, G3-M5).
            # These only apply to policies that expose a retriever over
            # chunked memory (e.g. AppendOnly). For monolithic-compressed
            # policies (growing_history, summarize_store, etc.) there is no
            # ranked retrieval to score — we emit None so downstream analysis
            # distinguishes N/A from a genuine 0.0.
            ret_precision: float | None = None
            ret_recall: float | None = None
            # AppendOnlyPolicy exposes its store as `.retriever` (not
            # `_retriever`); the attribute check must match.
            if is_append_only and hasattr(self.memory_policy, 'retriever'):
                ret_precision = 0.0
                ret_recall = 0.0
                from ..metrics.g3_metrics import compute_retrieval_precision, compute_retrieval_recall
                # For each query, check if retrieval returns relevant decisions
                for q in queries:
                    gold_ids = set(q.get("gold_decision_ids", []))
                    if gold_ids:
                        # Query the retriever
                        retrieved = self.memory_policy.retriever.cosine_search(
                            q["question"], top_k=5
                        ) if hasattr(self.memory_policy, 'retriever') else []
                        retrieved_texts = [text for _, text in retrieved]
                        # Check which gold decisions appear in retrieved chunks.
                        # Use the validator's digit-flank-safe presence helper
                        # so a short numeric keyword ("37") does not match
                        # inside an unrelated number ("$3700") in the chunk —
                        # same bug class fixed in s2/s6 validators.
                        retrieved_ids = set()
                        for did in gold_ids:
                            decision = next((d for d in self.gold_decisions if d["id"] == did), None)
                            if decision:
                                for rt in retrieved_texts:
                                    rt_lower = rt.lower()
                                    if any(_kw_boundary_present(kw, rt_lower)
                                           for kw in decision["keywords"]):
                                        retrieved_ids.add(did)
                                        break
                        if gold_ids:
                            ret_precision += len(retrieved_ids) / max(len(retrieved_texts), 1)
                            ret_recall += len(retrieved_ids) / len(gold_ids)

                n_queries = len(queries) if queries else 1
                ret_precision /= n_queries
                ret_recall /= n_queries

            # Only append numeric samples to the aging-curve raw series. For
            # policies without a retriever (ret_precision/recall = None) the
            # series stays empty and downstream plots/aging math skip cleanly.
            if ret_precision is not None:
                retrieval_precision_raw.append((t, ret_precision))
            if ret_recall is not None:
                retrieval_recall_raw.append((t, ret_recall))

            fidelity_raw.append((t, fidelity))
            bloat_raw.append((t, bloat))
            contradiction_raw.append((t, contradiction))
            query_acc_raw.append((t, query_acc))
            # Only seed the revision-aging curves with sessions that carry a
            # real signal — pre-first-revision sessions stay out of the series
            # so downstream slope/aging math doesn't anchor on structural zeros.
            if rev_aging["revision_fidelity_excess_count"] is not None:
                rev_excess_count_raw.append(
                    (t, rev_aging["revision_fidelity_excess_count"])
                )
            if rev_aging["coverage_verdict"] != "no_revisions":
                stale_residue_count_raw.append(
                    (t, rev_aging["stale_residue_count"])
                )

            # Concatenate query responses (truncated) so dependency_scorer's
            # forget_accuracy can scan for invalidated keywords. Without this,
            # forget_accuracy was silently saturated at 1.0 because no
            # scenario-side text was reachable from session_results.
            task_outputs_text = " ".join(
                str(r)[:500] for r in query_responses if r
            )

            # Token-cap diagnostics
            from agingbench.metrics.aging import count_response_tokens
            query_token_counts = [
                count_response_tokens(self.llm, str(r)) for r in query_responses if r
            ]
            valid_qt = [t for t in query_token_counts if t >= 0]

            sr = {
                "session": t,
                "title": transcript["title"],
                "query_accuracy": query_acc,
                "query_scores": query_scores,
                "fidelity": fidelity,
                "category_fidelity": fidelity_detail["category_fidelity"],
                "contradiction_rate": contradiction,
                "contradiction_count": contradiction_count,
                "revision_aging": rev_aging,
                "memory_bloat": bloat,
                "n_decisions_so_far": len(decisions_so_far),
                "retrieval_precision": ret_precision,
                "retrieval_recall": ret_recall,
                "task_outputs_text": task_outputs_text,
                "interference_probes": interference_probe_results,
                "response_tokens_per_query": query_token_counts,
                "response_tokens_max": max(valid_qt) if valid_qt else 0,
                "memory_write_tokens": out_tok,
            }
            session_results.append(sr)

            _log_traj("fidelity_contradiction", session=t,
                      query_accuracy=round(query_acc, 4),
                      fidelity=round(fidelity, 4),
                      contradiction_rate=round(contradiction, 4),
                      contradiction_count=contradiction_count,
                      memory_bloat=bloat,
                      n_decisions_so_far=len(decisions_so_far))

            _rev_excess_rate_logged = (
                round(rev_aging["revision_fidelity_excess"], 4)
                if rev_aging["revision_fidelity_excess"] is not None else None
            )
            _stale_rate_logged = (
                round(rev_aging["stale_residue_rate"], 4)
                if rev_aging["stale_residue_rate"] is not None else None
            )
            _log_traj("revision_aging", session=t,
                      revision_fidelity_excess=_rev_excess_rate_logged,
                      revision_fidelity_excess_count=rev_aging["revision_fidelity_excess_count"],
                      stale_residue_rate=_stale_rate_logged,
                      stale_residue_count=rev_aging["stale_residue_count"],
                      n_revised=rev_aging["n_revised"],
                      n_unrevised=rev_aging["n_unrevised"],
                      coverage_verdict=rev_aging["coverage_verdict"])

            self.tracer.log(
                "session_scored", parent_span_id=sess_span, session=t,
                query_accuracy=round(query_acc, 4),
                fidelity=round(fidelity, 4),
                contradiction_rate=round(contradiction, 4),
                memory_bloat=bloat,
            )
            self.tracer.log("session_end", parent_span_id=sess_span, session=t)
            _progress(
                f"session {t + 1}/{actual_sessions} end: fidelity={fidelity:.3f}, query_acc={query_acc:.3f}",
                session_t0,
            )

            _rev_summary = (
                f"rev_stale={rev_aging['stale_residue_count']}"
                f"({rev_aging['n_revised']}rev)"
                if rev_aging["coverage_verdict"] != "no_revisions"
                else "rev_stale=n/a"
            )
            print(f"  [S3] Session {t:2d}  fidelity={fidelity:.3f}  "
                  f"query_acc={query_acc:.3f}  contradict={contradiction:.3f}  "
                  f"{_rev_summary}  bloat={bloat}  decisions={len(decisions_so_far)}")

        # Build aging curves
        fidelity_curve = AgingCurve(
            exposures=[r[0] for r in fidelity_raw],
            scores=[r[1] for r in fidelity_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        bloat_curve = AgingCurve(
            exposures=[r[0] for r in bloat_raw],
            scores=[r[1] for r in bloat_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        contradiction_curve = AgingCurve(
            exposures=[r[0] for r in contradiction_raw],
            scores=[1.0 - r[1] for r in contradiction_raw],  # invert: 1=no contradictions
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        query_curve = AgingCurve(
            exposures=[r[0] for r in query_acc_raw],
            scores=[r[1] for r in query_acc_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )

        retrieval_precision_curve = AgingCurve(
            exposures=[r[0] for r in retrieval_precision_raw],
            scores=[r[1] for r in retrieval_precision_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        retrieval_recall_curve = AgingCurve(
            exposures=[r[0] for r in retrieval_recall_raw],
            scores=[r[1] for r in retrieval_recall_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )

        # Revision-aging count-form curves. Counts (not rates) so the curve
        # does not dilute as the revision pool grows — same discipline as
        # contradiction_count vs contradiction_rate.
        rev_excess_count_curve = AgingCurve(
            exposures=[r[0] for r in rev_excess_count_raw],
            scores=[r[1] for r in rev_excess_count_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        stale_residue_count_curve = AgingCurve(
            exposures=[r[0] for r in stale_residue_count_raw],
            scores=[r[1] for r in stale_residue_count_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )

        self.tracer.log(
            "run_end", parent_span_id=run_span,
            fidelity_curve=fidelity_raw,
            contradiction_curve=contradiction_raw,
            bloat_curve=bloat_raw,
            query_curve=query_acc_raw,
            retrieval_precision_curve=retrieval_precision_raw,
            retrieval_recall_curve=retrieval_recall_raw,
            rev_excess_count_curve=rev_excess_count_raw,
            stale_residue_count_curve=stale_residue_count_raw,
        )

        # Close trajectory log
        _log_traj("run_end", n_sessions=actual_sessions,
                  m_final=fidelity_raw[-1][1] if fidelity_raw else 0)
        traj_f.close()
        _progress(f"run complete: m_final={fidelity_raw[-1][1] if fidelity_raw else 0:.3f}")

        return {
            "fidelity_curve": fidelity_curve,
            "bloat_curve": bloat_curve,
            "contradiction_curve": contradiction_curve,
            "query_curve": query_curve,
            "retrieval_precision_curve": retrieval_precision_curve,
            "retrieval_recall_curve": retrieval_recall_curve,
            "rev_excess_count_curve": rev_excess_count_curve,
            "stale_residue_count_curve": stale_residue_count_curve,
            "fidelity_raw": fidelity_raw,
            "bloat_raw": bloat_raw,
            "contradiction_raw": contradiction_raw,
            "query_acc_raw": query_acc_raw,
            "retrieval_precision_raw": retrieval_precision_raw,
            "retrieval_recall_raw": retrieval_recall_raw,
            "rev_excess_count_raw": rev_excess_count_raw,
            "stale_residue_count_raw": stale_residue_count_raw,
            "session_results": session_results,
        }

    # ------------------------------------------------------------------
    # Tier-2 adapter path (black-box agents, e.g. OpenHands with a
    # controlled condenser). Additive: the Tier-1 run() above is untouched.
    # ------------------------------------------------------------------

    def _adapter_memory_text(self, adapter) -> str:
        """Read the agent's persisted memory for fidelity scoring.

        Prefer the adapter's declared memory text (notes/, .openhands_memory/).
        Fall back to scanning the workspace root for doc files, because a
        black-box agent often persists its knowledge base under a sensible name
        of its own choosing (e.g. ``knowledge_base.md`` at the root) rather than
        the exact ``notes/`` path the prompt suggested. Without this fallback the
        fidelity/contradiction metrics are blind to a memory the agent did keep —
        a faithfulness artifact, not a real aging signal.
        """
        try:
            mt = adapter.get_memory_text() or ""
        except Exception:
            mt = ""
        if mt.strip():
            return mt
        root = (getattr(adapter, "_cwd", None) or getattr(adapter, "_workspace_dir", None)
                or getattr(adapter, "workspace_dir", None))
        if not root:
            return mt
        from pathlib import Path
        parts, total = [], 0
        for f in sorted(Path(str(root)).rglob("*")):
            if not f.is_file() or f.suffix not in (".md", ".txt"):
                continue
            if ".openhands_persist" in str(f):  # conversation state, not memory
                continue
            try:
                txt = f.read_text(errors="replace")
            except Exception:
                continue
            parts.append(f"=== {f.name} ===\n{txt}")
            total += len(txt)
            if total > 50_000:
                break
        return "\n\n".join(parts)

    def _score_memory_metrics(self, current_memory: str, decisions_so_far: list, t: int) -> dict:
        """Score S3's memory-store metrics on ``current_memory``.

        Shared by the Tier-2 path. Calls the exact same validator functions the
        Tier-1 run() uses inline, so fidelity/contradiction/revision/bloat keep
        identical semantics — the only difference is the memory *source*
        (a black-box agent's notes vs. a controlled memory policy's store).
        """
        from ..scenarios.s3_knowledge_base.validator import (
            compute_fidelity_detailed, compute_contradiction_rate,
            compute_contradiction_count, score_revision_aging,
        )
        fidelity_detail = compute_fidelity_detailed(current_memory, decisions_so_far, at_session=t)
        rev_aging = score_revision_aging(current_memory, decisions_so_far, at_session=t)
        return {
            "fidelity_detail": fidelity_detail,
            "fidelity": fidelity_detail["fidelity"],
            "contradiction": compute_contradiction_rate(current_memory, decisions_so_far),
            "contradiction_count": compute_contradiction_count(current_memory, decisions_so_far),
            "rev_aging": rev_aging,
            "bloat": compute_memory_bloat(current_memory),
        }

    def run_adapter(self, adapter, n_sessions: int = 12, seed: int = 42,
                    reset_every: int = 0) -> dict:
        """Run S3 against a black-box AgentAdapter (Tier-2).

        Memory is owned by the agent (e.g. OpenHands' notes/ workspace +
        condenser), NOT by a MemoryPolicy. Each session delivers the new meeting
        transcript as an ingest turn, then asks the team queries; the agent's
        own context/condenser carry knowledge across sessions.

        Args:
            adapter: an AgentAdapter (send_message / reset_session / get_memory_text).
            n_sessions, seed: as in run().
            reset_every: clear the agent's conversation every N sessions
                (0 = never, so the condenser governs long-horizon memory — the
                meaningful regime for Level-A condenser comparisons). With a
                NoOp condenser and reset_every=0 a long run can overflow the
                model context; raise N or use a summarizing condenser then.

        Returns the SAME dict shape as run() so the CLI/report path is unchanged.
        Retrieval curves are empty (no ranked retriever in a black-box agent).
        """
        from ..scenarios.s3_knowledge_base.validator import score_query

        import random as _random
        _random.seed(seed)

        is_no_memory = False
        model_id = getattr(adapter, "_model", None) or self._model_id

        fidelity_raw, bloat_raw, contradiction_raw, query_acc_raw = [], [], [], []
        rev_excess_count_raw, stale_residue_count_raw = [], []
        session_results = []

        trajectory_path = self.tracer.path.parent / "trajectory.jsonl"
        traj_f = open(trajectory_path, "w", buffering=1)
        def _log_traj(event_type: str, **fields):
            import json as _json
            traj_f.write(_json.dumps(
                {"event": event_type, "timestamp": time.time(), **fields},
                ensure_ascii=False) + "\n")

        run_span = self.tracer.log(
            "run_start", parent_span_id=None, sut_id=self.sut_id,
            scenario=self.SCENARIO_ID, seed=seed, n_sessions=n_sessions,
            policy="adapter:" + type(adapter).__name__, oracle_mode=False,
        )

        actual_sessions = min(n_sessions, len(self.transcripts))
        print(f"  [S3][adapter] starting: sessions={actual_sessions}, "
              f"adapter={type(adapter).__name__}, model={model_id}, reset_every={reset_every}")

        from ..core.agent_adapter import AgentResponse as _AR
        def _safe_send(msg: str):
            """send_message that survives black-box failures (e.g. a context-
            window overflow when condenser=none accumulates past the model
            limit). Records the error and returns an empty response so the aging
            curve completes instead of aborting the whole run mid-stream."""
            try:
                return adapter.send_message(msg)
            except Exception as e:
                print(f"  [S3][adapter][warn] send failed: {type(e).__name__}: {str(e)[:140]}")
                return _AR(text="", metadata={"error": f"{type(e).__name__}: {e}",
                                              "condensations": 0, "cost_usd": 0.0})

        for t in range(actual_sessions):
            sess_span = self.tracer.log("session_start", parent_span_id=run_span, session=t)
            transcript = self.transcripts[t]
            transcript_text = transcript["transcript"]

            session_condensations = 0
            session_cost = 0.0
            session_errors = 0

            # ---- Ingest turn: deliver the new transcript; agent updates notes ----
            ingest_prompt = (
                f"New meeting transcript — {transcript['title']}:\n{transcript_text}\n\n"
                f"Update your project knowledge base with the key decisions, owners, "
                f"dates, dollar amounts, and version numbers from this meeting. "
                f"Revise any earlier decision that changed."
            )
            ingest = _safe_send(ingest_prompt)
            session_condensations += int(ingest.metadata.get("condensations", 0) or 0)
            session_cost += float(ingest.metadata.get("cost_usd", 0.0) or 0.0)
            session_errors += 1 if ingest.metadata.get("error") else 0

            # ---- Team queries (scored on the agent's answers) ----
            queries = self.queries_by_session[t]["queries"] if t < len(self.queries_by_session) else []
            query_responses, query_scores = [], []
            for q in queries:
                prompt = (
                    f"A team member asks: {q['question']}\n\n"
                    f"Consult your project knowledge base and answer with specific "
                    f"details (names, dates, dollar amounts, version numbers)."
                )
                resp = _safe_send(prompt)
                response = resp.text
                query_responses.append(response)
                sc = score_query(response, q)
                query_scores.append(sc)
                session_condensations += int(resp.metadata.get("condensations", 0) or 0)
                session_cost += float(resp.metadata.get("cost_usd", 0.0) or 0.0)
                session_errors += 1 if resp.metadata.get("error") else 0
                _log_traj("agent_output", session=t, phase="query",
                          query_id=q["query_id"], output=response[:500], score=sc)
                self.tracer.log("query_answered", parent_span_id=sess_span,
                                session=t, query_id=q["query_id"], score=sc)
            query_acc = sum(query_scores) / len(query_scores) if query_scores else 0.0

            # ---- Interference binding probes (forced-choice) ----
            interference_probe_results = []
            for bp in self._binding_probes_due_at(t, actual_sessions):
                bp_resp = _safe_send(
                    f"A team member asks: {bp['question']}\n\n"
                    f"Consult your project knowledge base and answer with the exact value only."
                )
                interference_probe_results.append({
                    "session": t, "task_id": bp.get("probe_id"),
                    "question": bp["question"], "response_text": bp_resp.text,
                    "gold_value": bp.get("gold_value"),
                    "distractor_value": bp.get("distractor_value"),
                })
                session_condensations += int(bp_resp.metadata.get("condensations", 0) or 0)
                session_cost += float(bp_resp.metadata.get("cost_usd", 0.0) or 0.0)

            # ---- Score memory-store metrics on the agent's own memory text ----
            current_memory = self._adapter_memory_text(adapter)
            decisions_so_far = self._get_decisions_up_to(t)
            m = self._score_memory_metrics(current_memory, decisions_so_far, t)
            fidelity = m["fidelity"]; contradiction = m["contradiction"]; rev_aging = m["rev_aging"]
            bloat = m["bloat"]

            fidelity_raw.append((t, fidelity))
            bloat_raw.append((t, bloat))
            contradiction_raw.append((t, contradiction))
            query_acc_raw.append((t, query_acc))
            if rev_aging["revision_fidelity_excess_count"] is not None:
                rev_excess_count_raw.append((t, rev_aging["revision_fidelity_excess_count"]))
            if rev_aging["coverage_verdict"] != "no_revisions":
                stale_residue_count_raw.append((t, rev_aging["stale_residue_count"]))

            task_outputs_text = " ".join(str(r)[:500] for r in query_responses if r)
            sr = {
                "session": t, "title": transcript["title"],
                "query_accuracy": query_acc, "query_scores": query_scores,
                "fidelity": fidelity, "category_fidelity": m["fidelity_detail"]["category_fidelity"],
                "contradiction_rate": contradiction, "contradiction_count": m["contradiction_count"],
                "revision_aging": rev_aging, "memory_bloat": bloat,
                "n_decisions_so_far": len(decisions_so_far),
                "retrieval_precision": None, "retrieval_recall": None,
                "task_outputs_text": task_outputs_text,
                "interference_probes": interference_probe_results,
                # Level-A observability: condenser activity + cost this session.
                "condensations": session_condensations,
                "condenser": ingest.metadata.get("condenser", {}),
                "session_cost_usd": round(session_cost, 6),
                "memory_chars": len(current_memory),
                "adapter_errors": session_errors,
            }
            session_results.append(sr)

            self.tracer.log("session_scored", parent_span_id=sess_span, session=t,
                            query_accuracy=round(query_acc, 4), fidelity=round(fidelity, 4),
                            contradiction_rate=round(contradiction, 4), memory_bloat=bloat,
                            condensations=session_condensations)
            self.tracer.log("session_end", parent_span_id=sess_span, session=t)
            print(f"  [S3][adapter] Session {t:2d}  fidelity={fidelity:.3f}  "
                  f"query_acc={query_acc:.3f}  contradict={contradiction:.3f}  "
                  f"condensed={session_condensations}  mem_chars={len(current_memory)}  "
                  f"errors={session_errors}  decisions={len(decisions_so_far)}")

            if reset_every and (t + 1) % reset_every == 0:
                adapter.reset_session()

        def _curve(raw):
            return AgingCurve(exposures=[r[0] for r in raw], scores=[r[1] for r in raw],
                              scenario=self.SCENARIO_ID, sut_id=self.sut_id)
        empty = _curve([])

        self.tracer.log("run_end", parent_span_id=run_span,
                        fidelity_curve=fidelity_raw, contradiction_curve=contradiction_raw,
                        bloat_curve=bloat_raw, query_curve=query_acc_raw)
        _log_traj("run_end", n_sessions=actual_sessions,
                  m_final=fidelity_raw[-1][1] if fidelity_raw else 0)
        traj_f.close()

        return {
            "fidelity_curve": _curve(fidelity_raw),
            "bloat_curve": _curve(bloat_raw),
            "contradiction_curve": AgingCurve(
                exposures=[r[0] for r in contradiction_raw],
                scores=[1.0 - r[1] for r in contradiction_raw],
                scenario=self.SCENARIO_ID, sut_id=self.sut_id),
            "query_curve": _curve(query_acc_raw),
            "retrieval_precision_curve": empty,
            "retrieval_recall_curve": empty,
            "rev_excess_count_curve": _curve(rev_excess_count_raw),
            "stale_residue_count_curve": _curve(stale_residue_count_raw),
            "fidelity_raw": fidelity_raw,
            "bloat_raw": bloat_raw,
            "contradiction_raw": contradiction_raw,
            "query_acc_raw": query_acc_raw,
            "retrieval_precision_raw": [],
            "retrieval_recall_raw": [],
            "rev_excess_count_raw": rev_excess_count_raw,
            "stale_residue_count_raw": stale_residue_count_raw,
            "session_results": session_results,
        }
