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
        else:
            data_dir = Path(__file__).parent.parent / "scenarios" / "s3_knowledge_base"
            with open(data_dir / "transcripts.json") as f:
                self.transcripts = json.load(f)["sessions"]
            with open(data_dir / "gold_timeline.json") as f:
                self.gold_decisions = json.load(f)["decisions"]
            with open(data_dir / "queries.json") as f:
                self.queries_by_session = json.load(f)["sessions"]

    def _get_decisions_up_to(self, session: int) -> list[dict]:
        """Get all gold decisions introduced up to and including this session."""
        return [d for d in self.gold_decisions if d["session"] <= session]

    def _build_tools(self, memory_text: str) -> ToolRegistry:
        """Build tools for the agent: search_memory only."""
        registry = ToolRegistry()

        def search_memory(arguments: dict) -> str:
            query = arguments.get("query", "")
            return memory_text[:3000] if memory_text else "(no memory available)"

        registry.register(ToolSpec(
            name="search_memory",
            version="1.0.0",
            description="Search the project knowledge base for past decisions, facts, and meeting notes.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"],
            },
            fn=search_memory,
        ))
        return registry

    def run(self, n_sessions: int = 12, seed: int = 42) -> dict:
        """Run S3 for n_sessions; return dict with aging curves and session results."""
        from ..scenarios.s3_knowledge_base.validator import (
            score_query, compute_fidelity_detailed, compute_contradiction_rate,
            compute_contradiction_count,
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
            agent = self.agent_class(
                llm=self.llm,
                memory_policy=self.memory_policy,
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

            fidelity_detail = compute_fidelity_detailed(current_memory, decisions_so_far)
            fidelity = fidelity_detail["fidelity"]
            contradiction = compute_contradiction_rate(current_memory, decisions_so_far)
            contradiction_count = compute_contradiction_count(current_memory, decisions_so_far)
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
                        # Check which gold decisions appear in retrieved chunks
                        retrieved_ids = set()
                        for did in gold_ids:
                            decision = next((d for d in self.gold_decisions if d["id"] == did), None)
                            if decision:
                                for rt in retrieved_texts:
                                    if any(kw.lower() in rt.lower() for kw in decision["keywords"]):
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
                "memory_bloat": bloat,
                "n_decisions_so_far": len(decisions_so_far),
                "retrieval_precision": ret_precision,
                "retrieval_recall": ret_recall,
                "task_outputs_text": task_outputs_text,
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

            print(f"  [S3] Session {t:2d}  fidelity={fidelity:.3f}  "
                  f"query_acc={query_acc:.3f}  contradict={contradiction:.3f}  "
                  f"bloat={bloat}  decisions={len(decisions_so_far)}")

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

        self.tracer.log(
            "run_end", parent_span_id=run_span,
            fidelity_curve=fidelity_raw,
            contradiction_curve=contradiction_raw,
            bloat_curve=bloat_raw,
            query_curve=query_acc_raw,
            retrieval_precision_curve=retrieval_precision_raw,
            retrieval_recall_curve=retrieval_recall_raw,
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
            "fidelity_raw": fidelity_raw,
            "bloat_raw": bloat_raw,
            "contradiction_raw": contradiction_raw,
            "query_acc_raw": query_acc_raw,
            "retrieval_precision_raw": retrieval_precision_raw,
            "retrieval_recall_raw": retrieval_recall_raw,
            "session_results": session_results,
        }
