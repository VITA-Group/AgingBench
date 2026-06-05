"""
agingbench/runner/s3_runner.py — S3 Project Knowledge Base Agent.

Implements the 12-session state machine for the Knowledge Base scenario.
Each session: agent reads transcript + M_{t-1}, answers 3 team queries,
writes interaction history to memory.

Produces G3 metrics: summarization_fidelity, memory_bloat, contradiction_rate.

Recall is measured on TWO channels by default:
  * ``query_accuracy``           — IN-CHANNEL. Conversational queries the agent
                                   answers as part of the workload; the Q/A pair
                                   enters memory at end-of-session, providing
                                   testing-effect rehearsal for future re-asks.
                                   This is the deployed-conversation lifespan
                                   number — what a user actually experiences.
  * ``held_out_query_accuracy``  — HELD-OUT. Eval-only probes about facts from
                                   strictly-earlier sessions, sampled disjoint
                                   from in-channel queries. Asked and scored,
                                   but NEVER written to memory. This is the
                                   clean substrate-decay number, directly
                                   comparable to S6 and open-loop memory
                                   benchmarks. Default 2 probes/session.

The delta ``query_accuracy − held_out_query_accuracy`` quantifies the
testing-effect contribution (range +0.10 to +0.23 across compression
budgets in Qwen3-14B-thinking + summarize_store, 3 seeds). Pass
``--held-out-probes 0`` to disable for reproducing legacy single-channel
results.
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


def _snake_case_policy_name(memory_policy) -> str:
    """Map memory policy class name to snake_case key used by awareness module.

    E.g. ``SummarizeStorePolicy`` → ``summarize_store``,
    ``AppendOnlyPolicy``     → ``append_only``.
    """
    import re
    cls = type(memory_policy).__name__
    cls = re.sub(r"Policy$", "", cls)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", cls).lower()


class S3Runner(BaseRunner):
    SCENARIO_ID = "s3_knowledge_base"

    def __init__(
        self,
        memory_policy: MemoryPolicy,
        llm,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        agent_class: type[AgentInterface] = ReferenceAgent,
        generated_data: dict | None = None,
        scenario_prompt_mode: str = "legacy",
        pressure=None,
        held_out_probes_per_session: int = 2,
    ):
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        if self.llm is not None:
            self.llm.tracer = self.tracer
        self.sut_id = sut_id
        # Held-out probe channel (default 2 probes/session, on by default since
        # the 3-seed budget sweep showed testing-effect delta of +0.10 to +0.23
        # across compression budgets — a non-trivial confound on in-channel
        # recall numbers). Sampled from STRICTLY-EARLIER sessions (lag > 0),
        # asked + scored, but Q/A is NEVER written to memory. Held-out
        # recall is the clean substrate-decay measurement; in-channel
        # recall is the deployed conversational lifespan number.
        # Set to 0 to disable (e.g. reproducing legacy single-channel results).
        self.held_out_probes_per_session = held_out_probes_per_session
        self.agent_class = agent_class
        # Opt-in scenario-aware system prompt. ``legacy`` (default) keeps
        # the existing REACT_SYSTEM template untouched. ``scenario_aware``
        # builds a per-scenario template with PressureConfig-derived
        # awareness and attaches it to the agent via system_template.
        self.scenario_prompt_mode = scenario_prompt_mode
        self.pressure = pressure
        if scenario_prompt_mode in ("scenario_aware", "scenario_aware_lean"):
            from ..prompts.scenario_aware import build_system_template
            self._scenario_system_template = build_system_template(
                scenario=self.SCENARIO_ID,
                pressure=pressure,
                memory_policy_type=_snake_case_policy_name(memory_policy),
                variant="lean" if scenario_prompt_mode == "scenario_aware_lean" else "standard",
            )
        else:
            self._scenario_system_template = None

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

    @staticmethod
    def _decision_to_probe_text(decision: dict, probe_idx: int, at_session: int) -> dict:
        """Convert a decision into a held-out probe (mirrors S3Generator._decision_to_query).

        Returns a probe dict with ``probe_id``, ``question``, ``gold_decision_ids``,
        ``keywords``, ``origin_session``. Kept here as a static helper so the
        runner does not need a reference to the generator instance.
        """
        cat = decision["category"]
        fact = decision["fact"]
        q_templates = {
            "budget": f"What was the budget figure for: {fact.split()[0]} {fact.split()[1] if len(fact.split()) > 1 else ''}?",
            "tech": f"What technology decision was made regarding {fact.split()[0]}?",
            "vendor": f"Which vendor was selected for {fact.split()[-1] if len(fact.split()) > 2 else 'this service'}?",
            "timeline": f"What is the timeline for {fact.split()[0]}?",
            "security": f"What security measure was decided for {fact.split()[-1]}?",
            "hiring": f"What hiring decision was made?",
            "infra": f"What infrastructure setup was decided?",
        }
        return {
            "probe_id": f"ho_s{at_session}_{probe_idx + 1}",
            "question": q_templates.get(cat, f"What was decided about {cat}?"),
            "gold_decision_ids": [decision["id"]],
            "keywords": decision["keywords"][:3],
            "origin_session": decision["session"],
        }

    def _build_held_out_probes(
        self,
        at_session: int,
        excluded_decision_ids: set[str],
        seed: int,
    ) -> list[dict]:
        """Sample ``held_out_probes_per_session`` probes from STRICTLY-EARLIER
        sessions (origin < at_session). Excludes decisions whose ids appear in
        ``excluded_decision_ids`` (so a held-out probe does not double-ask a
        decision the in-channel queries already covered this session).

        Deterministic: derives a per-session RNG from ``seed`` + ``at_session``
        so runs with the same seed produce identical held-out probe streams.
        """
        n = self.held_out_probes_per_session
        if n <= 0 or at_session == 0:
            return []
        import random as _random
        rng = _random.Random(seed * 100003 + at_session)
        pool = [
            d for d in self.gold_decisions
            if d["session"] < at_session and d["id"] not in excluded_decision_ids
        ]
        if not pool:
            return []
        k = min(n, len(pool))
        picks = rng.sample(pool, k)
        return [
            self._decision_to_probe_text(d, i, at_session)
            for i, d in enumerate(picks)
        ]

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

        # Accumulate all transcripts so far (used as scoring reference under NoMemoryPolicy)
        all_transcripts_text = ""

        is_append_only = type(self.memory_policy).__name__ == "AppendOnlyPolicy"

        fidelity_raw = []
        bloat_raw = []
        contradiction_raw = []
        query_acc_raw = []
        # Held-out probe accuracy curve — populated only when
        # held_out_probes_per_session > 0. Computed exactly like query_acc_raw
        # but on the held-out channel that is NEVER written to memory, so the
        # delta query_acc_raw[t] - held_out_acc_raw[t] quantifies the
        # testing-effect contribution to S3's recall at session t.
        held_out_acc_raw = []
        retrieval_precision_raw = []
        retrieval_recall_raw = []
        # Revision-aging trident raw series (count form for aging curves;
        # the rate form lives in session_results for per-session readability).
        rev_excess_count_raw = []
        stale_residue_count_raw = []
        session_results = []

        run_span = self.tracer.log(
            "run_start", parent_span_id=None,
            sut_id=self.sut_id, scenario=self.SCENARIO_ID,
            seed=seed, n_sessions=n_sessions,
            policy=type(self.memory_policy).__name__,
        )

        actual_sessions = min(n_sessions, len(self.transcripts))
        _progress(
            f"starting run: sessions={actual_sessions}, policy={type(self.memory_policy).__name__}"
        )

        for t in range(actual_sessions):
            session_t0 = time.time()
            sess_span = self.tracer.log(
                "session_start", parent_span_id=run_span, session=t,
            )

            transcript = self.transcripts[t]
            transcript_text = transcript["transcript"]
            all_transcripts_text += f"\n\n--- Session {t}: {transcript['title']} ---\n{transcript_text}"

            # Read memory: empty under NoMemoryPolicy, compressed via memory_policy otherwise.
            if t == 0 or is_no_memory:
                memory_text = ""
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
            # Opt-in scenario-aware system prompt: attach the prebuilt template
            # so the agent uses it instead of legacy REACT_SYSTEM. When mode is
            # ``legacy`` (default) or the scenario has no registered template,
            # ``_scenario_system_template`` is None and the agent's behavior is
            # bit-for-bit identical to before.
            if self._scenario_system_template is not None:
                agent.system_template = self._scenario_system_template

            # Agent context: memory + new transcript
            context = ""
            if memory_text:
                context += f"Project Knowledge Base (accumulated decisions):\n{memory_text}\n\n"
            context += f"New Meeting Transcript — {transcript['title']}:\n{transcript_text}\n\n"

            # Log memory state before queries
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

            # ---- Held-out probe channel ----
            # Sampled from strictly-earlier sessions, disjoint from in-channel
            # query decision ids this session. Asked and scored, but the
            # responses NEVER enter memory: the loop returns before
            # ``interaction`` is built. This isolates substrate decay from the
            # testing-effect rehearsal contributed by in-channel Q/A writes.
            held_out_results: list[dict] = []
            held_out_acc: Optional[float] = None
            if self.held_out_probes_per_session > 0:
                excluded_ids: set[str] = set()
                for q in queries:
                    excluded_ids.update(q.get("gold_decision_ids", []) or [])
                ho_probes = self._build_held_out_probes(
                    at_session=t,
                    excluded_decision_ids=excluded_ids,
                    seed=seed,
                )
                if ho_probes:
                    _progress(
                        f"session {t + 1}: held-out probes start ({len(ho_probes)})",
                        session_t0,
                    )
                ho_scores: list[float] = []
                for hp in ho_probes:
                    hp_prompt = (
                        f"{context}"
                        f"A team member asks: {hp['question']}\n\n"
                        f"Search your knowledge base if needed, then answer with specific details "
                        f"(names, dates, dollar amounts, version numbers)."
                    )
                    hp_result = agent.run_session(hp_prompt, session_id=t)
                    hp_response = hp_result["output"]
                    # score_query expects a query dict with at least 'keywords'
                    # and 'question'. _decision_to_probe_text produces a
                    # compatible shape so we can reuse the canonical scorer.
                    hp_score = score_query(hp_response, hp)
                    ho_scores.append(hp_score)
                    held_out_results.append({
                        "session": t,
                        "probe_id": hp["probe_id"],
                        "question": hp["question"],
                        "origin_session": hp.get("origin_session"),
                        "lag": t - int(hp.get("origin_session", t)),
                        "gold_decision_ids": hp["gold_decision_ids"],
                        "keywords": hp["keywords"],
                        "response_text": hp_response,
                        "score": hp_score,
                    })
                    self.tracer.log(
                        "held_out_probe_answered", parent_span_id=sess_span,
                        session=t, probe_id=hp["probe_id"],
                        origin_session=hp.get("origin_session"),
                        score=hp_score, turns=hp_result.get("turns", 0),
                    )
                if ho_scores:
                    held_out_acc = sum(ho_scores) / len(ho_scores)

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
            # Write interaction history to memory
            interaction = f"Session {t}: {transcript['title']}\n{transcript_text}\n"
            for q, r in zip(queries, query_responses):
                interaction += f"Q: {q['question']}\nA: {r}\n"

            out_tok = -1
            if not is_no_memory:
                _progress(f"session {t + 1}: memory write start", session_t0)
                self.memory_policy.write(interaction, llm=self.llm)

                compressed = self.memory_policy.read()
                in_tok = getattr(self.memory_policy, "last_input_tokens", 0)
                out_tok = getattr(self.memory_policy, "last_output_tokens", 0)


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
            if held_out_acc is not None:
                held_out_acc_raw.append((t, held_out_acc))
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
                "query_accuracy": query_acc,
                "fidelity": fidelity,
                "contradiction_rate": contradiction,
                "revision_aging": rev_aging,
                "memory_bloat": bloat,
                "retrieval_precision": ret_precision,
                "retrieval_recall": ret_recall,
                "task_outputs_text": task_outputs_text,
                "interference_probes": interference_probe_results,
                "response_tokens_per_query": query_token_counts,
                "response_tokens_max": max(valid_qt) if valid_qt else 0,
                "memory_write_tokens": out_tok,
                # Held-out probe channel results (empty list + None when
                # held_out_probes_per_session == 0; populated otherwise).
                "held_out_results": held_out_results,
                "held_out_query_accuracy": held_out_acc,
            }
            session_results.append(sr)


            _rev_excess_rate_logged = (
                round(rev_aging["revision_fidelity_excess"], 4)
                if rev_aging["revision_fidelity_excess"] is not None else None
            )
            _stale_rate_logged = (
                round(rev_aging["stale_residue_rate"], 4)
                if rev_aging["stale_residue_rate"] is not None else None
            )

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
        # Held-out query curve. Empty (zero-length) when the runner was
        # configured with held_out_probes_per_session == 0; downstream metrics
        # treat empty curves as N/A, not 0.
        held_out_query_curve = AgingCurve(
            exposures=[r[0] for r in held_out_acc_raw],
            scores=[r[1] for r in held_out_acc_raw],
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
        _progress(f"run complete: m_final={fidelity_raw[-1][1] if fidelity_raw else 0:.3f}")

        return {
            "fidelity_curve": fidelity_curve,
            "bloat_curve": bloat_curve,
            "contradiction_curve": contradiction_curve,
            "query_curve": query_curve,
            "held_out_query_curve": held_out_query_curve,
            "retrieval_precision_curve": retrieval_precision_curve,
            "retrieval_recall_curve": retrieval_recall_curve,
            "rev_excess_count_curve": rev_excess_count_curve,
            "stale_residue_count_curve": stale_residue_count_curve,
            "fidelity_raw": fidelity_raw,
            "bloat_raw": bloat_raw,
            "contradiction_raw": contradiction_raw,
            "query_acc_raw": query_acc_raw,
            "held_out_acc_raw": held_out_acc_raw,
            "retrieval_precision_raw": retrieval_precision_raw,
            "retrieval_recall_raw": retrieval_recall_raw,
            "rev_excess_count_raw": rev_excess_count_raw,
            "stale_residue_count_raw": stale_residue_count_raw,
            "session_results": session_results,
            "held_out_probes_per_session": self.held_out_probes_per_session,
        }
