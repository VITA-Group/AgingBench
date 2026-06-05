"""
agingbench/runner/s3_runner_unified.py — S3 with the unified FullReactAgent.

Standalone variant of ``s3_runner.S3Runner``. Same scenario data, same gold
timeline, same scoring (fidelity_detailed, contradiction, revision_aging,
query_accuracy). The only behavioural difference is how memory is delivered
to the agent.

Original S3Runner (dual-channel memory):
    context  = memory_policy.read() dumped into the user prompt
             + new transcript
             + question
    tools    = read_memory   (redundant — memory already in context)

S3 unified (tool-only memory):
    context  = new transcript + question                ← NO memory dump
    tools    = search_memory(query) -> top-K chunks     ← only access path

This is the same unification S6 received. It exists because S3's original
dual-channel design left the agent loop dead at runtime (median 1 turn,
0 tool calls): with memory already in the prompt, the search_memory tool
had nothing useful to offer.

Per-session telemetry — ``turn_stats``, ``loop_utilization`` — matches the
S6 unified runner. The held-out probe machinery added to S3Runner is
preserved here in the same role: ablation tool for testing-effect
quantification, NOT a primary metric (see paper §3 methods note).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Optional

from .base import BaseRunner
from .trace import TraceLogger
from ..metrics.aging import AgingCurve
from ..metrics.g3_metrics import compute_memory_bloat
from ..core.memory.base import MemoryPolicy
from ..core.full_react_agent import FullReactAgent, build_search_memory_tool
from ..core.tools import ToolRegistry


def _loop_utilization_from_turn_stats(turn_stats: list[dict]) -> dict:
    """Same summary shape as s6_runner_unified — kept in sync deliberately."""
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


class S3UnifiedRunner(BaseRunner):
    """S3 (Project Knowledge Base) using FullReactAgent + search_memory tool.

    Scoring shape is identical to ``S3Runner.run()`` so results can be diffed
    side-by-side with matched seed to validate that the unified design
    preserves the headline measurement.
    """

    SCENARIO_ID = "s3_knowledge_base"

    def __init__(
        self,
        memory_policy: MemoryPolicy,
        llm,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        generated_data: Optional[dict] = None,
        agent_max_turns: int = 10,
        search_memory_top_k: int = 3,
        held_out_probes_per_session: int = 2,
    ):
        # Held-out default = 2 (matches S3Runner; see s3_runner.py for the
        # rationale tied to the 3-seed budget-sweep finding).
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        if self.llm is not None:
            self.llm.tracer = self.tracer
        self.sut_id = sut_id
        self.agent_max_turns = agent_max_turns
        self.search_memory_top_k = search_memory_top_k
        self.held_out_probes_per_session = held_out_probes_per_session

        if generated_data:
            self.transcripts = generated_data["transcripts"]["sessions"]
            self.gold_decisions = generated_data["gold_timeline"]["decisions"]
            self.queries_by_session = generated_data["queries"]["sessions"]
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
            self.binding_probes_by_session = []
            self.binding_probe_lags = [1, 3]

        self._model_id = (
            getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
        )
        self._provider = "local_hf" if hasattr(llm, "tok") else "litellm"

    # -- helpers (mirror S3Runner) ----------------------------------------

    def _get_decisions_up_to(self, session: int) -> list[dict]:
        return [d for d in self.gold_decisions if d["session"] <= session]

    def _binding_probes_due_at(self, t: int, n_sessions: int) -> list[dict]:
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
                if t == n_sessions - 1:
                    due.extend(probes)
        return due

    @staticmethod
    def _decision_to_probe_text(decision: dict, probe_idx: int, at_session: int) -> dict:
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

    # -- main loop ---------------------------------------------------------

    def run(self, n_sessions: int = 12, seed: int = 42) -> dict:
        from ..scenarios.s3_knowledge_base.validator import (
            score_query, compute_fidelity_detailed, compute_contradiction_rate,
            compute_contradiction_count, score_revision_aging,
            _present as _kw_boundary_present,
        )

        import random as _random
        _random.seed(seed)

        self.memory_policy.reset()
        is_no_memory = type(self.memory_policy).__name__ == "NoMemoryPolicy"
        is_append_only = type(self.memory_policy).__name__ == "AppendOnlyPolicy"
        progress_on = os.getenv("AGINGBENCH_S3_PROGRESS", "1").lower() not in {
            "0", "false", "no", "off"
        }

        run_t0 = time.time()
        def _fmt_elapsed(start: float) -> str:
            delta = int(time.time() - start)
            m, s = divmod(delta, 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

        def _progress(message: str, session_start: float | None = None):
            if not progress_on:
                return
            run_elapsed = _fmt_elapsed(run_t0)
            tag = (
                f"[S3/unified][progress][run {run_elapsed}"
                + (f" | session {_fmt_elapsed(session_start)}]" if session_start else "]")
            )
            print(f"  {tag} {message}", flush=True)

        # Curves
        fidelity_raw, bloat_raw, contradiction_raw = [], [], []
        query_acc_raw: list[tuple[int, float]] = []
        held_out_acc_raw: list[tuple[int, float]] = []
        rev_excess_count_raw, stale_residue_count_raw = [], []
        session_results: list[dict] = []

        run_span = self.tracer.log(
            "run_start", parent_span_id=None,
            sut_id=self.sut_id, scenario=self.SCENARIO_ID,
            seed=seed, n_sessions=n_sessions,
            policy=type(self.memory_policy).__name__,
            agent="full_react_agent",
            agent_max_turns=self.agent_max_turns,
        )

        actual_sessions = min(n_sessions, len(self.transcripts))
        _progress(
            f"starting run: sessions={actual_sessions}, "
            f"policy={type(self.memory_policy).__name__}, "
            f"agent=FullReactAgent(max_turns={self.agent_max_turns}), "
            f"search_memory_top_k={self.search_memory_top_k}, "
            f"held_out_probes_per_session={self.held_out_probes_per_session}"
        )

        all_transcripts_text = ""

        for t in range(actual_sessions):
            session_t0 = time.time()
            transcript = self.transcripts[t]
            transcript_text = transcript["transcript"]
            all_transcripts_text += f"\n\n--- Session {t}: {transcript['title']} ---\n{transcript_text}"

            sess_span = self.tracer.log(
                "session_start", parent_span_id=run_span,
                session=t, transcript_id=transcript.get("id", t),
            )

            # --- Build tools (memory access ONLY through this) ---
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

            # --- Context: transcript ONLY, no memory dump ---
            context = (
                f"New Meeting Transcript — {transcript['title']}:\n"
                f"{transcript_text}\n\n"
            )

            queries = self.queries_by_session[t]["queries"] if t < len(self.queries_by_session) else []
            _progress(
                f"session {t + 1}/{actual_sessions} start: queries={len(queries)}",
                session_t0,
            )

            # --- In-channel queries ---
            query_responses = []
            query_scores = []
            query_turn_stats: list[dict] = []
            for q_i, q in enumerate(queries, start=1):
                prompt = (
                    f"{context}"
                    f"A team member asks: {q['question']}\n\n"
                    f"Search your knowledge base if you need facts from earlier "
                    f"sessions, then answer with specific details (names, dates, "
                    f"dollar amounts, version numbers)."
                )
                result = agent.run_session(prompt, session_id=t)
                response = result["output"]
                query_responses.append(response)
                query_scores.append(score_query(response, q))
                query_turn_stats.extend(result.get("turn_stats", []))

                self.tracer.log(
                    "query_answered", parent_span_id=sess_span,
                    session=t, query_id=q["query_id"],
                    score=query_scores[-1], turns=result.get("turns", 0),
                    n_tool_calls=len(result.get("tool_calls", [])),
                )
            query_acc = sum(query_scores) / len(query_scores) if query_scores else 0.0

            # --- Held-out probes (eval-only; never written) ---
            held_out_results: list[dict] = []
            held_out_acc: Optional[float] = None
            held_out_turn_stats: list[dict] = []
            if self.held_out_probes_per_session > 0:
                excluded_ids = set()
                for q in queries:
                    excluded_ids.update(q.get("gold_decision_ids", []) or [])
                ho_probes = self._build_held_out_probes(t, excluded_ids, seed)
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
                        f"Search your knowledge base if needed, then answer with "
                        f"specific details."
                    )
                    hp_result = agent.run_session(hp_prompt, session_id=t)
                    hp_response = hp_result["output"]
                    hp_score = score_query(hp_response, hp)
                    ho_scores.append(hp_score)
                    held_out_turn_stats.extend(hp_result.get("turn_stats", []))
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
                        n_tool_calls=len(hp_result.get("tool_calls", [])),
                    )
                if ho_scores:
                    held_out_acc = sum(ho_scores) / len(ho_scores)

            # --- Interference binding probes (same as S3Runner) ---
            interference_probe_results = []
            due_probes = self._binding_probes_due_at(t, actual_sessions)
            for bp in due_probes:
                bp_prompt = (
                    f"{context}"
                    f"A team member asks: {bp['question']}\n\n"
                    f"Search your knowledge base if needed, then answer with the "
                    f"exact value only."
                )
                bp_result = agent.run_session(bp_prompt, session_id=t)
                interference_probe_results.append({
                    "session": t,
                    "task_id": bp.get("probe_id"),
                    "question": bp["question"],
                    "response_text": bp_result["output"],
                    "gold_value": bp.get("gold_value"),
                    "distractor_value": bp.get("distractor_value"),
                })

            # --- Write interaction to memory (in-channel only; held-out excluded) ---
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
                    input_preview=interaction, output_preview=compressed or "",
                    thought=getattr(self.llm, "last_thought", ""),
                    cycle=t,
                )
                _progress(
                    f"session {t + 1}: memory write done (in_tok={in_tok}, out_tok={out_tok})",
                    session_t0,
                )

            # --- Scoring (identical to S3Runner) ---
            current_memory = self.memory_policy.read() if not is_no_memory else all_transcripts_text
            decisions_so_far = self._get_decisions_up_to(t)
            fidelity_detail = compute_fidelity_detailed(
                current_memory, decisions_so_far, at_session=t
            )
            fidelity = fidelity_detail["fidelity"]
            contradiction = compute_contradiction_rate(current_memory, decisions_so_far)
            rev_aging = score_revision_aging(current_memory, decisions_so_far, at_session=t)
            bloat = compute_memory_bloat(current_memory)

            fidelity_raw.append((t, fidelity))
            bloat_raw.append((t, bloat))
            contradiction_raw.append((t, contradiction))
            query_acc_raw.append((t, query_acc))
            if held_out_acc is not None:
                held_out_acc_raw.append((t, held_out_acc))
            if rev_aging["revision_fidelity_excess_count"] is not None:
                rev_excess_count_raw.append(
                    (t, rev_aging["revision_fidelity_excess_count"])
                )
            if rev_aging["coverage_verdict"] != "no_revisions":
                stale_residue_count_raw.append(
                    (t, rev_aging["stale_residue_count"])
                )

            # --- Loop utilization (combined across in-channel + held-out) ---
            combined_turn_stats = list(query_turn_stats) + list(held_out_turn_stats)
            loop_util = _loop_utilization_from_turn_stats(combined_turn_stats)

            session_results.append({
                "session": t,
                "query_accuracy": query_acc,
                "fidelity": fidelity,
                "contradiction_rate": contradiction,
                "revision_aging": rev_aging,
                "memory_bloat": bloat,
                "held_out_results": held_out_results,
                "held_out_query_accuracy": held_out_acc,
                "interference_probes": interference_probe_results,
                "turn_stats": combined_turn_stats,
                "loop_utilization": loop_util,
                "memory_write_tokens": out_tok,
            })

            _progress(
                f"session {t + 1}/{actual_sessions} end: "
                f"fidelity={fidelity:.3f}, query_acc={query_acc:.3f}, "
                f"turns_med={loop_util['turns_median']}, "
                f"tools={loop_util['tool_calls_total']}",
                session_t0,
            )
            self.tracer.log(
                "session_scored", parent_span_id=sess_span, session=t,
                fidelity=fidelity, query_accuracy=query_acc,
                loop_utilization=loop_util,
            )
            self.tracer.log(
                "session_end", parent_span_id=sess_span, session=t,
            )

        # --- Curves ---
        fidelity_curve = AgingCurve(
            exposures=[r[0] for r in fidelity_raw],
            scores=[r[1] for r in fidelity_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        query_curve = AgingCurve(
            exposures=[r[0] for r in query_acc_raw],
            scores=[r[1] for r in query_acc_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        held_out_query_curve = AgingCurve(
            exposures=[r[0] for r in held_out_acc_raw],
            scores=[r[1] for r in held_out_acc_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )

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
                sum(1 for lu in per_sess_loop if lu.get("exhausted_count", 0) > 0)
                / max(len(per_sess_loop), 1)
            ),
            "had_reasoning_share_overall": (
                sum(lu.get("had_reasoning_share", 0.0) for lu in per_sess_loop)
                / max(len(per_sess_loop), 1)
            ),
        }

        self.tracer.log(
            "run_end", parent_span_id=run_span,
            fidelity_curve=fidelity_raw, query_curve=query_acc_raw,
            held_out_curve=held_out_acc_raw,
            loop_utilization=run_loop_summary,
        )
        _progress(
            f"unified run complete: m_final="
            f"{fidelity_raw[-1][1] if fidelity_raw else 0:.3f}"
        )

        return {
            "fidelity_curve": fidelity_curve,
            "query_curve": query_curve,
            "held_out_query_curve": held_out_query_curve,
            "fidelity_raw": fidelity_raw,
            "bloat_raw": bloat_raw,
            "contradiction_raw": contradiction_raw,
            "query_acc_raw": query_acc_raw,
            "held_out_acc_raw": held_out_acc_raw,
            "rev_excess_count_raw": rev_excess_count_raw,
            "stale_residue_count_raw": stale_residue_count_raw,
            "session_results": session_results,
            "held_out_probes_per_session": self.held_out_probes_per_session,
            "loop_utilization": run_loop_summary,
        }
