"""
agingbench/runner/runner.py — S1 ScenarioRunner (formerly P2).

Implements the S1 cycle state machine for the Research Literature Agent
scenario. Uses structured trace logging with OpenInference-aligned field
names including token counts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from .base import BaseRunner, RunResult
from .trace import TraceLogger
from ..metrics.aging import AgingCurve, compute_half_life, compute_decay_slope
from ..metrics.dependency_scorer import _kw_in_text
from ..core.memory.base import MemoryPolicy
from ..core.memory.summarize_store import SummarizeStorePolicy
from ..scenarios.s1_research_literature.task_validator import run_tasks


class S1Runner(BaseRunner):
    """
    ScenarioRunner for Scenario S1 — Research Literature Agent.

    Cycle state machine
    -------------------
    Cycle 0:
      eval_text = source_doc (held in runner; never written to SQLite)
      run probes  →  m(0)
      memory_policy.write(source_doc, llm)

    Cycles 1..N:
      [no_memory]        : eval_text = current batch only (paper_batches mode)
                           or "" (single-doc mode).
      [summarize_store]  : eval_text = memory_policy.read()
      run probes  →  m(k)
      memory_policy.write(eval_text, llm)
    """

    SCENARIO_ID = "s1_research_literature"

    def __init__(
        self,
        source_doc_text: str,
        probes: list[dict],
        validator_fn: Callable,
        memory_policy: MemoryPolicy,
        llm,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        tasks: Optional[list[dict]] = None,
        generated_data: dict | None = None,
        score_via_response: bool = False,
        agent_class=None,
        agent_max_turns: int = 8,
    ):
        self.source_doc_text = source_doc_text
        self.probes = probes
        self.validator_fn = validator_fn
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        # Wire cost tracking: every LLM call now auto-logs an llm_call trace
        # event (probes + compaction), so cost_and_efficiency block sums correctly.
        if self.llm is not None:
            self.llm.tracer = self.tracer
        self.sut_id = sut_id
        self.tasks = tasks or []
        self.score_via_response = score_via_response
        # When agent_class is None, helper LLM calls go through llm.chat
        # directly. When set (ReferenceAgent), they route through a per-
        # cycle agent with eval_text-aware tools.
        self.agent_class = agent_class
        self.agent_max_turns = agent_max_turns

        # Infer model info for trace logging
        self._model_id = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
        self._provider = "local_hf" if hasattr(llm, "tok") else "litellm"

        # Load session-specific facts and paper batches.
        # When generated_data is provided (--generated mode), use its output
        # so that different seeds actually produce different task streams.
        # Previously the runner ALWAYS loaded from disk, which meant the
        # generator's seed-dependent output was silently ignored.
        if generated_data:
            pb = generated_data.get("paper_batches", {})
            self.paper_batches = pb.get("batches", [])
            self.cross_cycle_queries = pb.get("cross_cycle_queries", [])
            sf = generated_data.get("session_facts", {})
            self.session_facts = sf.get("facts", []) if isinstance(sf, dict) else sf
            # Capture the dependency graph's per-fact version history so the
            # baseline-corrected revision-aging trident (revision_fidelity_excess,
            # stale_residue_rate; see scenarios.s1_research_literature.revision_metrics)
            # can be scored per-cycle against M_t.
            dg = generated_data.get("dependency_graph") or {}
            self.dep_graph_facts = dg.get("facts", {}) if isinstance(dg, dict) else {}
        else:
            # Curated mode: load from disk
            facts_path = Path(__file__).parent.parent / "scenarios" / "s1_research_literature" / "session_facts.json"
            if facts_path.exists():
                with open(facts_path) as f:
                    self.session_facts = json.load(f)["facts"]
            else:
                self.session_facts = []

            batches_path = Path(__file__).parent.parent / "scenarios" / "s1_research_literature" / "paper_batches.json"
            if batches_path.exists():
                with open(batches_path) as f:
                    batch_data = json.load(f)
                    self.paper_batches = batch_data["batches"]
                    self.cross_cycle_queries = batch_data.get("cross_cycle_queries", [])
            else:
                self.paper_batches = []
                self.cross_cycle_queries = []
            self.dep_graph_facts = {}

    def _build_responder(self, eval_text: str, cycle: int):
        """Per-cycle agent-routed responder, or None when agent_class is unset."""
        if self.agent_class is None:
            return None
        from ..core.tool_helpers import build_default_tool_registry

        registry = build_default_tool_registry(lambda: eval_text)

        agent = self.agent_class(
            llm=self.llm,
            memory_policy=self.memory_policy,
            tools=registry,
            max_turns=self.agent_max_turns,
        )

        def _respond(framed_query: str) -> str:
            result = agent.run_session(framed_query, session_id=cycle)
            return result.get("output", "") or ""

        return _respond

    def _probe_lag_recall(self, eval_text: str, cycle: int) -> dict:
        """
        Probe recall of session-specific facts from prior cycles.
        Returns {lag_k: recalled (0 or 1)} for each available lag distance.
        """
        recall_by_lag = {}
        text_lower = eval_text.lower()
        for fact in self.session_facts:
            fact_cycle = fact["cycle"]
            if fact_cycle >= cycle:
                continue  # only probe facts from prior cycles
            lag = cycle - fact_cycle
            recalled = any(kw.lower() in text_lower for kw in fact["keywords"])
            recall_by_lag[lag] = 1 if recalled else 0
        return recall_by_lag

    def run(self, n_cycles: int = 8, seed: int = 42) -> dict:
        """Run the S1 loop for n_cycles.

        Returns a dict with: keyword_curve, task_curve, lag_recall_curve,
        recall_matrix, session_results, keyword_raw, task_raw, lag_recall_raw.
        The CLI _run_s1 unpacks this so every measured signal is persisted
        in metrics.json (no per-cycle data is silently dropped).
        """
        import random as _random
        # Seed all sources of runtime randomness so multi-seed runs differ.
        # Previously seed was accepted but never used — with temperature=0
        # local models the entire execution was identical across seeds.
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
        progress_on = os.getenv("AGINGBENCH_S1_PROGRESS", "1").lower() not in {
            "0", "false", "no", "off"
        }

        run_t0 = time.time()

        def _fmt_elapsed(start: float) -> str:
            delta = int(time.time() - start)
            m, s = divmod(delta, 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h:02d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        def _progress(msg: str, cycle_start: float | None = None) -> None:
            if not progress_on:
                return
            run_elapsed = _fmt_elapsed(run_t0)
            if cycle_start is None:
                print(f"  [S1][progress][run {run_elapsed}] {msg}", flush=True)
                return
            cycle_elapsed = _fmt_elapsed(cycle_start)
            print(
                f"  [S1][progress][run {run_elapsed} | cycle {cycle_elapsed}] {msg}",
                flush=True,
            )

        exposures: list[int] = []
        scores: list[float] = []
        task_scores_by_cycle: list[float] = []
        # Lag-curve tracking: recall_matrix[cycle][lag] = 0/1
        recall_matrix: list[dict] = []
        # Per-cycle structured records (mirrors session_results in S2-S7)
        session_results: list[dict] = []

        run_span = self.tracer.log(
            "run_start",
            parent_span_id=None,
            sut_id=self.sut_id,
            scenario=self.SCENARIO_ID,
            seed=seed,
            n_cycles=n_cycles,
            policy=type(self.memory_policy).__name__,
            **{"gen_ai.request.model": self._model_id},
        )

        # Track all content for cycle-0 paper-batches initialization
        all_content_so_far = ""
        # Track per-cohort keywords for longitudinal decay measurement
        cohort_keywords: list[list[str]] = []  # cohort_keywords[cycle] = keywords introduced at that cycle

        _progress(
            f"starting run: cycles={n_cycles + 1}, policy={type(self.memory_policy).__name__}"
        )

        for cycle in range(n_cycles + 1):
            cycle_t0 = time.time()
            cycle_span = self.tracer.log("cycle_start", parent_span_id=run_span, cycle=cycle)
            _progress(f"cycle {cycle + 1}/{n_cycles + 1} start", cycle_t0)

            # ---- Longitudinal mode: add new paper batch ----
            new_batch_text = ""  # set below when a new paper batch is added
            if self.paper_batches and cycle < len(self.paper_batches):
                batch = self.paper_batches[cycle]
                new_content = f"\n\n--- {batch['title']} ---\n{batch['content']}"
                all_content_so_far += new_content
                cohort_keywords.append(batch.get("keywords", []))
            else:
                cohort_keywords.append([])

            # ---- Determine eval text ----
            # S1 is a text-survival benchmark: keyword_recall is computed over
            # eval_text, not agent reasoning output.
            #   no_memory: current batch only — no cross-session carry-over
            #   baseline:  compressed memory + current batch
            if cycle == 0:
                if self.paper_batches:
                    eval_text = all_content_so_far
                else:
                    eval_text = self.source_doc_text
            elif is_no_memory:
                # No cross-session carry-over. Only the current batch is visible;
                # keyword_m decays to the fraction of probe keywords that happen to
                # appear in this session's batch alone.
                if self.paper_batches and cycle < len(self.paper_batches):
                    eval_text = self.paper_batches[cycle]["content"]
                else:
                    eval_text = ""
            else:
                # Read compressed memory, then append new batch
                eval_text = self.memory_policy.read()
                if self.paper_batches and cycle < len(self.paper_batches):
                    batch = self.paper_batches[cycle]
                    new_batch_text = f"\n\n--- {batch['title']} ---\n{batch['content']}"
                    eval_text += new_batch_text

            # ---- Build the per-cycle responder once. None unless
            # agent_class is set; helpers fall back to direct llm.chat.
            responder = self._build_responder(eval_text, cycle)

            # ---- Snapshot metric: fraction of the fixed probe set that is
            # findable in the agent's current eval_text view. One binary
            # score per probe, averaged.
            probe_scores, m_snapshot = self.validator_fn(eval_text, self.probes)

            # ---- Faithful trend-probe scoring for revision-via-DAG.
            # The cumulative snapshot mixes recall and trend-probe scores.
            # Break out the trend-tagged probes so dependency_scorer can
            # compute version_accuracy from per-probe outcomes instead of
            # falling back to the session-wide keyword_m proxy. Probes carry
            # dep_type="trend" and forbidden_keywords=[<pre-revision value>];
            # the validator already penalizes those, so reusing probe_scores
            # here gives the faithful per-probe verdict.
            trend_probe_results = []
            trend_probes_this_cycle = []
            for probe, score in zip(self.probes, probe_scores):
                if probe.get("dep_type") == "trend":
                    trend_probe_results.append({
                        "probe_id": probe.get("probe_id"),
                        "score": float(score),
                        "score_memory": float(score),
                        "expected_keywords": probe.get("keywords", []),
                        "forbidden_keywords": probe.get("forbidden_keywords", []),
                    })
                    trend_probes_this_cycle.append(probe)

            # Response-based keyword scoring (opt-in): ask the agent each
            # non-trend probe and score the response. Memory-based keyword_m
            # above isolates W+R; this captures W+R+U end-to-end.
            #
            # Two filters make the comparison fair to memory-based scoring:
            # (1) ``ask_at_cycle <= cycle`` — only probe facts that have
            #     actually been introduced by this cycle. Before this gate,
            #     cycle-0 was scoring against all 45 probes including ones
            #     about cycle-9 facts the agent couldn't possibly answer,
            #     deflating m_response by ~0.5 at the start of the run.
            # (2) Score twice: once with ``enforce_forbidden=True`` (the
            #     deployed default — citing a stale value counts as a
            #     failure) and once with ``enforce_forbidden=False``. The
            #     unguarded value is the apples-to-apples counterpart of
            #     memory-based keyword_m, which doesn't have a forbidden
            #     penalty; the guarded value retains the revision-aware
            #     semantics needed by the trident.
            keyword_response_results = []
            keyword_m_response = None
            keyword_m_response_unguarded = None
            if self.score_via_response:
                kw_probes = [
                    p for p in self.probes
                    if not p.get("dep_type")
                    and p.get("ask_at_cycle", 0) <= cycle
                ]
                if kw_probes:
                    from agingbench.scenarios.s1_research_literature.task_validator import (
                        run_keyword_probes as _run_keyword_probes,
                    )
                    k_scores, k_details = _run_keyword_probes(
                        kw_probes, eval_text, self.llm,
                        responder=responder,
                    )
                    # Re-score the same responses with forbidden enforcement
                    # disabled, for symmetry with memory-based keyword_m.
                    # Avoids re-calling the LLM — we score the already-captured
                    # response text from k_details.
                    from agingbench.scenarios.s1_research_literature.validator import (
                        score_probe as _score_probe,
                    )
                    k_scores_unguarded = [
                        _score_probe(d["response"], probe, enforce_forbidden=False)
                        for probe, d in zip(kw_probes, k_details)
                    ]
                    for probe, score, detail in zip(kw_probes, k_scores, k_details):
                        keyword_response_results.append({
                            "probe_id": probe["probe_id"],
                            "score_response": float(score),
                            "response_preview": detail["response"],
                        })
                    keyword_m_response = sum(k_scores) / len(k_scores) if k_scores else 0.0
                    keyword_m_response_unguarded = (
                        sum(k_scores_unguarded) / len(k_scores_unguarded)
                        if k_scores_unguarded else 0.0
                    )

            # Response-based trend scoring (opt-in): actually ask the agent
            # each trend probe with the current memory as context. Faithfully
            # measures revision behavior; memory-based score above is the
            # cheap fallback.
            if self.score_via_response and trend_probes_this_cycle:
                from agingbench.scenarios.s1_research_literature.task_validator import (
                    run_trend_probes as _run_trend_probes,
                )
                r_scores, r_details = _run_trend_probes(
                    trend_probes_this_cycle, eval_text, self.llm,
                    responder=responder,
                )
                by_id = {d["probe_id"]: d for d in r_details}
                for tr in trend_probe_results:
                    d = by_id.get(tr["probe_id"])
                    if d:
                        tr["score"] = float(d["score"])
                        tr["score_response"] = float(d["score"])
                        tr["response_preview"] = d["response"]

            # ---- Longitudinal metric: fraction of *every* keyword ever
            # introduced across batches [0..cycle] that is still findable.
            # Denominator grows with sessions, so this tracks cumulative
            # compression-aging survival rather than per-session hit rate.
            m_longitudinal = None
            cohort_survived = 0
            cohort_total = 0
            if self.paper_batches:
                eval_lower = eval_text.lower()
                for c in range(cycle + 1):
                    for kw in cohort_keywords[c]:
                        cohort_total += 1
                        # Digit-flank-safe: a short numeric keyword like "201"
                        # must not match inside a longer number ("2013"), which
                        # would inflate cumulative survival. Word keywords keep
                        # plain substring semantics (guards are no-ops for them).
                        if _kw_in_text(kw.lower(), eval_lower):
                            cohort_survived += 1
                if cohort_total > 0:
                    m_longitudinal = cohort_survived / cohort_total

            # The headline curve uses the longitudinal metric when it is
            # available (paper_batches mode — S1's canonical scoring since
            # the 2025 rewrite) and falls back to the snapshot metric for
            # legacy single-doc runs. Both are stored separately in
            # session_results so downstream analysis can pick either.
            m = m_longitudinal if m_longitudinal is not None else m_snapshot

            exposures.append(cycle)
            scores.append(m)

            # Telemetry: surface BOTH metrics when --score-via-response is on,
            # so external dashboards / mid-run analysis can compare the
            # memory-text substring view (m) against the end-to-end W+R+U
            # response view (m_response) without having to wait for the
            # end-of-run metrics.json. None when score_via_response is off.
            probe_batch_attrs = {
                "cycle": cycle,
                "probe_scores": probe_scores,
                "m": round(m, 4),
                "n_probes": len(probe_scores),
                # §5.2 exposure axis: t_writes tracks long-term memory writes
                "t_writes": getattr(self.memory_policy, "n_writes", 0),
            }
            if keyword_m_response is not None:
                probe_batch_attrs["m_response"] = round(keyword_m_response, 4)
            if keyword_m_response_unguarded is not None:
                probe_batch_attrs["m_response_unguarded"] = round(
                    keyword_m_response_unguarded, 4
                )
            self.tracer.log(
                "probe_batch",
                parent_span_id=cycle_span,
                **probe_batch_attrs,
            )

            # ---- Run compliance decision tasks (task_m) ----
            task_m = 0.0
            if self.tasks:
                _, task_m, task_details = run_tasks(
                    self.tasks, eval_text, self.llm, responder=responder,
                )
                task_scores_by_cycle.append(task_m)
                self.tracer.log(
                    "task_batch",
                    parent_span_id=cycle_span,
                    cycle=cycle,
                    task_m=round(task_m, 4),
                    n_tasks=len(self.tasks),
                    t_writes=getattr(self.memory_policy, "n_writes", 0),
                )

            # ---- Lag-curve probing (§5b) ----
            lag_recall = {}
            if cycle > 0:
                eval_lower = eval_text.lower()
                if self.paper_batches:
                    # Probe keywords from each prior cohort. Use
                    # _kw_in_text (digit-flank-safe) so short numeric
                    # keywords like '2' don't trivially match inside
                    # larger numbers ('20', '2024', etc.) and inflate
                    # lag_recall. Matches the matching policy used by
                    # the longitudinal headline (keyword_m_longitudinal)
                    # — previously this used a plain substring check
                    # which silently double-counted single-digit
                    # numeric keywords.
                    for prior_c in range(cycle):
                        lag = cycle - prior_c
                        kws = cohort_keywords[prior_c]
                        if kws:
                            survived = sum(
                                1 for kw in kws if _kw_in_text(kw.lower(), eval_lower)
                            )
                            lag_recall[lag] = survived / len(kws)
                elif self.session_facts:
                    lag_recall = self._probe_lag_recall(eval_text, cycle)

                if lag_recall:
                    avg_recall = sum(lag_recall.values()) / len(lag_recall)
                    self.tracer.log(
                        "lag_probe", parent_span_id=cycle_span,
                        cycle=cycle, lag_recall=lag_recall,
                        avg_recall=round(avg_recall, 4),
                    )
            recall_matrix.append({"cycle": cycle, "lag_recall": lag_recall})

            # Build per-cycle structured record so the CLI can persist
            # everything S1 measures (mirrors session_results in S2-S7).
            # n_probes_passed is summed from probe_scores so individual probe
            # outcomes are not lost in the saved file.
            # Token-cap diagnostics: eval_text is the agent's memory readout
            # (the material against which keyword probes are scored).
            from agingbench.metrics.aging import count_response_tokens
            eval_tokens = count_response_tokens(self.llm, eval_text)

            # Baseline-corrected revision-aging trident (S3-ported). Scores
            # M_t against the dependency graph's per-fact version history,
            # using unrevised facts as the compression-aging baseline so the
            # residual signal isolates revision-specific decay. Robust to
            # phrasing (no agent involvement) — see revision_metrics.py.
            from agingbench.scenarios.s1_research_literature.revision_metrics import (
                score_revision_aging as _score_rev_aging,
            )
            revision_trident = _score_rev_aging(
                eval_text, self.dep_graph_facts, cycle
            ) if self.dep_graph_facts else {
                "revision_fidelity_excess": None,
                "stale_residue_rate": None,
                "coverage_verdict": "no_graph",
                "n_revised": 0, "n_unrevised": 0,
            }

            # Snapshot-vs-headline coupling: in the default (non-paper_batches)
            # path, keyword_m IS m_snapshot — emitting both is pure
            # redundancy. Only surface keyword_m_snapshot when the
            # longitudinal-cohort path is active (paper_batches=True), where
            # keyword_m becomes m_longitudinal and m_snapshot is the
            # genuinely-distinct snapshot diagnostic.
            _entry = {
                "session": cycle,
                # Headline — whichever of snapshot/longitudinal is active.
                "keyword_m": round(m, 4),
                # Per-probe trend results: each entry is the faithful
                # revision-via-DAG verdict for one trend probe in this
                # cycle. Empty list when no trend probes fired. Consumed
                # by dependency_scorer.version_accuracy.
                "trend_probe_results": trend_probe_results,
                "lag_recall_avg": round(
                    sum(lag_recall.values()) / len(lag_recall), 4
                ) if lag_recall else 1.0,
                "eval_text_len": len(eval_text),
                # eval_text is the agent's view of memory at this cycle. We
                # store a truncated copy so dependency_scorer.forget_accuracy
                # can scan it for invalidated keywords (S1 has no separate
                # task outputs — eval_text IS the agent-visible state).
                "task_outputs_text": eval_text[:10000],
                "response_tokens_max": eval_tokens,
                # Baseline-corrected revision-aging signals: compare survival
                # rates of revised vs unrevised facts in M_t at this cycle.
                # See agingbench.scenarios.s1_research_literature.revision_metrics
                # for the full method.
                "revision_aging": revision_trident,
            }
            # ─── Conditional emissions (omit when off, don't clutter) ───────
            # lag_recall per-lag breakdown only when there ARE prior cycles.
            # At cycle 0 the dict is empty — lag_recall_avg defaults to 1.0
            # which is the value to consume.
            if lag_recall:
                _entry["lag_recall"] = {int(k): round(v, 4) for k, v in lag_recall.items()}
            # Snapshot only when distinct from headline (longitudinal mode).
            if m_longitudinal is not None:
                _entry["keyword_m_snapshot"] = round(m_snapshot, 4)
                # cohort counts are the raw form of the longitudinal metric;
                # only meaningful in this mode.
                _entry["cohort_survived"] = cohort_survived
                _entry["cohort_total"] = cohort_total
            # Response-mode keyword recall only when explicitly enabled.
            # Two variants: the guarded form (forbidden-aware, deployed
            # default) and the unguarded form (no forbidden penalty —
            # symmetric with memory-based keyword_m, used to isolate the
            # pure U-stage gap from the revision-tracking failure mode).
            if self.score_via_response:
                _entry["keyword_m_response"] = (
                    round(keyword_m_response, 4)
                    if keyword_m_response is not None else None
                )
                _entry["keyword_m_response_unguarded"] = (
                    round(keyword_m_response_unguarded, 4)
                    if keyword_m_response_unguarded is not None else None
                )
                _entry["keyword_response_results"] = keyword_response_results
            # task_m only when tasks were actually loaded.
            if self.tasks:
                _entry["task_m"] = round(task_m, 4)
            session_results.append(_entry)

            self.tracer.log("cycle_end", parent_span_id=cycle_span, cycle=cycle,
                            m=round(m, 4), task_m=round(task_m, 4))

            lag_str = ""
            if lag_recall:
                avg_r = sum(lag_recall.values()) / len(lag_recall)
                lag_str = f"  lag_recall={avg_r:.3f}"
            task_str = f"  task_m={task_m:.3f}" if self.tasks else ""
            # Response-based scoring (--score-via-response): print alongside
            # memory-based m so the in-flight stdout makes the W+R-only vs
            # W+R+U comparison visible without having to dig into
            # session_results after the run completes. The _ung suffix is
            # the unguarded (no forbidden penalty) variant — apples-to-
            # apples with memory-based scoring. None when the flag is off,
            # in which case we emit nothing extra.
            resp_str = ""
            if keyword_m_response is not None:
                resp_str = f"  kw_m_resp={keyword_m_response:.3f}"
                if keyword_m_response_unguarded is not None:
                    resp_str += f"  kw_m_resp_ung={keyword_m_response_unguarded:.3f}"
            print(f"  [S1] Cycle {cycle:2d}  keyword_m={m:.3f}{resp_str}  "
                  f"({sum(probe_scores)}/{len(probe_scores)} probes){task_str}{lag_str}  "
                  f"text_len={len(eval_text)}")
            _progress(
                f"cycle {cycle + 1}/{n_cycles + 1} scored: keyword_m={m:.3f}"
                + (
                    f", kw_m_resp={keyword_m_response:.3f}"
                    if keyword_m_response is not None else ""
                ),
                cycle_t0,
            )

            if cycle == n_cycles:
                break

            # ---- Compression step ----
            if not is_no_memory:
                _progress(f"cycle {cycle + 1}: compression start", cycle_t0)
                if cycle == 0 or not new_batch_text:
                    write_payload = eval_text
                else:
                    write_payload = new_batch_text
                input_len = len(write_payload)
                self.memory_policy.write(write_payload, llm=self.llm)
                compressed = self.memory_policy.read()

                # Read token usage if the policy captured it
                in_tok = getattr(self.memory_policy, "last_input_tokens", 0)
                out_tok = getattr(self.memory_policy, "last_output_tokens", 0)
                if session_results:
                    session_results[-1]["memory_write_tokens"] = out_tok

                self.tracer.log_llm_call(
                    parent_span_id=cycle_span,
                    model=self._model_id,
                    provider=self._provider,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    input_preview=eval_text,
                    output_preview=compressed,
                    thought=getattr(self.llm, "last_thought", ""),
                    cycle=cycle,
                )
                self.tracer.log(
                    "compress",
                    parent_span_id=cycle_span,
                    cycle=cycle,
                    input_len=input_len,
                    output_len=len(compressed),
                    **{
                        "gen_ai.usage.input_tokens": in_tok,
                        "gen_ai.usage.output_tokens": out_tok,
                    },
                )
                _progress(
                    f"cycle {cycle + 1}: compression done (in_tok={in_tok}, out_tok={out_tok})",
                    cycle_t0,
                )

            _progress(f"cycle {cycle + 1}/{n_cycles + 1} end", cycle_t0)

        keyword_curve = AgingCurve(
            exposures=exposures, scores=scores,
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        task_curve = AgingCurve(
            exposures=exposures[:len(task_scores_by_cycle)],
            scores=task_scores_by_cycle,
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        ) if task_scores_by_cycle else None

        # Build lag-recall curve: average recall at each cycle
        lag_recall_scores = []
        for entry in recall_matrix:
            lr = entry["lag_recall"]
            if lr:
                lag_recall_scores.append(sum(lr.values()) / len(lr))
            else:
                lag_recall_scores.append(1.0)  # no prior facts to forget at cycle 0

        lag_recall_curve = AgingCurve(
            exposures=exposures[:len(lag_recall_scores)],
            scores=lag_recall_scores,
            scenario=self.SCENARIO_ID,
            sut_id=self.sut_id,
        )

        self.tracer.log(
            "run_end",
            parent_span_id=run_span,
            keyword_curve=list(zip(keyword_curve.exposures, keyword_curve.scores)),
            task_curve=list(zip(task_curve.exposures, task_curve.scores)) if task_curve else [],
            recall_matrix=[e["lag_recall"] for e in recall_matrix],
            lag_recall_curve=list(zip(lag_recall_curve.exposures, lag_recall_curve.scores)),
            half_life=compute_half_life(keyword_curve),
            slope=round(compute_decay_slope(keyword_curve), 5),
            m0=scores[0] if scores else None,
            m_final=scores[-1] if scores else None,
            task_m0=task_scores_by_cycle[0] if task_scores_by_cycle else None,
            task_m_final=task_scores_by_cycle[-1] if task_scores_by_cycle else None,
        )
        _progress(f"run complete: m_final={scores[-1] if scores else 0:.3f}")

        return {
            "keyword_curve": keyword_curve,
            "task_curve": task_curve,
            "lag_recall_curve": lag_recall_curve,
            "recall_matrix": recall_matrix,
            "session_results": session_results,
            "keyword_raw": list(zip(exposures, scores)),
            "task_raw": list(zip(exposures[:len(task_scores_by_cycle)],
                                  task_scores_by_cycle)) if task_scores_by_cycle else [],
            "lag_recall_raw": list(zip(exposures[:len(lag_recall_scores)],
                                        lag_recall_scores)),
        }


# Backward compatibility alias
P2Runner = S1Runner
