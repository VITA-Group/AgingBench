"""
agingbench/runner/runner.py — S1 ScenarioRunner (formerly P2).

Implements the S1 cycle state machine for the Research Literature Agent
scenario. Tracks memory_bloat per cycle (G3-M2). Uses structured trace
logging with OpenInference-aligned field names including token counts.
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
from ..metrics.g3_metrics import compute_memory_bloat
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
                           or "" (single-doc mode). Consistent with S2/S4/S6
                           where no_memory means "no cross-session carry-over".
                           For the oracle-source upper bound, use --oracle-mode.
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
        oracle_mode: bool = False,
        oracle_retrieval: bool = False,
        oracle_store: bool = False,
        incontext_ceiling: bool = False,
        ceiling_max_tokens: int = 100_000,
        generated_data: dict | None = None,
        score_via_response: bool = False,
    ):
        # Back-compat alias: legacy oracle_mode -> oracle_store under the
        # new clean attribution framework (§5.2). In S1 the behavioral
        # difference between C3 and C4 is minor because S1 scores text
        # survival against an eval_text rather than agent reasoning output,
        # but the provenance still differs.
        if oracle_mode and not oracle_store:
            oracle_store = True
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
        self.oracle_mode = oracle_mode
        self.oracle_retrieval = oracle_retrieval   # C2
        self.oracle_store = oracle_store           # C3
        self.incontext_ceiling = incontext_ceiling # C4
        self.ceiling_max_tokens = ceiling_max_tokens
        self.score_via_response = score_via_response

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
        recall_matrix, session_results, keyword_raw, task_raw, lag_recall_raw,
        bloat_raw. The CLI _run_s1 unpacks this so every measured signal is
        persisted in metrics.json (no per-cycle data is silently dropped).
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
        bloat_series: list[int] = []
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
            oracle_mode=self.oracle_mode,
            **{"gen_ai.request.model": self._model_id},
        )

        # Track all content for oracle mode
        all_content_so_far = ""
        # Track per-cohort keywords for longitudinal decay measurement
        cohort_keywords: list[list[str]] = []  # cohort_keywords[cycle] = keywords introduced at that cycle

        _progress(
            f"starting run: cycles={n_cycles + 1}, policy={type(self.memory_policy).__name__}, "
            f"oracle={self.oracle_mode}"
        )

        for cycle in range(n_cycles + 1):
            cycle_t0 = time.time()
            cycle_span = self.tracer.log("cycle_start", parent_span_id=run_span, cycle=cycle)
            _progress(f"cycle {cycle + 1}/{n_cycles + 1} start", cycle_t0)

            # ---- Longitudinal mode: add new paper batch ----
            if self.paper_batches and cycle < len(self.paper_batches):
                batch = self.paper_batches[cycle]
                new_content = f"\n\n--- {batch['title']} ---\n{batch['content']}"
                all_content_so_far += new_content
                cohort_keywords.append(batch.get("keywords", []))
            else:
                cohort_keywords.append([])

            # ---- Determine eval text ----
            # S1 is a text-survival benchmark: keyword_recall is computed over
            # eval_text, not agent reasoning output. Under the clean C_i
            # framework (§5.2):
            #   C0 (is_no_memory)    : current batch only — no cross-session carry-over
            #   C1 (baseline)        : compressed memory + current batch
            #   C2 (oracle_retrieval): ABSTAIN in S1. S1's single-blob memory
            #                          has no separate retrieve step, so C2
            #                          cannot be instrumented distinctly from
            #                          C3. Aliased to C3 content at runtime;
            #                          flagged c2_abstain_s1=True in the run
            #                          output so downstream analysis/plots can
            #                          honestly merge the two cells instead of
            #                          pretending they were measured apart.
            #   C3 (oracle_store)    : full clean raw source
            #   C4 (incontext_ceiling): full raw source tail-truncated to
            #                          ceiling_max_tokens * 4 chars (matches
            #                          the S2/S4/S6 _read_c4 convention). When
            #                          the corpus fits, C4 equals C3; when the
            #                          budget bites, old cohorts drop first —
            #                          which is exactly what the ceiling
            #                          measures.
            if cycle == 0:
                if self.paper_batches:
                    eval_text = all_content_so_far
                else:
                    eval_text = self.source_doc_text
            elif self.incontext_ceiling:
                _raw = all_content_so_far if self.paper_batches else self.source_doc_text
                _max_chars = max(1, self.ceiling_max_tokens) * 4
                eval_text = _raw if len(_raw) <= _max_chars else _raw[-_max_chars:]
            elif self.oracle_store or self.oracle_retrieval:
                # C3 and C2-abstain: full clean raw source.
                eval_text = all_content_so_far if self.paper_batches else self.source_doc_text
            elif is_no_memory:
                # C0: no cross-session carry-over. Only the current batch is visible;
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
                    eval_text += f"\n\n--- {batch['title']} ---\n{batch['content']}"

            # ---- Track memory_bloat (G3-M2) ----
            bloat = compute_memory_bloat(eval_text)
            bloat_series.append(bloat)

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
            keyword_response_results = []
            keyword_m_response = None
            if self.score_via_response:
                kw_probes = [p for p in self.probes if not p.get("dep_type")]
                if kw_probes:
                    from agingbench.scenarios.s1_research_literature.task_validator import (
                        run_keyword_probes as _run_keyword_probes,
                    )
                    k_scores, k_details = _run_keyword_probes(
                        kw_probes, eval_text, self.llm,
                    )
                    for probe, score, detail in zip(kw_probes, k_scores, k_details):
                        keyword_response_results.append({
                            "probe_id": probe["probe_id"],
                            "score_response": float(score),
                            "response_preview": detail["response"],
                        })
                    keyword_m_response = sum(k_scores) / len(k_scores) if k_scores else 0.0

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
                        if kw.lower() in eval_lower:
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

            self.tracer.log(
                "probe_batch",
                parent_span_id=cycle_span,
                cycle=cycle,
                probe_scores=probe_scores,
                m=round(m, 4),
                n_probes=len(probe_scores),
                memory_bloat=bloat,
                # §5.2 exposure axis: t_writes tracks long-term memory writes
                t_writes=getattr(self.memory_policy, "n_writes", 0),
            )

            # ---- Run compliance decision tasks (task_m) ----
            task_m = 0.0
            if self.tasks:
                _, task_m, task_details = run_tasks(self.tasks, eval_text, self.llm)
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
                    # Probe keywords from each prior cohort
                    for prior_c in range(cycle):
                        lag = cycle - prior_c
                        kws = cohort_keywords[prior_c]
                        if kws:
                            survived = sum(1 for kw in kws if kw.lower() in eval_lower)
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

            session_results.append({
                "session": cycle,
                # Headline — whichever of snapshot/longitudinal is active.
                "keyword_m": round(m, 4),
                # Snapshot: fraction of the fixed probe set findable in
                # eval_text right now. Binary per probe.
                "keyword_m_snapshot": round(m_snapshot, 4),
                "probe_based_passed": int(sum(probe_scores)),
                "probe_based_total": len(probe_scores),
                # Per-probe trend results: each entry is the faithful
                # revision-via-DAG verdict for one trend probe in this
                # cycle. Empty list when no trend probes fired. Consumed
                # by dependency_scorer.version_accuracy.
                "trend_probe_results": trend_probe_results,
                # End-to-end W+R+U keyword recall (None when not enabled).
                "keyword_m_response": (
                    round(keyword_m_response, 4) if keyword_m_response is not None else None
                ),
                "keyword_response_results": keyword_response_results,
                # Longitudinal: cumulative cohort-keyword survival (None
                # when paper_batches is off).
                "keyword_m_longitudinal": (
                    round(m_longitudinal, 4) if m_longitudinal is not None else None
                ),
                "cohort_survived": cohort_survived if self.paper_batches else None,
                "cohort_total": cohort_total if self.paper_batches else None,
                "task_m": round(task_m, 4),
                "memory_bloat": bloat,
                "lag_recall": {int(k): round(v, 4) for k, v in lag_recall.items()},
                "lag_recall_avg": round(
                    sum(lag_recall.values()) / len(lag_recall), 4
                ) if lag_recall else 1.0,
                "eval_text_len": len(eval_text),
                # eval_text is the agent's view of memory at this cycle. We
                # store a truncated copy so dependency_scorer.forget_accuracy
                # can scan it for invalidated keywords (S1 has no separate
                # task outputs — eval_text IS the agent-visible state).
                "task_outputs_text": eval_text[:1500],
                "response_tokens_max": eval_tokens,
            })

            self.tracer.log("cycle_end", parent_span_id=cycle_span, cycle=cycle,
                            m=round(m, 4), task_m=round(task_m, 4),
                            memory_bloat=bloat)

            lag_str = ""
            if lag_recall:
                avg_r = sum(lag_recall.values()) / len(lag_recall)
                lag_str = f"  lag_recall={avg_r:.3f}"
            task_str = f"  task_m={task_m:.3f}" if self.tasks else ""
            print(f"  [S1] Cycle {cycle:2d}  keyword_m={m:.3f}  "
                  f"({sum(probe_scores)}/{len(probe_scores)} probes){task_str}{lag_str}  "
                  f"bloat={bloat}  text_len={len(eval_text)}")
            _progress(
                f"cycle {cycle + 1}/{n_cycles + 1} scored: keyword_m={m:.3f}",
                cycle_t0,
            )

            if cycle == n_cycles:
                break

            # ---- Compression step ----
            # Skipped under C3 (oracle_store) and C4 (incontext_ceiling):
            # both paths bypass the SUT's memory_policy write. C2
            # (oracle_retrieval) still writes (agent "believes" it's storing
            # normally; eval_text uses the oracle source above).
            _uses_sut_policy = (
                not is_no_memory
                and not self.oracle_mode
                and not self.oracle_store
                and not self.incontext_ceiling
            )
            if _uses_sut_policy:
                _progress(f"cycle {cycle + 1}: compression start", cycle_t0)
                input_len = len(eval_text)
                self.memory_policy.write(eval_text, llm=self.llm)
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
            bloat_series=bloat_series,
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

        # Attribution provenance stamp. attribution_mode is kept stable
        # across runners for downstream filtering; the S1-specific
        # c2_abstain_s1 flag (emitted in the return dict below) tells
        # readers that oracle_retrieval in S1 was aliased to oracle_store.
        if self.incontext_ceiling:
            _attr_mode = "c4_incontext_ceiling"
        elif self.oracle_store:
            _attr_mode = "c3_oracle_store"
        elif self.oracle_retrieval:
            _attr_mode = "c2_oracle_retrieval"
        else:
            _attr_mode = "c1_baseline"

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
            "bloat_raw": list(zip(exposures, bloat_series)),
            "attribution_schema": "v2_clean",
            "attribution_mode": _attr_mode,
            # True when oracle_retrieval was requested on S1 — the run used
            # oracle_store semantics because S1's single-blob memory has no
            # distinct retrieve step. Plots/tables should merge C2 and C3 for
            # S1 when this flag is set.
            "c2_abstain_s1": bool(self.oracle_retrieval),
            "ceiling_max_tokens": (
                self.ceiling_max_tokens if self.incontext_ceiling else None
            ),
        }


# Backward compatibility alias
P2Runner = S1Runner
