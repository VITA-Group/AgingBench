"""
agingbench/runner/s5_runner.py — S5 Self-Planning Notebook evaluator.

The agent manages its own workspace files. The benchmark only controls:
  (1) what tasks to send (via S5Generator task stream)
  (2) when to reset conversation state (every session_length interactions)
  (3) how to score responses (keyword match + workspace inspection)

Works with ANY AgentAdapter — Claude Code, ReactFileAdapter, Codex, custom.

Aging mechanisms measured:
  - Planning complexity (more facts = harder reasoning)
  - Recall precision (correct fact among competitors)
  - Interference (similar facts confuse each other)
  - Update propagation (corrections not applied everywhere)
  - Decision fragility (inconsistency grows with state)
  - Planning fatigue (same task degrades over time)
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

from .base import BaseRunner, RunResult
from .trace import TraceLogger
from ..core.agent_adapter import AgentAdapter, AgentResponse
from ..metrics.aging import AgingCurve, summarize
from ..metrics.g1_metrics import compute_keyword_survival
from ..metrics.deep_tier_metrics import (
    compute_overwrite_loss_rate,
    compute_workspace_fidelity,
    compute_cohort_keyword_m,
    compute_interference_rate,
    compute_conflation_rate,
    compute_specificity_score,
    compute_version_aware_ufr,
    compute_reasoning_depth,
    compute_source_attribution_accuracy,
)


class S5Runner(BaseRunner):
    """Self-Planning evaluator (Tier 1 with workspace-file access)."""

    SCENARIO_ID = "s5_self_planning"

    def __init__(
        self,
        adapter: AgentAdapter,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        session_length: int = 12,
        generated_data: dict | None = None,
        n_recall_probes_per_session: int = 5,
        reset_history: bool = True,
        maintenance_events: list | None = None,
    ):
        self.adapter = adapter
        self.reset_history = reset_history
        self.tracer = tracer
        # Wire cost tracking through the adapter's inner LLM (ReactFileAdapter etc.).
        # If the adapter wraps a BaseLLM, every chat call now logs an llm_call
        # trace event so the AgingCard cost block populates correctly. AgentAdapters
        # that subprocess out (e.g. ClaudeCodeAgentAdapter) have no Python LLM and
        # require per-adapter cost-capture work — tracked separately.
        _adapter_llm = getattr(self.adapter, "llm", None)
        if _adapter_llm is not None and hasattr(_adapter_llm, "tracer"):
            _adapter_llm.tracer = self.tracer
        self.sut_id = sut_id
        self.session_length = session_length
        self.generated_data = generated_data or {}
        self.n_recall_probes = n_recall_probes_per_session
        self.maintenance_events = maintenance_events or []

    def _find_workspace(self) -> Path | None:
        """Find the workspace directory from the adapter."""
        for attr in ('cwd', '_cwd', 'workspace_dir', '_workspace_dir'):
            val = getattr(self.adapter, attr, None)
            if val:
                p = Path(val)
                if p.exists():
                    return p
        return None

    def _setup_workspace(self):
        """Create CLAUDE.md and notes/ in the workspace to guide the agent."""
        workspace = self._find_workspace()
        if workspace is None:
            return

        claude_md = workspace / "CLAUDE.md"
        if not claude_md.exists():
            claude_md.write_text(
                "# Personal Assistant Memory\n\n"
                "You are a long-running personal assistant. The user will share "
                "personal information across many conversations.\n\n"
                "IMPORTANT: Save all user preferences, facts, budgets, dates, names, "
                "and constraints to files in the `notes/` directory. Each topic should "
                "have its own file (e.g., `notes/dining.md`, `notes/contacts.md`).\n\n"
                "At the start of each conversation, read your notes to recall prior context.\n"
            )

        notes_dir = workspace / "notes"
        notes_dir.mkdir(exist_ok=True)

    def run(self, n_sessions: int = 10, seed: int = 42) -> RunResult:
        rng = random.Random(seed)

        task_stream = self.generated_data.get("task_stream", {})
        tasks = task_stream.get("tasks", [])
        recall_probes = self.generated_data.get("recall_probes", {}).get("probes", [])
        facts_registry = self.generated_data.get("facts_registry", [])

        if not tasks:
            raise ValueError("No tasks in generated_data. Run S5Generator first.")

        # Set up workspace with CLAUDE.md + notes/ before first interaction
        self._setup_workspace()

        self.tracer.log("run_start", sut_id=self.sut_id, scenario="s5_self_planning",
                        n_sessions=n_sessions, session_length=self.session_length)

        session_results = []
        task_scores_by_session = []
        recall_matrix: dict[int, dict[int, float]] = {}  # [block][lag] -> accuracy
        workspace_snapshots = []

        global_interaction = 0

        for block in range(n_sessions):
            # Apply maintenance events scheduled for this block.
            # S7 is Tier-2 (agent-managed workspace), so maintenance events
            # operate on workspace FILES rather than on a MemoryPolicy object.
            # Supported event types:
            #   workspace_flush   — delete all notes/ files (simulates storage loss)
            #   workspace_recompact — merge all notes/* into one memory.md
            #   memory_compaction   — falls back to MemoryPolicy.apply if available
            for event in self.maintenance_events:
                if event.session == block:
                    shock_type = event.event_type
                    ws_dir = self._find_workspace()
                    if ws_dir and shock_type == "workspace_flush":
                        notes = ws_dir / "notes"
                        if notes.exists():
                            import shutil
                            n_deleted = sum(1 for _ in notes.iterdir())
                            shutil.rmtree(notes)
                            notes.mkdir()
                            self.tracer.log(
                                "life_event", session=block,
                                event_type="workspace_flush",
                                files_deleted=n_deleted,
                                params=event.params,
                            )
                            print(f"  [S7] === MAINTENANCE: workspace_flush at block {block} ({n_deleted} files deleted) ===")
                    elif ws_dir and shock_type == "workspace_recompact":
                        notes = ws_dir / "notes"
                        if notes.exists():
                            all_text = []
                            for f in sorted(notes.iterdir()):
                                if f.is_file():
                                    all_text.append(f"## {f.name}\n{f.read_text()}")
                                    f.unlink()
                            merged = ws_dir / "notes" / "memory.md"
                            merged.write_text("\n\n".join(all_text))
                            self.tracer.log(
                                "life_event", session=block,
                                event_type="workspace_recompact",
                                files_merged=len(all_text),
                                params=event.params,
                            )
                            print(f"  [S7] === MAINTENANCE: workspace_recompact at block {block} ({len(all_text)} files → memory.md) ===")
                    else:
                        # Generic fallback — try memory policy dispatch
                        ws_policy = getattr(self.adapter, '_workspace_policy', None)
                        if ws_policy:
                            shock_type = event.apply(ws_policy)
                        self.tracer.log("life_event", session=block,
                                        event_type=shock_type, params=event.params)

            self.tracer.log("session_start", session=block)
            t0 = time.time()

            # Optionally reset conversation (keep workspace files)
            if self.reset_history:
                self.adapter.reset_session()

            # Get tasks for this block
            block_tasks = [t for t in tasks if t.get("session_block") == block]

            block_scores = []
            block_tool_calls = []
            block_files_changed = []
            block_responses = []  # track response text per task
            interference_responses = []  # for interference metric
            block_task_scores: dict[str, float] = {}  # task_id → score (for drift scorer)

            # Execute tasks
            for task in block_tasks:
                response = self.adapter.send_message(task["prompt"])
                global_interaction += 1

                # Score based on task type
                score = self._score_response(response.text, task)
                block_scores.append(score)
                block_responses.append(response.text)
                block_tool_calls.extend(response.tool_calls)
                block_files_changed.extend(response.files_changed)
                block_task_scores[task["id"]] = score

                # Track interference task responses
                if task.get("type") == "interference":
                    interference_responses.append({
                        "target_keywords": task.get("target_keywords", []),
                        "competitor_keywords": task.get("competitor_keywords", []),
                        "response_text": response.text,
                    })

                self.tracer.log("task_completed",
                    session=block,
                    task_id=task["id"],
                    task_type=task["type"],
                    score=score,
                    turns=response.metadata.get("turns", 0),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    prompt=task["prompt"],
                    response_text=response.text[:2000],  # cap at 2K chars
                    files_changed=response.files_changed,
                )

            # Run sampled recall probes — split into primed (explicit file-read
            # instruction) and spontaneous (just the question, tests proactive recall)
            available_probes = [p for p in recall_probes if p["available_after_block"] < block]
            if available_probes:
                sampled = self._sample_recall_probes(available_probes, block, rng)
                probe_scores = []
                primed_scores = []
                spontaneous_scores = []

                for i, probe in enumerate(sampled):
                    # Alternate: even-indexed probes are spontaneous, odd are primed
                    is_spontaneous = (i % 2 == 0)

                    if is_spontaneous:
                        # Spontaneous: just ask — tests whether agent proactively reads files
                        probe_prompt = probe["question"]
                    else:
                        # Primed: explicit file-read instruction — tests file-write fidelity
                        probe_prompt = (
                            "Before answering, read ALL files in the notes/ directory. "
                            "The answer is in your saved notes. Then answer this question:\n\n"
                            + probe["question"]
                        )

                    probe_response = self.adapter.send_message(probe_prompt)
                    probe["_resp_text"] = probe_response.text
                    global_interaction += 1

                    p_score = 1.0 if any(
                        kw.lower() in probe_response.text.lower()
                        for kw in probe["keywords"]
                    ) else 0.0
                    probe_scores.append(p_score)
                    if is_spontaneous:
                        spontaneous_scores.append(p_score)
                    else:
                        primed_scores.append(p_score)

                    lag = block - probe["available_after_block"]
                    if block not in recall_matrix:
                        recall_matrix[block] = {}
                    recall_matrix[block][lag] = recall_matrix[block].get(lag, [])
                    if isinstance(recall_matrix[block][lag], list):
                        recall_matrix[block][lag].append(p_score)

                    self.tracer.log("recall_probe",
                        session=block,
                        probe_id=probe["id"],
                        lag=lag,
                        score=p_score,
                        probe_type="spontaneous" if is_spontaneous else "primed",
                        prompt=probe_prompt[:200],
                        response_text=probe_response.text[:2000],
                        keywords=probe["keywords"],
                    )

                recall_accuracy = sum(probe_scores) / len(probe_scores) if probe_scores else 1.0
                primed_recall = (sum(primed_scores) / len(primed_scores)) if primed_scores else None
                spontaneous_recall = (sum(spontaneous_scores) / len(spontaneous_scores)) if spontaneous_scores else None
            else:
                recall_accuracy = 1.0
                probe_scores = []
                primed_recall = None
                spontaneous_recall = None

            # Workspace inspection (transparent adapters)
            ws_state = self.adapter.get_workspace_state()
            memory_text = self.adapter.get_memory_text()
            workspace_snapshots.append({
                "session": block,
                "n_files": len(ws_state.get("files", [])),
                "total_bytes": ws_state.get("total_bytes", 0),
            })

            # Compute keyword_m on workspace (if transparent)
            keyword_m = None
            overwrite_loss = None
            ws_fidelity = None
            cohort_kw = None
            if memory_text:
                relevant_facts = [f for f in facts_registry
                                  if f.get("session_block", f.get("introduced_at_block", 999)) <= block]
                if relevant_facts:
                    survived = sum(
                        1 for f in relevant_facts
                        if any(kw.lower() in memory_text.lower() for kw in f.get("keywords", []))
                    )
                    keyword_m = survived / len(relevant_facts)

                # Self-planning metrics: overwrite loss, workspace fidelity, cohort keyword_m
                # Tag facts with introduced_at_block if not already set
                tagged_facts = []
                for f in facts_registry:
                    ff = dict(f)
                    if "introduced_at_block" not in ff:
                        ff["introduced_at_block"] = ff.get("session_block", 0)
                    tagged_facts.append(ff)

                overwrite_loss = compute_overwrite_loss_rate(
                    tagged_facts, memory_text, block,
                )
                ws_fidelity = compute_workspace_fidelity(
                    tagged_facts, memory_text, block,
                )
                cohort_kw = compute_cohort_keyword_m(
                    tagged_facts, memory_text, block,
                )

            # ── Accumulation-complexity metrics ──
            # Interference rate
            interf_rate = compute_interference_rate(interference_responses) if interference_responses else None

            # Specificity score (average across task responses with gold keywords)
            spec_scores = []
            for bt, br in zip(block_tasks, block_responses):
                gold_kws = bt.get("eval_keywords", [])
                if gold_kws and br:
                    spec_scores.append(compute_specificity_score(br, gold_kws))
            avg_specificity = sum(spec_scores) / len(spec_scores) if spec_scores else None

            # Reasoning depth (on cross-reference and plan tasks)
            depth_scores = []
            for bt, br in zip(block_tasks, block_responses):
                if bt.get("type") in ("recall_compare", "plan") and br:
                    ref_facts = bt.get("references_facts", [])
                    if ref_facts:
                        req = [{"keywords": f.get("keywords", [])}
                               for f in facts_registry if f["id"] in ref_facts]
                        if req:
                            depth_scores.append(compute_reasoning_depth(br, req))
            avg_reasoning_depth = sum(depth_scores) / len(depth_scores) if depth_scores else None

            # Source attribution (on recall tasks that have source info)
            attrib_scores = []
            for bt, br in zip(block_tasks, block_responses):
                if bt.get("type") == "recall_precise" and br:
                    ref_ids = bt.get("references_facts", [])
                    for fid in ref_ids:
                        fact = next((f for f in facts_registry if f["id"] == fid), None)
                        if fact and fact.get("keywords"):
                            # Person name is typically the second keyword
                            who_kws = [fact["keywords"][1]] if len(fact["keywords"]) > 1 else []
                            gold_src = {"who_keywords": who_kws}
                            attrib_scores.append(compute_source_attribution_accuracy(br, gold_src))
            avg_attribution = sum(attrib_scores) / len(attrib_scores) if attrib_scores else None

            # Version-aware UFR from recall probes
            va_ufr = None
            if probe_scores and memory_text:
                ufr_data = []
                for i, (probe, p_score) in enumerate(zip(sampled, probe_scores)):
                    ufr_data.append({
                        "keywords": probe.get("keywords", []),
                        "response_text": probe.get("_resp_text", ""),
                    })
                if ufr_data:
                    va_ufr = compute_version_aware_ufr(ufr_data, memory_text)

            # Task accuracy for this block
            task_accuracy = sum(block_scores) / len(block_scores) if block_scores else 1.0
            task_scores_by_session.append(task_accuracy)

            # Proactive check rate (file reads before answers)
            n_file_reads_before = sum(
                1 for tc in block_tool_calls
                if tc.get("tool") in ("read_file", "list_files")
            )
            n_tasks = len(block_tasks)
            proactive_rate = n_file_reads_before / max(n_tasks, 1)

            elapsed = time.time() - t0
            # Concatenate task + recall probe responses (truncated) so
            # dependency_scorer.forget_accuracy can scan for invalidated keywords.
            task_outputs_text = " ".join(
                str(r)[:300] for r in block_responses if r
            )

            # Token-cap diagnostics
            from agingbench.metrics.aging import count_response_tokens
            llm_for_tok = getattr(self.adapter, "llm", None)
            response_token_counts = [
                count_response_tokens(llm_for_tok, str(r)) if llm_for_tok else -1
                for r in block_responses if r
            ]
            valid_resp_tokens = [t for t in response_token_counts if t >= 0]

            session_result = {
                "session": block,
                "task_accuracy": task_accuracy,
                "recall_accuracy": recall_accuracy,
                "response_tokens_per_task": response_token_counts,
                "response_tokens_max": max(valid_resp_tokens) if valid_resp_tokens else 0,
                "primed_recall": primed_recall,
                "spontaneous_recall": spontaneous_recall,
                "task_scores": block_task_scores,
                "keyword_m": keyword_m,
                "overwrite_loss_rate": overwrite_loss,
                "workspace_fidelity": ws_fidelity,
                "cohort_keyword_m": cohort_kw,
                "proactive_check_rate": proactive_rate,
                # Accumulation-complexity metrics
                "interference_rate": interf_rate,
                "specificity_score": avg_specificity,
                "reasoning_depth": avg_reasoning_depth,
                "source_attribution": avg_attribution,
                "version_aware_ufr": va_ufr,
                # Workspace state
                "n_files_in_workspace": ws_state.get("files", []),
                "workspace_bytes": ws_state.get("total_bytes", 0),
                "n_tool_calls": len(block_tool_calls),
                "files_changed": block_files_changed,
                "elapsed_s": elapsed,
                "task_outputs_text": task_outputs_text,
            }
            session_results.append(session_result)

            self.tracer.log("session_end", session=block,
                task_accuracy=task_accuracy,
                recall_accuracy=recall_accuracy,
                keyword_m=keyword_m,
                overwrite_loss_rate=overwrite_loss,
                workspace_fidelity=ws_fidelity,
                cohort_keyword_m=cohort_kw,
                proactive_rate=proactive_rate,
                n_workspace_files=len(ws_state.get("files", [])),
            )

            ol_str = f"{overwrite_loss:.3f}" if overwrite_loss is not None else "n/a"
            wf_str = f"{ws_fidelity:.3f}" if ws_fidelity is not None else "n/a"
            km_str = f"{keyword_m:.3f}" if keyword_m is not None else "n/a"
            print(f"  [S7] Block {block:2d}  task={task_accuracy:.3f}  "
                  f"recall={recall_accuracy:.3f}  "
                  f"kw_m={km_str}  loss={ol_str}  ws_fid={wf_str}  "
                  f"files={len(ws_state.get('files', []))}  "
                  f"({elapsed:.0f}s)")

        # Aggregate recall_matrix: convert lists to means
        recall_matrix_agg = {}
        for block, lags in recall_matrix.items():
            recall_matrix_agg[block] = {}
            for lag, scores in lags.items():
                if isinstance(scores, list):
                    recall_matrix_agg[block][lag] = sum(scores) / len(scores)
                else:
                    recall_matrix_agg[block][lag] = scores

        # Build lag curves
        lag_curves = {}
        for block, lags in recall_matrix_agg.items():
            for lag, score in lags.items():
                if lag not in lag_curves:
                    lag_curves[lag] = []
                lag_curves[lag].append((block, score))

        # Build aging curves
        # Derive exposures from the actual number of scored sessions, not
        # n_sessions: if a run terminates early (adapter error) session_results
        # is shorter, and a hardcoded range(n_sessions) would desync the
        # exposures/scores zip in compute_decay_slope. Identical to range(
        # n_sessions) on a complete run.
        recall_scores = [sr["recall_accuracy"] for sr in session_results]
        primary_curve = AgingCurve(
            exposures=list(range(len(recall_scores))),
            scores=recall_scores,
            scenario="s5_self_planning",
            sut_id=self.sut_id,
        )

        task_scores = [sr["task_accuracy"] for sr in session_results]
        task_curve = AgingCurve(
            exposures=list(range(len(task_scores))),
            scores=task_scores,
            scenario="s5_self_planning",
            sut_id=self.sut_id,
        )

        self.tracer.log("run_end", sut_id=self.sut_id)

        return RunResult(
            primary_curve=primary_curve,
            session_results=session_results,
            secondary_curves={
                "task_accuracy": task_curve,
            },
            raw={
                "recall_matrix": recall_matrix_agg,
                "lag_curves": lag_curves,
                "workspace_snapshots": workspace_snapshots,
                "recall_raw": [(i, s) for i, s in enumerate(recall_scores)],
                "task_raw": [(i, s) for i, s in enumerate(task_scores)],
            },
        )

    def _score_response(self, response_text: str, task: dict) -> float:
        """Score an agent response against task eval_keywords."""
        keywords = task.get("eval_keywords", [])
        if not keywords:
            return 1.0  # plan tasks without keywords scored as 1 (qualitative)

        matched = sum(
            1 for kw in keywords
            if kw.lower() in response_text.lower()
        )
        return matched / len(keywords)

    def _sample_recall_probes(
        self, available: list[dict], current_block: int, rng: random.Random
    ) -> list[dict]:
        """Sample recall probes at varying lag distances."""
        if len(available) <= self.n_recall_probes:
            return available

        # Group by lag distance
        by_lag: dict[int, list[dict]] = {}
        for probe in available:
            lag = current_block - probe["available_after_block"]
            by_lag.setdefault(lag, []).append(probe)

        # Target lag distances: 1, 2, 4, 8, max
        target_lags = sorted(set([1, 2, 4, 8, max(by_lag.keys())]))
        sampled = []

        for target_lag in target_lags:
            if len(sampled) >= self.n_recall_probes:
                break
            # Find closest available lag
            closest_lag = min(by_lag.keys(), key=lambda l: abs(l - target_lag))
            candidates = by_lag[closest_lag]
            if candidates:
                probe = rng.choice(candidates)
                if probe not in sampled:
                    sampled.append(probe)
                    candidates.remove(probe)

        # Fill remaining slots randomly
        remaining = [p for p in available if p not in sampled]
        while len(sampled) < self.n_recall_probes and remaining:
            probe = rng.choice(remaining)
            sampled.append(probe)
            remaining.remove(probe)

        return sampled
