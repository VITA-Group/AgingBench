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
import re
import time
from pathlib import Path
from typing import Optional


# Max characters of an agent response considered when keyword-scoring.
# Defends against "memory dump everything at the top of every response" agent
# strategies that would otherwise score 1.0 by mention alone, regardless of
# whether the answer is focused on the asked question. We score the LAST
# `_S5_SCORE_TAIL_CHARS` of the response (typically where the final answer
# lives) plus the FIRST `_S5_SCORE_HEAD_CHARS` to allow concise answers that
# lead with the answer. The combined window captures focused answers without
# rewarding fact-dump strategies that bury everything in the middle.
_S5_SCORE_HEAD_CHARS = 1500
_S5_SCORE_TAIL_CHARS = 2500


def _s5_scoring_window(response_text: str) -> str:
    """Return the substring of ``response_text`` that should be keyword-scored.

    For short responses (≤ HEAD+TAIL) returns the full text. For long
    responses returns ``head + " ... " + tail`` so the focused answer area
    is preserved without giving credit to middle-of-response dump content.
    """
    text = response_text or ""
    if len(text) <= _S5_SCORE_HEAD_CHARS + _S5_SCORE_TAIL_CHARS:
        return text
    return text[:_S5_SCORE_HEAD_CHARS] + " ... " + text[-_S5_SCORE_TAIL_CHARS:]


def _s5_kw_present(keyword: str, lower_text: str) -> bool:
    """Word-boundary-aware presence check (mirrors S6/S1 fix).

    Allows optional `s`/`es` plural suffix so common English plurals
    (e.g. "amazons", "addresses") still match the singular keyword.
    """
    kw = (keyword or "").lower().strip()
    if not kw:
        return False
    return re.search(
        r"(?<![A-Za-z0-9])"
        + re.escape(kw)
        + r"(?:es|s)?"
        + r"(?![A-Za-z0-9])",
        lower_text,
    ) is not None


def _s5_active_keywords(probe: dict, block: int) -> list:
    """Keywords valid for a recall probe at ``block``.

    When the probe's source fact was revised, the generator records a
    ``keywords_history`` = [(orig_block, old_kws), (update_block, new_kws), …];
    we pick the latest entry whose block is <= ``block`` (old value before the
    update, new value after). Without this the probe scores the agent against
    the stale pre-revision value forever — rewarding staleness and penalizing a
    correct update. Falls back to ``probe["keywords"]`` when no history exists.
    """
    hist = probe.get("keywords_history")
    if not hist:
        return probe.get("keywords", [])
    active = probe.get("keywords", [])
    for sess, kws in hist:
        if sess <= block:
            active = kws
        else:
            break
    return active


def _s5_is_mechanics_failure(text: str) -> bool:
    """True when an adapter response is a tool-loop breakage rather than a real
    answer: the ReactFileAdapter sentinel ``"(max turns reached)"`` or an
    ``"ERROR:"`` tool-error observation. Such responses score 0 on every
    keyword, which would masquerade as recall aging — so they are excluded from
    recall_accuracy and reported separately as mechanics_failure_rate."""
    t = (text or "").strip()
    return t == "(max turns reached)" or t.lstrip().startswith("ERROR:")

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
        binding_probes = self.generated_data.get("binding_probes", [])
        facts_registry = self.generated_data.get("facts_registry", [])

        # Forced binding probes (gold-vs-distractor) are re-asked at a small
        # fixed set of lags after the pair is injected, so the cost stays
        # bounded regardless of n_sessions. Each (probe, lag) is asked through
        # the agent's OWN workspace tool-loop (same send_message path recall
        # probes use), forcing the agent to recover the gold value from its
        # own notes; citing the distractor is a binding failure.
        # Lags are taken from pressure.confusable_probe_lags if the generator
        # propagated it into generated_data; otherwise a small fixed default.
        binding_probe_lags = (
            self.generated_data.get("confusable_probe_lags") or [1, 3]
        )
        # Interference-binding source. "topic" (opt-in via
        # PressureConfig.confusable_topic_matched) reuses the same-category
        # competitor probe S5 already asks each block — drawn from S5's own
        # personal facts (topic-matched) and per-block dense, at NO extra agent
        # calls. "generic" (default) keeps the business-pool binding probes
        # asked at lags, preserving reproducibility of prior runs.
        interference_binding_source = self.generated_data.get(
            "interference_binding_source", "generic"
        )

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
                        "task_id": task.get("id"),
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
                    response_text=response.text[:10000],
                    files_changed=response.files_changed,
                )

            # Run sampled recall probes — split into primed (explicit file-read
            # instruction) and spontaneous (just the question, tests proactive recall)
            available_probes = [p for p in recall_probes if p["available_after_block"] < block]
            if available_probes:
                sampled = self._sample_recall_probes(available_probes, block, rng)
                probe_scores = []          # raw 0/1, aligned with `sampled`
                mech_flags = []            # tool-loop-failure flag, aligned with `sampled`
                spont_flags = []           # is_spontaneous, aligned with `sampled`

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

                    # Word-boundary match within a bounded scoring window
                    # (see _s5_scoring_window for rationale).
                    _scored_text = _s5_scoring_window(probe_response.text).lower()
                    # Use the value valid at THIS block: if the source fact was
                    # revised, score against the post-update keywords (revision
                    # faithfulness) rather than the frozen original.
                    _active_kw = _s5_active_keywords(probe, block)
                    mech = _s5_is_mechanics_failure(probe_response.text)
                    p_score = 1.0 if any(
                        _s5_kw_present(kw, _scored_text)
                        for kw in _active_kw
                    ) else 0.0
                    probe_scores.append(p_score)
                    mech_flags.append(mech)
                    spont_flags.append(is_spontaneous)

                    lag = block - probe["available_after_block"]
                    if block not in recall_matrix:
                        recall_matrix[block] = {}
                    recall_matrix[block][lag] = recall_matrix[block].get(lag, [])
                    # Exclude tool-loop failures from the lag matrix too — they
                    # are not recall evidence.
                    if not mech and isinstance(recall_matrix[block][lag], list):
                        recall_matrix[block][lag].append(p_score)

                    self.tracer.log("recall_probe",
                        session=block,
                        probe_id=probe["id"],
                        lag=lag,
                        score=p_score,
                        mechanics_failure=mech,
                        probe_type="spontaneous" if is_spontaneous else "primed",
                        prompt=probe_prompt[:200],
                        response_text=probe_response.text[:10000],
                        keywords=probe["keywords"],
                    )

                # Aggregate over GENUINE answers only: a tool-loop crash
                # ("(max turns reached)" / "ERROR:") scores 0 on every keyword,
                # which would masquerade as recall aging — so it is excluded from
                # recall_accuracy and surfaced separately as mechanics_failure_rate.
                valid = [ps for ps, mf in zip(probe_scores, mech_flags) if not mf]
                primed_valid = [ps for ps, mf, sp in zip(probe_scores, mech_flags, spont_flags)
                                if not mf and not sp]
                spont_valid = [ps for ps, mf, sp in zip(probe_scores, mech_flags, spont_flags)
                               if not mf and sp]
                # Down-weight, don't deflate: mean over genuine answers; falls
                # back to the no-measurable-probe semantics (1.0) only when there
                # is nothing genuine to score — in which case mechanics_failure_rate
                # flags the session as unmeasured.
                recall_accuracy = (sum(valid) / len(valid)) if valid else 1.0
                primed_recall = (sum(primed_valid) / len(primed_valid)) if primed_valid else None
                spontaneous_recall = (sum(spont_valid) / len(spont_valid)) if spont_valid else None
                mechanics_failure_rate = (sum(mech_flags) / len(mech_flags)) if mech_flags else None
            else:
                recall_accuracy = 1.0
                probe_scores = []
                primed_recall = None
                spontaneous_recall = None
                mechanics_failure_rate = None

            # ---- Forced binding probes (interference, measured by default) ----
            # For each injected confusable pair, re-ask the gold-vs-distractor
            # probe at a small fixed set of lags through the agent's own
            # workspace tool-loop (same send_message path recall probes use).
            # The agent must recover the gold value from its own notes; citing
            # the distractor is a binding failure. These feed only
            # score_interference_binding — they do NOT touch recall_accuracy.
            interference_probe_results = []

            def _mech_fail(txt: str) -> bool:
                # Adapter sentinel / tool-error observation: a tool-loop
                # breakage, NOT a confusable mis-binding. Flagged so aging
                # analysis can filter it. The scorer ignores unknown fields.
                return (txt.strip() == "(max turns reached)"
                        or txt.lstrip().startswith("ERROR:"))

            if interference_binding_source == "topic":
                # TOPIC-MATCHED, zero extra agent calls: reuse the same-category
                # competitor probe S5 already asked THIS block (drawn from S5's
                # own personal facts). gold = the target fact's value; distractor
                # = a same-category competitor's value. Per-block dense.
                for ir in interference_responses:
                    tgt = ir.get("target_keywords") or []
                    comp = ir.get("competitor_keywords") or []
                    if not tgt or not comp:
                        continue  # need both a gold and a confusable distractor
                    resp_text = ir.get("response_text", "") or ""
                    interference_probe_results.append({
                        "session": block,
                        "task_id": ir.get("task_id"),
                        "question": "(same-category competitor probe)",
                        "response_text": resp_text,
                        "gold_value": tgt[0],
                        "distractor_value": comp[0],
                        "mechanics_failure": _mech_fail(resp_text),
                    })
            else:
                # GENERIC (default): re-ask each injected business-pool pair's
                # binding probe at a small fixed set of lags through the agent's
                # own workspace tool-loop. Citing the distractor is a binding
                # failure. Feeds only score_interference_binding (not recall).
                for bp in binding_probes:
                    # Due at injection_block + lag for each configured lag. If NO
                    # lag lands inside the run horizon (injected too late — e.g. a
                    # short run), fall back to the final block so each pair is
                    # scored at least once (n_probes > 0), matching S3/S4. Cost is
                    # bounded: a pair is asked at most len(lags) times.
                    inj = bp["available_after_block"]
                    reachable = [inj + lag for lag in binding_probe_lags
                                 if lag > 0 and inj + lag < n_sessions]
                    due = (block in reachable) if reachable else (block == n_sessions - 1)
                    if not due:
                        continue
                    resp = self.adapter.send_message(bp["question"])
                    resp_text = resp.text or ""
                    mech_fail = _mech_fail(resp_text)
                    interference_probe_results.append({
                        "session": block,
                        "task_id": bp["probe_id"],
                        "question": bp["question"],
                        "response_text": resp_text,
                        "gold_value": bp["gold_value"],
                        "distractor_value": bp["distractor_value"],
                        "mechanics_failure": mech_fail,
                    })
                    global_interaction += 1
                    self.tracer.log("interference_binding_probe",
                        session=block,
                        probe_id=bp["probe_id"],
                        question=bp["question"][:200],
                        response_text=resp_text[:10000],
                        gold_value=bp["gold_value"],
                        distractor_value=bp["distractor_value"],
                        mechanics_failure=mech_fail,
                    )

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
                # Fraction of recall probes this block whose response was a
                # tool-loop breakage (adapter sentinel / ERROR), excluded from
                # recall_accuracy. Lets analysis filter mechanics from aging.
                "mechanics_failure_rate": mechanics_failure_rate,
                "task_scores": block_task_scores,
                "keyword_m": keyword_m,
                "overwrite_loss_rate": overwrite_loss,
                "workspace_fidelity": ws_fidelity,
                "cohort_keyword_m": cohort_kw,
                "proactive_check_rate": proactive_rate,
                # Forced binding probes (gold-vs-distractor) for
                # score_interference_binding (interference measured by default).
                "interference_probes": interference_probe_results,
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
                    # A bucket can be empty when every probe at this (block, lag)
                    # was a tool-loop failure and thus excluded — skip it rather
                    # than divide by zero (no genuine recall evidence here).
                    if not scores:
                        continue
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
        """Score an agent response against task eval_keywords.

        Uses word-boundary matching (so "234" doesn't match inside "$2340")
        and restricts the scored text to a head+tail window (so a memory-dump
        strategy that floods the middle with all known facts doesn't trivially
        score 1.0 on every task).
        """
        keywords = task.get("eval_keywords", [])
        if not keywords:
            return 1.0  # plan tasks without keywords scored as 1 (qualitative)

        scored_text = _s5_scoring_window(response_text).lower()
        matched = sum(
            1 for kw in keywords
            if _s5_kw_present(kw, scored_text)
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
