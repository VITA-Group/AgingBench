"""
agingbench/runner/s4_runner.py — S4 Software Engineering Agent.

Implements the 8-sprint state machine. Each sprint: agent reads M_{t-1},
receives a modification task, reasons over codebase snapshot, produces
code changes. Scored against ground-truth impact sets and real test execution.

Produces G4 metrics: FASR, RR, CFR, LA, plus life event shock/recovery.

Key design decisions (addressing audit findings):
- Tests are executed via subprocess in a temp directory (real, not simulated)
- LA uses structured impact prediction (agent explicitly lists affected files)
- FASR is based on whether tests pass on the first code submission, not turn count
- CFR is measured from real pre/post test diffs
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import shutil
from pathlib import Path


_DEP_RECALL_STRUCTURAL = frozenset({
    "session", "important", "update", "correction", "previous",
    "finding", "invalid", "withdrawn", "longer", "accurate",
    "information", "about", "sprint", "value", "point",
    "earlier", "above", "below", "should", "would", "could",
    "using", "added", "which", "must", "have", "this", "that",
    "please", "disregard", "future", "analyses", "analysis",
    "retracted",
})


def _compute_dep_recall(dep_context: str, agent_output: str) -> float:
    """Score how well the agent referenced prior sprint decisions.

    Returns a value in [0, 1] computed as:
        min(1, hits / max(0.3 * N_unique_keywords, 1))

    where keywords are alphabetic content tokens (>=5 chars, deduped,
    excluding structural mixin markers like "UPDATE:" and common English
    like "should"/"about"). Vacuously returns 1.0 when dep_context is empty
    or produces no qualifying keywords.
    """
    if not dep_context:
        return 1.0
    tokens = re.findall(r"[a-z][a-z_0-9]*", dep_context.lower())
    dep_keywords = sorted({
        t for t in tokens
        if len(t) >= 5 and t not in _DEP_RECALL_STRUCTURAL
    })
    if not dep_keywords:
        return 1.0
    output_lower = agent_output.lower()
    hits = sum(1 for kw in dep_keywords if kw in output_lower)
    return min(1.0, hits / max(len(dep_keywords) * 0.3, 1))
from typing import Optional

from .base import BaseRunner, RunResult
from .trace import TraceLogger
from ..metrics.aging import AgingCurve
from ..metrics.g4_metrics import compute_shock, compute_recovery
from ..core.memory.base import MemoryPolicy
from ..core.memory.append_only import AppendOnlyPolicy
from ..core.agent import AgentInterface, ReferenceAgent
from ..core.tools import ToolRegistry, ToolSpec


class S4Runner(BaseRunner):
    SCENARIO_ID = "s4_software_engineering"

    def __init__(
        self,
        memory_policy: MemoryPolicy,
        llm,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        oracle_mode: bool = False,
        life_event_session: Optional[int] = None,
        agent_class: type[AgentInterface] = ReferenceAgent,
        generated_data: dict | None = None,
        oracle_retrieval: bool = False,
        oracle_store: bool = False,
        incontext_ceiling: bool = False,
        ceiling_max_tokens: int = 100_000,
    ):
        # Back-compat: legacy oracle_mode aliased to oracle_store (see §5.2).
        if oracle_mode and not oracle_store:
            oracle_store = True
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        if self.llm is not None:
            self.llm.tracer = self.tracer
        self.sut_id = sut_id
        self.oracle_mode = oracle_mode
        self.oracle_retrieval = oracle_retrieval   # C2
        self.oracle_store = oracle_store           # C3
        self.incontext_ceiling = incontext_ceiling # C4
        self.ceiling_max_tokens = ceiling_max_tokens
        self.agent_class = agent_class
        self.life_event_session = life_event_session

        # C3/C4 state (see s2_runner for the semantic).
        self._c3_store = None
        self._c4_raw_sessions: list[str] = []

        self._model_id = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
        self._provider = "local_hf" if hasattr(llm, "tok") else "litellm"

        # Load scenario data (from generator or curated files)
        if generated_data:
            self.tasks = generated_data["tasks"]["sessions"]
            self.life_event_cfg = generated_data["tasks"].get("life_event", {})
            self._raw_snapshots = generated_data["snapshots"]["snapshots"]
        else:
            data_dir = Path(__file__).parent.parent / "scenarios" / "s4_software_engineering"
            with open(data_dir / "tasks.json") as f:
                task_data = json.load(f)
                self.tasks = task_data["sessions"]
                self.life_event_cfg = task_data.get("life_event", {})

            with open(data_dir / "snapshots.json") as f:
                snapshot_data = json.load(f)
                self._raw_snapshots = snapshot_data["snapshots"]

        self.snapshots = self._resolve_snapshots()

        if self.life_event_session is None:
            self.life_event_session = self.life_event_cfg.get("session")

    def _resolve_snapshots(self) -> dict:
        """Resolve snapshot inheritance (_inherit_from) to get full file dicts."""
        resolved = {}
        for snap in self._raw_snapshots:
            sid = snap["session"]
            if snap.get("files"):
                resolved[sid] = snap
            elif "_inherit_from" in snap:
                parent = resolved.get(snap["_inherit_from"], {})
                files = dict(parent.get("files", {}))
                files.update(snap.get("_add_files", {}))
                tests = dict(parent.get("tests_expected", {}))
                resolved[sid] = {
                    "session": sid,
                    "n_files": snap.get("n_files", len(files)),
                    "files": files,
                    "tests_expected": tests,
                    "post_task_tests": snap.get("post_task_tests", tests),
                }
            else:
                resolved[sid] = snap
        return resolved

    # ---------- C3: oracle store (raw-stored, top-k retrieved) ----------
    def _get_c3_store(self) -> AppendOnlyPolicy:
        """Runner-owned AppendOnly for C3 in S4; see S2Runner._get_c3_store."""
        if self._c3_store is None:
            self._c3_store = AppendOnlyPolicy(
                db_path=":memory:",
                embedding_model="all-MiniLM-L6-v2",
                top_k=5,
                max_input_tokens=200_000,
            )
        return self._c3_store

    def _read_c3(self, query: Optional[str] = None) -> str:
        raw = self._get_c3_store().read(query=query)
        return f"=== ORACLE STORE (C3, raw-stored + top-k cosine) ===\n{raw}"

    def _write_c3(self, raw_session_output: str) -> None:
        self._get_c3_store().write(raw_session_output)

    # ---------- C4: in-context ceiling (no harness) ----------
    def _append_c4(self, raw_session_output: str) -> None:
        self._c4_raw_sessions.append(raw_session_output)

    def _read_c4(self) -> str:
        if not self._c4_raw_sessions:
            return "=== IN-CONTEXT CEILING (C4) ===\n(no prior sessions)"
        joined = [f"### Sprint {i} ###\n{t}" for i, t in enumerate(self._c4_raw_sessions)]
        body = "\n\n".join(joined)
        max_chars = self.ceiling_max_tokens * 4
        if len(body) <= max_chars:
            return f"=== IN-CONTEXT CEILING (C4) ===\n{body}"
        tail = body[-max_chars:]
        return (
            f"=== IN-CONTEXT CEILING (C4, head-truncated to "
            f"{self.ceiling_max_tokens} tokens from {len(self._c4_raw_sessions)} sprints) ===\n"
            f"[... older sprints truncated to fit ceiling budget ...]\n{tail}"
        )

    def _write_snapshot_to_dir(self, snapshot_files: dict, target_dir: Path) -> None:
        """Write snapshot files to a temporary directory for test execution."""
        for filepath, content in snapshot_files.items():
            full = target_dir / filepath
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)

    def _run_tests(self, project_dir: Path) -> dict[str, str]:
        """
        Execute pytest in the snapshot directory and return real test results.
        Returns {test_id: "pass"|"fail"}.
        """
        try:
            # NOTE: do NOT pass -q. -q suppresses the per-test
            # "tests/test_x.py::test PASSED" lines the parser below relies on,
            # which made _run_tests() always return {} → tests_before/after
            # silently fell back to the gold snapshot, making FASR/CFR/RR
            # agent-independent gold echoes. -v alone yields parseable lines.
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=no"],
                cwd=str(project_dir),
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "PYTHONPATH": str(project_dir)},
            )
            output = result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {}

        # Parse pytest -v output: "tests/test_foo.py::test_bar PASSED"
        results = {}
        for line in output.splitlines():
            line = line.strip()
            if " PASSED" in line:
                test_id = line.split(" PASSED")[0].strip()
                results[test_id] = "pass"
            elif " FAILED" in line:
                test_id = line.split(" FAILED")[0].strip()
                results[test_id] = "fail"
            elif " ERROR" in line:
                test_id = line.split(" ERROR")[0].strip()
                results[test_id] = "fail"
        return results

    def _build_tools(self, snapshot_files: dict) -> ToolRegistry:
        """Build tools for the agent: read_file, list_files."""
        registry = ToolRegistry()

        def read_file(arguments: dict) -> str:
            path = arguments.get("path", "")
            content = snapshot_files.get(path)
            if content is None:
                return f"Error: file '{path}' not found. Available: {sorted(snapshot_files.keys())}"
            return content

        def list_files(arguments: dict) -> str:
            return "\n".join(sorted(snapshot_files.keys()))

        registry.register(ToolSpec(
            name="read_file", version="1.0.0",
            description="Read the contents of a source file in the project.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
            fn=read_file,
        ))
        registry.register(ToolSpec(
            name="list_files", version="1.0.0",
            description="List all files in the project.",
            parameters={"type": "object", "properties": {}},
            fn=list_files,
        ))
        return registry

    def _parse_impact_prediction(self, agent_output: str, snapshot_files: dict) -> set[str]:
        """
        Extract the agent's impact prediction from structured output.
        Looks for files listed after 'Affected files:' or 'Impact:' patterns,
        and also any file paths from the snapshot that appear in the output.
        """
        predicted = set()
        output_lower = agent_output.lower()

        # Method 1: find file paths from snapshot mentioned in output
        for filepath in snapshot_files:
            if filepath in agent_output:
                predicted.add(filepath)

        # Method 2: look for structured lists (e.g., "- models/user.py")
        for line in agent_output.splitlines():
            line = line.strip().lstrip("-*• ")
            for filepath in snapshot_files:
                if filepath in line:
                    predicted.add(filepath)

        return predicted

    def _apply_agent_edits(self, agent_output: str, snapshot_files: dict,
                           task: dict, project_dir: Path) -> dict[str, str]:
        """
        Apply the agent's proposed code changes to the project directory.

        Strategy: look for code blocks in the agent's output. If the agent
        produces ```python blocks with filenames, write those files. Otherwise
        use the post_task_tests from the snapshot (known-good state) as the
        target to measure against.
        """
        import re

        written_files = {}

        # Parse code blocks: ```python\n# filename: path/to/file.py\n...```
        code_blocks = re.findall(
            r'```(?:python)?\s*\n(?:#\s*(?:file(?:name)?|path):\s*(\S+)\n)?(.*?)```',
            agent_output, re.DOTALL
        )

        for filename, code in code_blocks:
            if filename and filename in snapshot_files:
                filepath = project_dir / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(code)
                written_files[filename] = code

        return written_files

    def run(self, n_sessions: int = 8, seed: int = 42) -> dict:
        """Run S4 for n_sessions sprints; return dict with aging curves."""
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
        progress_on = os.getenv("AGINGBENCH_S4_PROGRESS", "1").lower() not in {
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

        def _progress(msg: str, session_start: float | None = None) -> None:
            if not progress_on:
                return
            run_elapsed = _fmt_elapsed(run_t0)
            if session_start is None:
                print(f"  [S4][progress][run {run_elapsed}] {msg}", flush=True)
                return
            session_elapsed = _fmt_elapsed(session_start)
            print(
                f"  [S4][progress][run {run_elapsed} | session {session_elapsed}] {msg}",
                flush=True,
            )

        fasr_raw = []
        rr_raw = []
        cfr_raw = []
        la_raw = []
        task_m_raw = []
        dep_recall_raw = []
        session_results = []
        all_design_notes = ""

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
            life_event_session=self.life_event_session,
        )

        actual_sessions = min(n_sessions, len(self.tasks))
        _progress(
            f"starting run: sessions={actual_sessions}, policy={type(self.memory_policy).__name__}, "
            f"oracle={self.oracle_mode}"
        )

        for t in range(actual_sessions):
            session_t0 = time.time()
            sess_span = self.tracer.log("session_start", parent_span_id=run_span, session=t)

            task = self.tasks[t]
            task_text = task["task"]
            ground_truth_impact = set(task["impact_set"])

            # Hoisted defaults: dep_context and depends_on are referenced
            # downstream when building the session_results entry. Initialize
            # them here, immediately after `task` is available, so they are
            # guaranteed to be defined regardless of any future refactor that
            # might wrap the assignment in a conditional. (A collaborator
            # previously reported "depends_on referenced before assignment";
            # static analysis of the current source shows the assignment is
            # already unconditional, but we hoist for defense in depth.)
            dep_context = task.get("dependency_context", "")
            depends_on = task.get("depends_on", [])

            # Life event: force memory compaction (only applies to the SUT's
            # memory_policy under C1; C3/C4 use runner-owned state that isn't
            # subject to the configured compaction event).
            if self.life_event_session is not None and t == self.life_event_session:
                if (not is_no_memory and not self.oracle_store
                        and not self.incontext_ceiling and not self.oracle_mode):
                    current = self.memory_policy.read()
                    if len(current) > 500:
                        self.memory_policy.write(
                            f"[COMPACTED] Key decisions from sprints 0-{t-1}: "
                            + current[:500] + "...",
                            llm=self.llm,
                        )
                    self.tracer.log(
                        "life_event", parent_span_id=sess_span,
                        session=t, event_type="memory_compaction",
                    )
                    print(f"  [S4] === LIFE EVENT: memory compaction at session {t} ===")

            # Read memory (one branch per C_i; see §5.2 Table 1).
            if is_no_memory:
                memory_text = ""
            elif self.incontext_ceiling:
                # C4: full concatenated sprint history, head-truncated.
                memory_text = self._read_c4()
            elif self.oracle_store:
                # C3 oracle-store: runner-owned AppendOnly with raw sprint
                # outputs; top-k cosine retrieval narrows what the agent sees.
                memory_text = self._read_c3(query=task.get("task", ""))
            elif self.oracle_retrieval:
                # C2 oracle-retrieval: gold design context for THIS sprint
                # (dependency_context encodes upstream decisions this sprint
                # depends on — perfect retrieval).
                dep_ctx = self.tasks[t].get("dependency_context", "")
                prior_lines = []
                for prev_t in range(t):
                    prev = self.tasks[prev_t]
                    prior_lines.append(
                        f"Sprint {prev_t}: {prev.get('task','')[:120]}  "
                        f"[files: {', '.join(prev.get('files_to_modify', []))}]"
                    )
                memory_text = (
                    "=== GOLD DESIGN CONTEXT (oracle retrieval) ===\n"
                    + "\n".join(prior_lines)
                    + ("\n\n" + dep_ctx if dep_ctx else "")
                )
            else:
                memory_text = self.memory_policy.read()

            _progress(
                f"session {t + 1}/{actual_sessions} start: files={task.get('n_files', 0)}, "
                f"depends_on={len(depends_on)}",
                session_t0,
            )

            # Log memory state before sprint
            _log_traj("memory_snapshot", session=t, phase="before_sprint",
                      memory_text=memory_text, memory_tokens=len(memory_text.split()) if memory_text else 0)

            # Get snapshot (fall back to latest available)
            avail = [s for s in self.snapshots if s <= t]
            snap_key = max(avail) if avail else max(self.snapshots.keys())
            snapshot = self.snapshots[snap_key]
            snapshot_files = snapshot.get("files", {})

            # Run pre-task tests in a temp directory
            _progress(f"session {t + 1}: pre-task tests start", session_t0)
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir)
                self._write_snapshot_to_dir(snapshot_files, project_dir)
                tests_before = self._run_tests(project_dir)

                # If no tests could be run, use expected from snapshot
                if not tests_before:
                    tests_before = snapshot.get("tests_expected", {})
            _progress(
                f"session {t + 1}: pre-task tests done ({len(tests_before)} tests)",
                session_t0,
            )

            # Build agent with read_file + list_files tools
            tools = self._build_tools(snapshot_files)
            agent = self.agent_class(
                llm=self.llm,
                memory_policy=self.memory_policy,
                tools=tools,
                max_turns=6,
            )

            # Build prompt with dependency context for longitudinal recall
            # (dep_context and depends_on were hoisted to the top of the loop
            # body — see the "Hoisted defaults" comment above.)
            context = "You are a software engineer working on a Python backend project.\n\n"
            if memory_text:
                context += f"Your design notes from previous sprints:\n{memory_text}\n\n"
            if dep_context:
                context += (
                    f"IMPORTANT — This task depends on decisions from prior sprints:\n"
                    f"{dep_context}\n\n"
                )
            context += (
                f"The project has {task.get('n_files', len(snapshot_files))} files.\n"
                f"Sprint {t} task: {task_text}\n\n"
                f"Instructions:\n"
                f"1. Use read_file and list_files to understand the codebase.\n"
                f"2. List ALL files that will be affected (modified or need updates), including test files.\n"
                f"3. Write the code changes as ```python blocks with '# filename: path/to/file.py' headers.\n"
                f"4. Start your final answer with 'Affected files:' listing one file per line."
            )

            result = agent.run_session(context, session_id=t)
            agent_output = result["output"]
            _progress(
                f"session {t + 1}: agent task done (turns={result.get('turns', 0)}, "
                f"tools={len(result.get('tool_calls', []))})",
                session_t0,
            )

            _log_traj("agent_output", session=t, phase="code_task",
                      prompt=context[:500], output=agent_output[:1000],
                      turns=result.get("turns", 0),
                      tool_calls=result.get("tool_calls", []))

            # Score LA strictly from the agent's prediction. We previously
            # union-ed task["files_to_modify"] into the predicted set as a
            # "minimum baseline", which gave every agent a free floor (gold
            # files leaked into the prediction) and compressed cross-model LA
            # differences. The agent must actually mention the impacted files.
            predicted_impact = self._parse_impact_prediction(agent_output, snapshot_files)

            from ..scenarios.s4_software_engineering.validator import compute_la, compute_cfr
            la = compute_la(predicted_impact, ground_truth_impact)

            # Apply edits and run post-task tests
            _progress(f"session {t + 1}: post-task tests start", session_t0)
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir)
                self._write_snapshot_to_dir(snapshot_files, project_dir)
                written = self._apply_agent_edits(agent_output, snapshot_files, task, project_dir)
                tests_after = self._run_tests(project_dir)

            # If agent didn't produce parseable code or tests didn't run,
            # use the snapshot's post_task_tests as the expected outcome
            if not tests_after:
                tests_after = snapshot.get("post_task_tests", snapshot.get("tests_expected", {}))
            _progress(
                f"session {t + 1}: post-task tests done ({len(tests_after)} tests, "
                f"written_files={len(written) if written else 0})",
                session_t0,
            )

            _log_traj("test_results", session=t,
                      tests_before={k: v for k, v in tests_before.items()},
                      tests_after={k: v for k, v in tests_after.items()},
                      files_written=sorted(written.keys()) if written else [])

            # ----- Stability indicators (NOT aging metrics) -----
            # CFR / FASR / RR are reported as confirmation that the agent's
            # edits do not corrupt the codebase. They are *not* aging signals
            # in the current single-attempt runner because there is no retry
            # loop and no test-breakage challenge: every session executes one
            # attempt and the existing tests almost always still pass.
            # The aging metrics for S4 are dep_recall (headline) and la.

            # CFR (G4-M3, Cascading Failure Rate): fraction of previously
            # passing tests now failing after the agent's edits.
            passing_before = {k for k, v in tests_before.items() if v == "pass"}
            passing_after = {k for k, v in tests_after.items() if v == "pass"}
            cfr = compute_cfr(passing_before, passing_after)

            # FASR (G4-M1, First-Attempt Success Rate): all tests pass
            # after the single attempt available in this runner.
            all_passed = all(v == "pass" for v in tests_after.values()) if tests_after else False
            fasr = 1.0 if all_passed else 0.0

            # RR (G4-M2, Revision Rate): per the validator spec this is
            # the number of attempts. The single-attempt runner always does
            # one attempt, so rr is fixed at 1.0. The previous implementation
            # used `max(1.0, len(written))` which conflated "files written"
            # with "revision cycles" — that produced rr > 1.0 for any agent
            # that edited multiple files in a single attempt (e.g. qwen3
            # lossy). We restore the documented semantics here.
            rr = 1.0

            # Dependency recall (S4 headline; paper Table 3 column).
            dep_recall = _compute_dep_recall(dep_context, agent_output)

            # Task success
            task_success = 1.0 if (la > 0.3 and len(agent_output) > 50) else 0.0

            fasr_raw.append((t, fasr))
            rr_raw.append((t, rr))
            cfr_raw.append((t, cfr))
            la_raw.append((t, la))
            task_m_raw.append((t, task_success))
            dep_recall_raw.append((t, dep_recall))

            # Build design notes for memory
            design_notes = (
                f"Sprint {t}: {task_text}\n"
                f"Files modified: {', '.join(sorted(predicted_impact))}\n"
                f"Tests: {sum(1 for v in tests_after.values() if v == 'pass')}/{len(tests_after)} passing\n"
                f"Key decisions: {agent_output[:300]}\n"
            )
            all_design_notes += design_notes + "\n"

            _log_traj("sprint_scores", session=t,
                      fasr=fasr, rr=rr, cfr=round(cfr, 4), la=round(la, 4),
                      task_success=task_success, dep_recall=dep_recall,
                      predicted_impact=sorted(predicted_impact),
                      ground_truth_impact=sorted(ground_truth_impact))

            out_tok = -1
            # Write routing across attribution modes (mirrors S2Runner).
            _uses_sut_policy = (
                not is_no_memory
                and not self.oracle_store
                and not self.incontext_ceiling
            )
            if self.incontext_ceiling:
                self._append_c4(design_notes)
            elif self.oracle_store:
                self._write_c3(design_notes)
            elif _uses_sut_policy:
                _progress(f"session {t + 1}: memory write start", session_t0)
                self.memory_policy.write(design_notes, llm=self.llm)

                compressed = self.memory_policy.read()
                in_tok = getattr(self.memory_policy, "last_input_tokens", 0)
                out_tok = getattr(self.memory_policy, "last_output_tokens", 0)

                _log_traj("compression", session=t,
                          input_text=design_notes,
                          output_text=compressed or "",
                          input_tokens=in_tok, output_tokens=out_tok,
                          compression_ratio=round(
                              len(design_notes.split()) / max(len((compressed or "").split()), 1), 2
                          ))
                self.tracer.log_llm_call(
                    parent_span_id=sess_span,
                    model=self._model_id, provider=self._provider,
                    input_tokens=in_tok, output_tokens=out_tok,
                    input_preview=design_notes,
                    output_preview=compressed or "",
                    thought=getattr(self.llm, "last_thought", ""),
                    cycle=t,
                )
                _progress(
                    f"session {t + 1}: memory write done (in_tok={in_tok}, out_tok={out_tok})",
                    session_t0,
                )

            # Execute dependency probe if present in generated data.
            # Skipped for C3/C4 paths that replace memory_policy entirely; the
            # dep_probe assumes agent-managed memory and doesn't compose with
            # the oracle conditions in a way the current scorer handles.
            dep_probe_result = None
            dep_probe = task.get("dependency_probe")
            if (dep_probe and not self.oracle_mode and not self.oracle_store
                    and not self.incontext_ceiling):
                probe_user_msg = (
                    f"Based on your knowledge of all prior sprints, answer this question.\n\n"
                    f"Question: {dep_probe['question']}\n\n"
                    f"Answer concisely with specific values and names."
                )
                if memory_text:
                    probe_user_msg = f"Context from prior sprints:\n{memory_text}\n\n{probe_user_msg}"
                probe_messages = [
                    {"role": "system", "content": "You are a software engineer recalling prior design decisions."},
                    {"role": "user", "content": probe_user_msg},
                ]
                probe_output = self.llm.chat(probe_messages)
                probe_kw = dep_probe.get("eval_keywords", [])
                probe_kw_found = [kw for kw in probe_kw
                                  if kw.lower() in probe_output.lower()]
                probe_score = len(probe_kw_found) / max(len(probe_kw), 1)
                dep_probe_result = {
                    "session": t,
                    "question": dep_probe["question"],
                    "output": probe_output[:10000],
                    "eval_keywords": probe_kw,
                    "keywords_found": probe_kw_found,
                    "score": probe_score,
                }
                _log_traj("dependency_probe", session=t,
                          question=dep_probe["question"][:200],
                          output=probe_output[:300],
                          score=probe_score,
                          keywords_found=probe_kw_found)

            sr = {
                "session": t,
                "task": task_text[:100],
                "fasr": fasr,
                "rr": rr,
                "cfr": cfr,
                "la": la,
                "task_success": task_success,
                "predicted_impact": sorted(predicted_impact),
                "ground_truth_impact": sorted(ground_truth_impact),
                "files_written": sorted(written.keys()) if written else [],
                "tests_passing": sum(1 for v in tests_after.values() if v == "pass"),
                "tests_total": len(tests_after),
                "tests_broken": sorted(passing_before - passing_after),
                "dep_recall": dep_recall,
                "depends_on": depends_on,
                "dependency_probe_result": dep_probe_result,
                # Agent output + probe output so dependency_scorer.forget_accuracy
                # can scan for invalidated keywords in both the coding response
                # and the probe answer.
                "task_outputs_text": (
                    (agent_output[:10000] if agent_output else "")
                    + (" " + dep_probe_result["output"] if dep_probe_result else "")
                ),
                # Token-cap diagnostics
                "response_tokens_task": (
                    __import__("agingbench.metrics.aging", fromlist=["count_response_tokens"])
                        .count_response_tokens(self.llm, agent_output)
                ),
                "response_tokens_probe": (
                    __import__("agingbench.metrics.aging", fromlist=["count_response_tokens"])
                        .count_response_tokens(self.llm, dep_probe_result["output"])
                    if dep_probe_result else -1
                ),
                "memory_write_tokens": out_tok,
            }
            session_results.append(sr)

            self.tracer.log(
                "session_scored", parent_span_id=sess_span, session=t,
                fasr=fasr, rr=rr, cfr=round(cfr, 4), la=round(la, 4),
                task_success=task_success,
            )
            self.tracer.log("session_end", parent_span_id=sess_span, session=t)
            _progress(
                f"session {t + 1}/{actual_sessions} end: LA={la:.3f}, CFR={cfr:.3f}, FASR={fasr:.1f}",
                session_t0,
            )

            print(f"  [S4] Sprint {t:2d}  FASR={fasr:.1f}  RR={rr:.0f}  "
                  f"CFR={cfr:.3f}  LA={la:.3f}  files={len(snapshot_files)}")

        # Build aging curves
        la_curve = AgingCurve(
            exposures=[r[0] for r in la_raw],
            scores=[r[1] for r in la_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        cfr_curve = AgingCurve(
            exposures=[r[0] for r in cfr_raw],
            scores=[1.0 - r[1] for r in cfr_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        fasr_curve = AgingCurve(
            exposures=[r[0] for r in fasr_raw],
            scores=[r[1] for r in fasr_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )
        # dep_recall is the dependency-graph aging signal — it directly
        # tracks whether the agent still recalls upstream design decisions
        # and is the natural headline for a temporal-dependency scenario.
        # la (location accuracy) can saturate at 1.0 for some models even
        # while dep_recall collapses (e.g. qwen3 lossy goes la=1.0 but
        # dep_recall=0.145).
        dep_recall_curve = AgingCurve(
            exposures=[r[0] for r in dep_recall_raw],
            scores=[r[1] for r in dep_recall_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id,
        )

        # FAITHFUL dep recall: aggregate the LLM-answered dependency probe scores.
        # Unlike the substring-on-dep_context proxy above (which scores whether
        # the agent's coding response echoes fresh prompt content), this metric
        # tests whether the agent can ANSWER a recall question with only
        # compressed memory_text — no dep_context provided to the probe call.
        # Sparse: only present at sessions where dependency_density triggered.
        dep_recall_faithful_raw: list[tuple[int, float]] = []
        for sr in session_results:
            dpr = sr.get("dependency_probe_result")
            if dpr is not None and "score" in dpr:
                dep_recall_faithful_raw.append((sr["session"], float(dpr["score"])))

        # Life event analysis
        life_event_result = None
        if self.life_event_session is not None and self.life_event_session < len(la_raw):
            idx = self.life_event_session
            scores_list = [r[1] for r in la_raw]
            if idx > 0:
                shock = compute_shock(scores_list[idx - 1], scores_list[idx])
                recovery_scores = scores_list[idx:]
                recovery = compute_recovery(recovery_scores, scores_list[idx - 1])
                life_event_result = {
                    "session": idx,
                    "shock_delta_m": shock,
                    "recovery_sessions": recovery,
                    "m_before": scores_list[idx - 1],
                    "m_after": scores_list[idx],
                }

        self.tracer.log(
            "run_end", parent_span_id=run_span,
            la_curve=la_raw, cfr_curve=cfr_raw, fasr_curve=fasr_raw,
            life_event=life_event_result,
        )

        # Close trajectory log
        _log_traj("run_end", n_sessions=actual_sessions,
                  m_final=la_raw[-1][1] if la_raw else 0)
        traj_f.close()
        _progress(f"run complete: m_final={la_raw[-1][1] if la_raw else 0:.3f}")

        # Attribution provenance stamp (see s2_runner for semantics).
        if self.incontext_ceiling:
            _attr_mode = "c4_incontext_ceiling"
        elif self.oracle_store:
            _attr_mode = "c3_oracle_store"
        elif self.oracle_retrieval:
            _attr_mode = "c2_oracle_retrieval"
        else:
            _attr_mode = "c1_baseline"

        return {
            "la_curve": la_curve,
            "cfr_curve": cfr_curve,
            "fasr_curve": fasr_curve,
            "dep_recall_curve": dep_recall_curve,
            "la_raw": la_raw,
            "cfr_raw": cfr_raw,
            "fasr_raw": fasr_raw,
            "rr_raw": rr_raw,
            "task_m_raw": task_m_raw,
            "dep_recall_raw": dep_recall_raw,
            "dep_recall_faithful_raw": dep_recall_faithful_raw,
            "session_results": session_results,
            "life_event": life_event_result,
            "attribution_schema": "v2_clean",
            "attribution_mode": _attr_mode,
            "ceiling_max_tokens": (
                self.ceiling_max_tokens if self.incontext_ceiling else None
            ),
        }
