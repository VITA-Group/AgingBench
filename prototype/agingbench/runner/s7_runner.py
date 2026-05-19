"""
agingbench/runner/s7_runner.py — S7+ Research-Notes CLI runner.

Wraps a Tier-2 AgentAdapter (OpenHands, Claude Code, etc.) around the S7+
scripted coding task. Each session:
  1. Send task prompt (agent writes/modifies code in workspace)
  2. Run pytest against the workspace (functional ground-truth signal)
  3. Send held-out probes one at a time, forcing agent to rely on
     workspace files (not conversation memory) between turns by calling
     adapter.reset_session() after the task and before probes
  4. Score probes by keyword match against expected answers
  5. Aggregate per-session: task_score, probe_score, pytest_pass_rate,
     workspace_fidelity, FactGraph-typed metrics

Output schema mirrors S7's metrics.json with additional FactGraph-scoped
submetrics (version_accuracy, interference_resistance, accumulator_error).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.agent_adapter import AgentAdapter


def _keyword_match_score(response: str, expected: list[str],
                         forbidden: list[str] | None = None) -> float:
    """Fraction of expected keywords present minus penalty for forbidden ones."""
    if not expected:
        return 1.0
    response_lower = response.lower()
    hits = sum(1 for kw in expected if kw.lower() in response_lower)
    base = hits / len(expected)
    if forbidden:
        penalties = sum(1 for kw in forbidden if kw.lower() in response_lower)
        base = max(0.0, base - 0.5 * (penalties / max(1, len(forbidden))))
    return base


def _extract_count_from_text(text: str) -> int | None:
    """Find the first plausible integer in a response (for accumulator probes).

    Looks for explicit numerals first, then English number words up to 20.
    Returns None if nothing parses.
    """
    if not text:
        return None
    for m in re.finditer(r"\b(\d{1,3})\b", text):
        return int(m.group(1))
    words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17,
    }
    low = text.lower()
    for w, n in words.items():
        if f" {w} " in f" {low} " or low.startswith(w + " "):
            return n
    return None


def _accumulator_gold_at(factgraph: dict, session_idx: int,
                         name: str = "test_count") -> float | None:
    """Ground-truth accumulator value at end of session_idx."""
    accs = factgraph.get("accumulators", {}) if factgraph else {}
    a = accs.get(name)
    if not a:
        return None
    total = a.get("initial_value", 0.0)
    for d in a.get("deltas", []):
        if d.get("session", 0) <= session_idx:
            total += d.get("amount", 0.0)
    return total


def _workspace_fidelity(workspace: Path, canonical_keywords: list[str]) -> float:
    """Fraction of canonical keywords present in the agent's workspace files."""
    if not canonical_keywords:
        return 1.0
    blobs = []
    for ext in ("*.md", "*.py", "*.txt", "*.json"):
        for f in workspace.rglob(ext):
            if f.is_file() and f.stat().st_size < 200_000:
                try:
                    blobs.append(f.read_text(errors="replace").lower())
                except Exception:
                    continue
    combined = "\n".join(blobs)
    hits = sum(1 for kw in canonical_keywords if kw.lower() in combined)
    return hits / len(canonical_keywords)


def _fix_escape_artifacts(workspace: Path):
    """Post-process agent-written files to fix OpenHands file-editor escape bugs.

    The bug: gpt-4o-mini + the default file_editor tool sometimes emits
    escaped sequences (\\n, \\t, \\", \\\\) inside file contents, which makes
    Python/Markdown/JSON files syntactically broken. We attempt cumulative
    unescaping and only commit a fix if it makes a .py file *more* parseable
    (i.e., reduces SyntaxErrors) or the file was already unparseable.

    We deliberately do NOT "fix" already-parseable Python code even if it
    contains escape-like tokens, because those might be legitimate strings.
    """
    import ast

    def _py_parseable(text: str) -> bool:
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False

    candidates = [".py", ".md", ".txt", ".toml", ".cfg", ".json", ".yaml", ".yml"]
    for ext in candidates:
        for f in workspace.rglob(f"*{ext}"):
            if not f.is_file() or f.stat().st_size > 200_000:
                continue
            if any(bad in f.parts for bad in
                   (".openhands_persist", "__pycache__", ".openhands_memory",
                    "notes.egg-info", "build", "dist")):
                continue
            try:
                text = f.read_text(errors="replace")
            except Exception:
                continue

            if ext == ".py":
                # For Python: only apply fix if file is currently broken AND
                # the fix makes it parseable. Prevents "fixing" valid strings.
                if _py_parseable(text):
                    continue
                candidates_fix = [
                    text.replace('\\"\\"\\"', '"""'),
                    text.replace("\\'\\'\\'", "'''"),
                    text.replace('\\"', '"'),
                    text.replace("\\'", "'"),
                ]
                # Apply all in combination if file has no real newlines
                agg = text
                if "\n" not in agg and "\\n" in agg:
                    agg = agg.replace("\\n", "\n")
                if "\\t" in agg and "\t" not in agg:
                    agg = agg.replace("\\t", "    ")
                agg = agg.replace('\\"\\"\\"', '"""')
                agg = agg.replace('\\"', '"')
                agg = agg.replace("\\'", "'")
                if "\\\\" in agg and "\\n" not in agg:
                    agg = agg.replace("\\\\", "\\")
                candidates_fix.append(agg)

                # Commit the first candidate that is parseable
                for fixed in candidates_fix:
                    if _py_parseable(fixed) and fixed != text:
                        f.write_text(fixed)
                        break
            else:
                # Non-Python: heuristic — un-escape only if no real newlines exist
                original = text
                if "\n" not in text and "\\n" in text:
                    text = text.replace("\\n", "\n")
                if ext in (".md", ".txt") and "\\\"" in text:
                    text = text.replace("\\\"", "\"")
                if text != original:
                    f.write_text(text)


# Keep old name as alias so the caller below still works
_fix_literal_newlines = _fix_escape_artifacts


def _pip_install_workspace(workspace: Path, py: str) -> dict:
    """Try `pip install -e .` in the workspace. Returns status dict."""
    setup_exists = (workspace / "setup.py").exists() or (workspace / "pyproject.toml").exists()
    if not setup_exists:
        return {"installed": False, "reason": "no setup.py or pyproject.toml"}
    try:
        r = subprocess.run([py, "-m", "pip", "install", "-e", ".", "--quiet"],
                           cwd=workspace, capture_output=True, text=True, timeout=60)
        return {"installed": r.returncode == 0,
                "stderr": r.stderr[-500:] if r.stderr else ""}
    except subprocess.TimeoutExpired:
        return {"installed": False, "reason": "pip install timeout"}


def _run_pytest(workspace: Path, session_idx: int, tests_dir: Path) -> dict:
    """Run pytest in the workspace, gated by --scenario-session."""
    cmd = [
        "python", "-m", "pytest", "-q",
        "--scenario-session", str(session_idx),
        str(tests_dir),
    ]
    env = os.environ.copy()
    env["AGINGBENCH_S7PLUS_WORKSPACE"] = str(workspace)
    # Prefer the openhands env's python (has pytest preinstalled); fall back
    # to the current interpreter otherwise. Users on non-standard layouts
    # can override via OPENHANDS_BRIDGE_PYTHON.
    bridge_py = os.environ.get("OPENHANDS_BRIDGE_PYTHON", sys.executable)
    if Path(bridge_py).exists():
        cmd[0] = bridge_py
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return {"passed": 0, "failed": 0, "skipped": 0, "total": 0,
                "pass_rate": 0.0, "error": "pytest timeout"}
    # Parse summary line: "X passed, Y failed, Z skipped in ..."
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0
    skipped = int(m.group(1)) if (m := re.search(r"(\d+) skipped", out)) else 0
    non_skipped = passed + failed + errors
    pass_rate = passed / non_skipped if non_skipped > 0 else 0.0
    return {
        "passed": passed, "failed": failed, "errors": errors, "skipped": skipped,
        "total": non_skipped, "pass_rate": pass_rate,
    }


class S7Runner:
    def __init__(
        self,
        adapter: AgentAdapter,
        tracer,
        sut_id: str,
        generated_data: dict,
        tests_dir: Path,
        workspace_dir: Path,
        max_probe_retries: int = 0,
        snapshots_dir: Path | None = None,
        archive_dir: Path | None = None,
    ):
        self.adapter = adapter
        self.tracer = tracer
        self.sut_id = sut_id
        self.data = generated_data
        self.tests_dir = tests_dir
        self.workspace = workspace_dir
        self.max_probe_retries = max_probe_retries
        self.fg = generated_data.get("dependency_graph", {})
        # When workspace_dir is an isolated tmpdir (to prevent context
        # contamination), snapshots_dir and archive_dir point at the
        # original experiment output so per-session source and final state
        # still land in the archival location we report.
        self.snapshots_dir = snapshots_dir or (workspace_dir.parent / "snapshots")
        self.archive_dir = archive_dir  # None = don't copy back at end

    # ------------------------------------------------------------------ run loop

    def run(self, n_sessions: int | None = None, seed: int = 42) -> dict:
        sessions = self.data["sessions"]
        if n_sessions is not None:
            sessions = sessions[:n_sessions]

        per_session_metrics: list[dict] = []
        recall_curve: list[tuple[int, float]] = []
        task_curve: list[tuple[int, float]] = []
        pytest_curve: list[tuple[int, float]] = []
        ws_fid_curve: list[tuple[int, float]] = []

        for sess in sessions:
            t0 = time.time()
            idx = sess["session_idx"]

            # 1. Send task prompt
            self.adapter.reset_session()
            try:
                task_resp = self.adapter.send_message(sess["task_prompt"])
                task_text = task_resp.text
                task_tokens = task_resp.input_tokens + task_resp.output_tokens
            except Exception as e:
                task_text = f"[ERROR: {type(e).__name__}: {e}]"
                task_tokens = 0

            # 2. Post-process + pip install (fail-soft) + pytest slice
            _fix_literal_newlines(self.workspace)
            bridge_py = os.environ.get("OPENHANDS_BRIDGE_PYTHON", sys.executable)
            install_result = _pip_install_workspace(self.workspace, bridge_py)
            pytest_result = _run_pytest(self.workspace, idx, self.tests_dir)
            pytest_result["pip_install"] = install_result

            # 2b. Snapshot the post-session workspace source files so we can
            # diagnose "design issue" vs "real aging" by re-running a later
            # session's tests against this session's code, or inspecting the
            # specific source at that point in time.
            snap_dir = self.snapshots_dir / f"session_{idx:02d}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            for src_ext in (".py", ".md", ".txt", ".toml", ".cfg"):
                for src in self.workspace.rglob(f"*{src_ext}"):
                    if any(bad in src.parts for bad in
                           (".openhands_persist", "__pycache__",
                            "notes.egg-info", "build", "dist", "snapshots")):
                        continue
                    rel = src.relative_to(self.workspace)
                    dst = snap_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

            # 3. Send probes (fresh conversation each probe — force file reads)
            probe_scores = []
            probe_details = []
            for p in sess["eval_probes"]:
                self.adapter.reset_session()  # wipe conversation, keep workspace
                try:
                    pr = self.adapter.send_message(p["prompt"])
                    score = _keyword_match_score(
                        pr.text, p["expected_keywords"], p["forbidden_keywords"]
                    )
                except Exception as e:
                    pr_text = f"[ERROR: {e}]"
                    pr = None
                    score = 0.0
                probe_scores.append(score)
                # num_turns at probe time is the key signal for the
                # "memory-vs-lookup" strategy distinction:
                #   - num_turns == 1 → answered immediately (relied on memory)
                #   - num_turns >> 1 → invoked tools to look things up
                probe_turns = (pr.metadata.get("num_turns", 0)
                               if pr and getattr(pr, "metadata", None) else 0)
                probe_details.append({
                    "id": p["id"],
                    "metric_tag": p["metric_tag"],
                    "probe_type": p["probe_type"],
                    "chain_depth": p["chain_depth"],
                    "score": score,
                    "num_turns": probe_turns,
                    "response_text": (pr.text[:300] if pr else "error"),
                })

            # 4. Workspace fidelity (all canonical keywords across all sessions so far)
            all_canonical = set()
            for s2 in sessions[:idx + 1]:
                for p2 in s2["eval_probes"]:
                    all_canonical.update(p2["expected_keywords"])
            ws_fid = _workspace_fidelity(self.workspace, list(all_canonical))

            # 5. Mechanism-scoped metrics — canonical four + raw per-tag
            by_tag: dict[str, list[float]] = {}
            for pd in probe_details:
                by_tag.setdefault(pd["metric_tag"], []).append(pd["score"])
            mech_metrics = {
                f"m_{tag}": sum(v) / len(v) if v else None
                for tag, v in by_tag.items()
            }

            # Canonical four-mechanism aggregates (match Table 3 column semantics)
            # - Compression: workspace_fidelity (keyword_m for Tier-2 files)
            # - Interference: mean score on metric_tag="interference" probes
            # - Revision (explicit version): mean score on metric_tag="version_accuracy"
            # - Revision (latent accumulator): |extracted_count - gold| for
            #   metric_tag="accumulator" probes; if multiple, average absolute error
            # - Maintenance: Δ_pre/post is computed at the run level, not per session
            comp_score = ws_fid
            interf_score = (sum(by_tag.get("interference", [])) /
                            len(by_tag["interference"])
                            if by_tag.get("interference") else None)
            rev_explicit = (sum(by_tag.get("version_accuracy", [])) /
                            len(by_tag["version_accuracy"])
                            if by_tag.get("version_accuracy") else None)
            # Latent accumulator: parse number from agent response, compare to
            # the *probe-specific* gold. Prefer the first numeric keyword in
            # the probe's expected_keywords (this is the actual correct count
            # the probe asks about — command count, test count, etc.).
            # Fall back to the shared FactGraph `test_count` accumulator if
            # the probe has no numeric keyword.
            accum_errors = []
            for pd in probe_details:
                if pd["metric_tag"] != "accumulator":
                    continue
                extracted = _extract_count_from_text(pd["response_text"])
                # Look up the original probe definition to get expected_keywords
                probe_def = next(
                    (sp for sp in sess["eval_probes"] if sp["id"] == pd["id"]),
                    None,
                )
                gold = None
                if probe_def:
                    for kw in probe_def.get("expected_keywords", []):
                        try:
                            gold = int(kw)
                            break
                        except (ValueError, TypeError):
                            continue
                if gold is None:
                    gold = _accumulator_gold_at(self.fg, idx, name="test_count")
                if extracted is not None and gold is not None:
                    accum_errors.append(abs(extracted - gold))
            rev_latent_err = (sum(accum_errors) / len(accum_errors)
                              if accum_errors else None)

            mechanism_metrics = {
                "m_compression": comp_score,
                "m_interference": interf_score,
                "m_revision_explicit": rev_explicit,
                "m_revision_latent_abs_err": rev_latent_err,
            }

            # Probe-time look-up intensity. Distinguishes memory-from-context
            # (low turns) from look-up-from-files (high turns), separating
            # true memory aging from "agent didn't bother storing/retrieving".
            probe_turn_counts = [pd.get("num_turns", 0) for pd in probe_details]
            mechanism_metrics["probe_turns_mean"] = (
                sum(probe_turn_counts) / len(probe_turn_counts)
                if probe_turn_counts else 0.0
            )
            mechanism_metrics["probe_turns_max"] = (
                max(probe_turn_counts) if probe_turn_counts else 0
            )
            # Split: probe-turns when correct vs incorrect
            correct_turns = [pd["num_turns"] for pd in probe_details if pd["score"] >= 0.5]
            wrong_turns = [pd["num_turns"] for pd in probe_details if pd["score"] < 0.5]
            mechanism_metrics["probe_turns_when_correct"] = (
                sum(correct_turns) / len(correct_turns) if correct_turns else None
            )
            mechanism_metrics["probe_turns_when_wrong"] = (
                sum(wrong_turns) / len(wrong_turns) if wrong_turns else None
            )

            recall = sum(probe_scores) / len(probe_scores) if probe_scores else 0.0
            task_score = 1.0 if "[ERROR" not in task_text else 0.0  # coarse; refined by pytest
            elapsed = time.time() - t0

            session_result = {
                "session": idx,
                "task_prompt_tokens": task_tokens,
                "task_score_coarse": task_score,
                "pytest_pass_rate": pytest_result["pass_rate"],
                "pytest_passed": pytest_result["passed"],
                "pytest_failed": pytest_result["failed"],
                "pytest_total": pytest_result["total"],
                "probe_recall": recall,
                "workspace_fidelity": ws_fid,
                "elapsed_s": elapsed,
                "probe_details": probe_details,
                **mech_metrics,
                **mechanism_metrics,
            }
            per_session_metrics.append(session_result)
            recall_curve.append((idx, recall))
            task_curve.append((idx, task_score))
            pytest_curve.append((idx, pytest_result["pass_rate"]))
            ws_fid_curve.append((idx, ws_fid))

            if self.tracer:
                self.tracer.log("session_end", session=idx,
                                pytest_pass_rate=pytest_result["pass_rate"],
                                probe_recall=recall, workspace_fidelity=ws_fid)

            # Console progress
            print(
                f"  [S7+] Block {idx:2d}  task={task_score:.2f}  "
                f"pytest={pytest_result['pass_rate']:.3f} "
                f"({pytest_result['passed']}/{pytest_result['total']})  "
                f"probes={recall:.3f}  ws_fid={ws_fid:.3f}  ({elapsed:.0f}s)"
            )

        # Aging-curve aggregates on probe_recall (the primary longitudinal signal)
        m0 = recall_curve[0][1] if recall_curve else 1.0
        m_final = recall_curve[-1][1] if recall_curve else 0.0
        half_life = self._compute_half_life(recall_curve)
        slope = self._compute_slope(recall_curve)

        # Maintenance Δ_pre/post around lifecycle events (session 3 = schema
        # migration; session 8 = SQLite migration). Pre/post windows are
        # 2 sessions each (or whatever fits given run length).
        maint_deltas = {}
        for event_name, k in [("schema_migration_s3", 3), ("sqlite_migration_s8", 8)]:
            pre_window = [v for t, v in recall_curve if k - 2 <= t < k]
            post_window = [v for t, v in recall_curve if k < t <= k + 2]
            if pre_window and post_window:
                maint_deltas[event_name] = {
                    "pre_mean": sum(pre_window) / len(pre_window),
                    "post_mean": sum(post_window) / len(post_window),
                    "delta": (sum(post_window) / len(post_window)) -
                             (sum(pre_window) / len(pre_window)),
                    "window_size": 2,
                }

        # Run-level mechanism aggregates (mean over sessions where metric defined)
        def _mean_over_sessions(key):
            vals = [s.get(key) for s in per_session_metrics
                    if s.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        run_level_mechanisms = {
            "m_compression_mean": _mean_over_sessions("m_compression"),
            "m_compression_final": (per_session_metrics[-1].get("m_compression")
                                    if per_session_metrics else None),
            "m_interference_mean": _mean_over_sessions("m_interference"),
            "m_revision_explicit_mean": _mean_over_sessions("m_revision_explicit"),
            "m_revision_latent_abs_err_mean":
                _mean_over_sessions("m_revision_latent_abs_err"),
            "m_maintenance_delta": maint_deltas,
            # Run-level probe-turn aggregates — distinguishes lookup from memory
            "probe_turns_mean": _mean_over_sessions("probe_turns_mean"),
            "probe_turns_when_correct_mean":
                _mean_over_sessions("probe_turns_when_correct"),
            "probe_turns_when_wrong_mean":
                _mean_over_sessions("probe_turns_when_wrong"),
        }

        return {
            "scenario": "s7_research_notes",
            "sut_id": self.sut_id,
            "n_sessions": len(per_session_metrics),
            "m0": m0,
            "m_final": m_final,
            "half_life": half_life,
            "decay_slope": slope,
            # `checkpoints` is the AgingCard's canonical per-session trajectory
            # field. For S7 the recall_curve is the headline metric, so it
            # doubles as the checkpoints series. Aliased rather than
            # duplicated so downstream consumers can use either name; renamed
            # the headline_metric to make the alias explicit.
            "checkpoints": list(recall_curve),
            "headline_metric": "recall_accuracy",
            "recall_curve": recall_curve,
            "task_curve": task_curve,
            "pytest_curve": pytest_curve,
            "workspace_fidelity_curve": ws_fid_curve,
            "per_session": per_session_metrics,
            "factgraph_summary": self.fg.get("summary", {}),
            **run_level_mechanisms,
        }

    # Note: the run() method returns above. After run() returns, the
    # CLI wrapper is responsible for calling _archive_workspace if an
    # archive_dir was set — done separately to keep run() pure.

    def archive_workspace_if_set(self):
        """Copy isolated-workspace contents to archive_dir at run end."""
        if not self.archive_dir:
            return
        import shutil as _sh
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        for src in self.workspace.rglob("*"):
            if any(bad in src.parts for bad in
                   (".openhands_persist", "__pycache__",
                    "notes.egg-info", "build", "dist")):
                continue
            if not src.is_file():
                continue
            rel = src.relative_to(self.workspace)
            dst = self.archive_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                _sh.copy2(src, dst)
            except Exception:
                pass

    @staticmethod
    def _compute_slope(curve: list[tuple[int, float]]) -> float:
        if len(curve) < 2:
            return 0.0
        xs = [c[0] for c in curve]
        ys = [c[1] for c in curve]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den if den else 0.0

    @staticmethod
    def _compute_half_life(curve: list[tuple[int, float]]) -> float:
        """Sessions until recall first crosses below 0.5 (inf if never)."""
        if not curve:
            return float("inf")
        m0 = curve[0][1]
        target = 0.5 * m0
        for t, v in curve:
            if v <= target:
                return float(t)
        return float("inf")
