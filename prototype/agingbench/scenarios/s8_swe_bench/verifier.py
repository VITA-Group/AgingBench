"""S8 SWE-bench-Aging — patch application + test verification (Phase 4).

Per session:
  1. apply_diff_in_container: git-apply the agent's solution.diff inside
     the SWE-bench container's /testbed.
  2. run_verification: invoke pytest on the issue's FAIL_TO_PASS +
     PASS_TO_PASS tests; parse pass/fail.
  3. Score: pass=1 iff (patch applied cleanly) AND (all FAIL_TO_PASS now
     pass) AND (all PASS_TO_PASS still pass); else 0.

This is a lightweight wrapper around SWE-bench's expected verifier
contract — we use pytest directly instead of their full
run_evaluation.py since we already control the container lifecycle.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Avoid hard import dependency at module-load time for tests.


@dataclass
class ApplyResult:
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    diff_bytes: int


@dataclass
class VerifyResult:
    passed: bool
    n_fail_to_pass_total: int
    n_fail_to_pass_passed: int
    n_pass_to_pass_total: int
    n_pass_to_pass_passed: int
    fail_to_pass_results: dict[str, str] = field(default_factory=dict)  # test_id -> 'passed'|'failed'
    pass_to_pass_results: dict[str, str] = field(default_factory=dict)
    raw_log: str = ""
    error: Optional[str] = None
    duration_sec: Optional[float] = None


# ---- patch application ----------------------------------------------------

def apply_diff_in_container(session, diff_text: Optional[str]) -> ApplyResult:
    """git-apply a diff inside the container's /testbed.

    Empty / missing diff is treated as a successful no-op apply (the
    verifier will then run tests against the unchanged base; FAIL_TO_PASS
    tests will fail, scoring=0).
    """
    if not diff_text or not diff_text.strip():
        return ApplyResult(
            success=True, exit_code=0, stdout="", stderr="(empty diff; no apply)",
            diff_bytes=0,
        )

    # Stage the diff inside the container at /tmp/s8_solution.diff.
    # Use base64 to avoid shell-escape pitfalls in heredocs / pipes.
    import base64
    encoded = base64.b64encode(diff_text.encode("utf-8")).decode("ascii")
    stage = session.exec(
        f"echo {encoded} | base64 -d > /tmp/s8_solution.diff",
        timeout_sec=30,
    )
    if stage.exit_code != 0:
        return ApplyResult(
            success=False, exit_code=stage.exit_code, stdout=stage.stdout,
            stderr=f"diff stage failed: {stage.stderr}",
            diff_bytes=len(diff_text),
        )

    # Apply with -p1. Try multiple strategies (most strict -> most lenient)
    # since agents sometimes produce well-formed-but-not-quite-git-strict
    # diffs (missing diff --git headers, fuzzy hunks, etc.).
    strategies = [
        ("git apply --whitespace=fix",
         "cd /testbed && git apply --whitespace=fix /tmp/s8_solution.diff"),
        ("git apply --whitespace=fix --recount",
         "cd /testbed && git apply --whitespace=fix --recount /tmp/s8_solution.diff"),
        ("git apply --whitespace=fix --3way",
         "cd /testbed && git apply --whitespace=fix --3way /tmp/s8_solution.diff"),
        ("patch -p1 --fuzz=3",
         "cd /testbed && patch -p1 --fuzz=3 --no-backup-if-mismatch < /tmp/s8_solution.diff"),
        ("patch -l -p1 --fuzz=10 --ignore-whitespace",
         "cd /testbed && patch -l -p1 --fuzz=10 --ignore-whitespace --no-backup-if-mismatch < /tmp/s8_solution.diff"),
    ]
    last = None
    for strategy_name, cmd in strategies:
        apply = session.exec(cmd, timeout_sec=60)
        last = apply
        if apply.exit_code == 0:
            return ApplyResult(
                success=True,
                exit_code=0,
                stdout=f"[{strategy_name}]\n{apply.stdout}",
                stderr=apply.stderr,
                diff_bytes=len(diff_text),
            )
        # If a strategy partially applied (some hunks landed), the repo
        # state is contaminated — reset before trying the next strategy.
        session.exec("cd /testbed && git checkout -- . 2>/dev/null || true",
                     timeout_sec=15)
    return ApplyResult(
        success=False,
        exit_code=last.exit_code if last else 1,
        stdout=last.stdout if last else "",
        stderr=(f"all 3 apply strategies failed; last stderr:\n{last.stderr}"
                if last else "no apply strategy attempted"),
        diff_bytes=len(diff_text),
    )


# ---- test execution + parsing --------------------------------------------

# Pytest formats are line-oriented; constrain whitespace to non-newline so
# patterns can't match across line boundaries.
_PYTEST_PASS_RE = re.compile(r"PASSED[ \t]+(\S+)")
_PYTEST_FAIL_RE = re.compile(r"FAILED[ \t]+(\S+)")
_PYTEST_INLINE_PASS_RE = re.compile(r"(\S+::\S+)[ \t]+PASSED")
_PYTEST_INLINE_FAIL_RE = re.compile(r"(\S+::\S+)[ \t]+(?:FAILED|ERROR)")


def _parse_pytest_log(log: str) -> dict[str, str]:
    """Return dict mapping test_id -> 'passed'|'failed'.

    Pass results win over fail (for the inline form, pytest only writes
    one of the two per test, so this is consistent).
    """
    results: dict[str, str] = {}
    for m in _PYTEST_FAIL_RE.finditer(log):
        results[m.group(1)] = "failed"
    for m in _PYTEST_INLINE_FAIL_RE.finditer(log):
        results[m.group(1)] = "failed"
    for m in _PYTEST_PASS_RE.finditer(log):
        results[m.group(1)] = "passed"
    for m in _PYTEST_INLINE_PASS_RE.finditer(log):
        results[m.group(1)] = "passed"
    return results


def _is_django_test_id(test_id: str) -> bool:
    """Django SWE-bench f2p IDs look like:
    `test_method (full.module.Path.ClassName)`."""
    return bool(re.match(r"^test_\w+\s+\(\S+\)$", test_id.strip()))


def _django_id_to_dotted(test_id: str) -> str:
    """`test_X (a.b.C)` -> `a.b.C.test_X`."""
    m = re.match(r"^(test_\w+)\s+\((\S+)\)$", test_id.strip())
    if not m:
        return test_id
    return f"{m.group(2)}.{m.group(1)}"


# Django runtests verbose output line:
#   test_method (full.module.Class) ... ok
#   test_method (full.module.Class) ... FAIL
#   test_method (full.module.Class) ... ERROR
#   test_method (full.module.Class) ... skipped 'reason'
_DJANGO_TEST_LINE_RE = re.compile(
    r"^(test_\w+)\s+\((\S+)\)\s*\.\.\.\s+(ok|FAIL|ERROR|skipped[^$]*)$",
    re.MULTILINE,
)


def _parse_django_log(log: str) -> dict[str, str]:
    """Return dict mapping SWE-bench-format test_id -> 'passed'|'failed'."""
    results: dict[str, str] = {}
    for m in _DJANGO_TEST_LINE_RE.finditer(log):
        method, klass, status = m.group(1), m.group(2), m.group(3).strip()
        test_id = f"{method} ({klass})"
        if status == "ok":
            results[test_id] = "passed"
        elif status.startswith("skipped"):
            results[test_id] = "passed"   # skipped is not a failure
        else:
            results[test_id] = "failed"
    return results


def run_verification(session,
                     fail_to_pass: list[str],
                     pass_to_pass: list[str],
                     test_patch: Optional[str] = None,
                     timeout_sec: int = 600,
                     max_pass_to_pass: int = 30) -> VerifyResult:
    """Run pytest against FAIL_TO_PASS + a sample of PASS_TO_PASS tests.

    PASS_TO_PASS lists can be 50+ tests; running them all per session
    multiplies wall time. We sample up to `max_pass_to_pass` (default 30)
    to keep per-session verification under ~60s; the sample is
    deterministic (first N).

    `test_patch` (optional): SWE-bench-Verified often delivers FAIL_TO_PASS
    tests as a *new* test function added by a separate `test_patch` diff
    that is part of the issue metadata. Without applying it, the F2P tests
    don't exist at the base commit; older pytest (<6.0) then aborts the
    WHOLE collection on the missing-test ID, dropping P2P results to zero.
    When provided, the patch is applied inside /testbed before pytest runs.
    """
    import time
    t0 = time.time()

    if test_patch and test_patch.strip():
        # Apply test_patch (adds F2P tests). Use the same multi-strategy
        # set as apply_diff_in_container — strict git apply first, fall
        # back to permissive `patch -l` for fuzzy hunks. Failure is
        # logged in the VerifyResult.error but doesn't abort the run
        # (pytest will then error per-test, which is parseable).
        import base64
        encoded = base64.b64encode(test_patch.encode("utf-8")).decode("ascii")
        stage = session.exec(
            f"echo {encoded} | base64 -d > /tmp/s8_test_patch.diff",
            timeout_sec=30,
        )
        tp_apply = None
        if stage.exit_code == 0:
            for cmd in (
                "cd /testbed && git apply --whitespace=fix /tmp/s8_test_patch.diff",
                "cd /testbed && git apply --whitespace=fix --recount /tmp/s8_test_patch.diff",
                "cd /testbed && patch -p1 --fuzz=3 --no-backup-if-mismatch < /tmp/s8_test_patch.diff",
                ("cd /testbed && patch -l -p1 --fuzz=10 --ignore-whitespace "
                 "--no-backup-if-mismatch < /tmp/s8_test_patch.diff"),
            ):
                tp_apply = session.exec(cmd, timeout_sec=30)
                if tp_apply.exit_code == 0:
                    break

    p2p_sample = pass_to_pass[:max_pass_to_pass]
    all_targets = list(fail_to_pass) + list(p2p_sample)
    if not all_targets:
        return VerifyResult(
            passed=False, n_fail_to_pass_total=0, n_fail_to_pass_passed=0,
            n_pass_to_pass_total=0, n_pass_to_pass_passed=0,
            error="no FAIL_TO_PASS or PASS_TO_PASS tests configured",
            duration_sec=0.0,
        )

    # Framework detection: SWE-bench's Django instances use Django's
    # tests/runtests.py with test IDs in `test_X (module.Class)` form.
    # Everything else uses pytest with `path::test_id` form.
    is_django = any(_is_django_test_id(t) for t in all_targets)

    if is_django:
        # Convert test IDs to dotted form for runtests command line.
        dotted = [_django_id_to_dotted(t) if _is_django_test_id(t) else t
                  for t in all_targets]
        quoted = " ".join(f"'{t}'" for t in dotted)
        cmd = (
            "cd /testbed && "
            "if [ -x /opt/miniconda3/envs/testbed/bin/python ]; then "
            "  PY=/opt/miniconda3/envs/testbed/bin/python; "
            "elif [ -x /opt/miniconda3/bin/python ]; then "
            "  PY=/opt/miniconda3/bin/python; "
            "else PY=python; fi && "
            f"$PY tests/runtests.py --verbosity 2 {quoted} 2>&1"
        )
        proc = session.exec(cmd, timeout_sec=timeout_sec)
        duration = round(time.time() - t0, 3)
        parsed = _parse_django_log(proc.stdout + "\n" + proc.stderr)
    else:
        # Quote each test path for the shell.
        quoted = " ".join(f"'{t}'" for t in all_targets)
        cmd = (
            "cd /testbed && "
            "if [ -x /opt/miniconda3/envs/testbed/bin/python ]; then "
            "  PY=/opt/miniconda3/envs/testbed/bin/python; "
            "elif [ -x /opt/miniconda3/bin/python ]; then "
            "  PY=/opt/miniconda3/bin/python; "
            "else PY=python; fi && "
            f"$PY -m pytest -rA --tb=no --continue-on-collection-errors {quoted}"
        )
        proc = session.exec(cmd, timeout_sec=timeout_sec)
        duration = round(time.time() - t0, 3)
        parsed = _parse_pytest_log(proc.stdout + "\n" + proc.stderr)

    f2p_results = {t: parsed.get(t, "missing") for t in fail_to_pass}
    p2p_results = {t: parsed.get(t, "missing") for t in p2p_sample}

    n_f2p_passed = sum(1 for v in f2p_results.values() if v == "passed")
    n_p2p_passed = sum(1 for v in p2p_results.values() if v == "passed")

    # SWE-bench scoring rule: ALL fail_to_pass MUST pass AND ALL pass_to_pass MUST pass.
    passed = (
        len(fail_to_pass) > 0
        and n_f2p_passed == len(fail_to_pass)
        and n_p2p_passed == len(p2p_sample)
    )

    # Truncate raw log so AgingCard doesn't balloon.
    log_excerpt = (proc.stdout + "\n--- STDERR ---\n" + proc.stderr)[:8000]

    return VerifyResult(
        passed=passed,
        n_fail_to_pass_total=len(fail_to_pass),
        n_fail_to_pass_passed=n_f2p_passed,
        n_pass_to_pass_total=len(p2p_sample),
        n_pass_to_pass_passed=n_p2p_passed,
        fail_to_pass_results=f2p_results,
        pass_to_pass_results=p2p_results,
        raw_log=log_excerpt,
        duration_sec=duration,
    )


# ---- SWE-bench instance metadata loader ----------------------------------

class _IssueMetadataCache:
    """Lazy loader for SWE-bench-Verified instance rows.

    Pulled from HuggingFace once per process; cached for fast per-session
    lookup of FAIL_TO_PASS / PASS_TO_PASS lists.
    """
    _cache: dict[str, dict] = {}

    @classmethod
    def get(cls, instance_id: str) -> dict:
        if instance_id in cls._cache:
            return cls._cache[instance_id]
        try:
            from datasets import load_dataset
            ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            for row in ds:
                cls._cache[row["instance_id"]] = {
                    "instance_id": row["instance_id"],
                    "base_commit": row["base_commit"],
                    "version": row["version"],
                    "problem_statement": row["problem_statement"],
                    "patch": row["patch"],
                    "test_patch": row["test_patch"],
                    "fail_to_pass": json.loads(row["FAIL_TO_PASS"]),
                    "pass_to_pass": json.loads(row["PASS_TO_PASS"]),
                }
        except Exception as exc:                                # noqa: BLE001
            return {"instance_id": instance_id, "error": f"{type(exc).__name__}: {exc}"}
        return cls._cache.get(instance_id, {"instance_id": instance_id,
                                            "error": "not_found_in_dataset"})


def get_instance_metadata(instance_id: str) -> dict:
    return _IssueMetadataCache.get(instance_id)
