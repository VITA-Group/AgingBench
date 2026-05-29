"""
S7+ Research-Notes CLI — data-driven test suite for procedurally-generated
blocks (session_idx >= 10).

The S7 generator (extension_mode="procedural") emits one declarative test spec
per new command / schema revision / migration it schedules. The runner writes
those specs to a JSON file and points AGINGBENCH_S7_GENERATED_TESTS at it; this
module turns each spec into a session-marked pytest case, reusing the same
`--scenario-session` gating defined in conftest.py.

If the env var is unset (curriculum-only runs, n <= 10), no specs load and this
module contributes no functional tests — so it never affects the hand-written
test_cli.py results.

Spec kinds (emitted by _S7ProceduralExtender):
  command_runs  : {command, args?}     -> CLI invocation returns 0
  schema_field  : {field}              -> a freshly-added note carries the field
  backend_file  : {file}               -> the migrated storage file exists
  smoke         : {commands: [...]}    -> each command still returns 0
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Spec loading (data-driven). Empty when the runner didn't emit generated tests.
# ---------------------------------------------------------------------------

def _load_specs() -> list[dict]:
    path = os.environ.get("AGINGBENCH_S7_GENERATED_TESTS")
    if not path or not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


_SPECS = _load_specs()


# ---------------------------------------------------------------------------
# CLI helpers — self-contained copies of the test_cli.py discovery logic so this
# module has no import-time dependency on the sibling test module.
# ---------------------------------------------------------------------------

def _notes_cmd(workspace: Path) -> list[str]:
    import shutil as _sh
    py = os.environ.get("OPENHANDS_BRIDGE_PYTHON", sys.executable)
    if _sh.which("notes"):
        return ["notes"]
    for mod in ("notes", "notes_cli", "notescli"):
        probe = subprocess.run(
            [py, "-m", mod, "--help"], cwd=workspace,
            capture_output=True, text=True, timeout=10,
        )
        if probe.returncode == 0:
            return [py, "-m", mod]
    for candidate in ("notes.py", "cli.py", "main.py", "app.py", "__main__.py"):
        p = workspace / candidate
        if p.exists():
            return [py, str(p)]
    for pkg in workspace.glob("*/__main__.py"):
        return [py, "-m", pkg.parent.name]
    return ["notes"]


def _run_notes(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HOME"] = str(workspace)
    env["PYTHONPATH"] = f"{workspace}:{env.get('PYTHONPATH', '')}"
    cmd = [*_notes_cmd(workspace), *args]
    return subprocess.run(
        cmd, cwd=workspace, capture_output=True, text=True, env=env, timeout=30,
    )


def _workspace() -> Path:
    p = os.environ.get("AGINGBENCH_S7PLUS_WORKSPACE")
    if not p:
        pytest.skip("AGINGBENCH_S7PLUS_WORKSPACE env var not set")
    ws = Path(p)
    if not ws.exists():
        pytest.skip(f"workspace {ws} does not exist")
    return ws


# ---------------------------------------------------------------------------
# Parametrized, session-marked test cases (gated by conftest --scenario-session)
# ---------------------------------------------------------------------------

def _spec_id(s: dict) -> str:
    key = s.get("command") or s.get("field") or s.get("file") or "smoke"
    return f"s{s.get('session', 0)}_{s.get('kind', 'x')}_{key}"


def _params():
    out = []
    for s in _SPECS:
        out.append(pytest.param(
            s, marks=pytest.mark.session(int(s.get("session", 0))), id=_spec_id(s),
        ))
    return out


@pytest.mark.parametrize("spec", _params())
def test_generated_block(spec):
    ws = _workspace()
    kind = spec.get("kind")

    if kind == "command_runs":
        r = _run_notes(ws, spec["command"], *spec.get("args", []))
        assert r.returncode == 0, f"`{spec['command']}` failed: {r.stderr[:200]}"

    elif kind == "schema_field":
        # New notes must carry the field. If the agent migrated off per-note
        # JSON to a DB backend, we can't introspect files — skip rather than
        # false-fail (the version_accuracy probe still scores the schema).
        _run_notes(ws, "add", "--title", "GenSchemaProbe",
                   "--body", "body", "--tags", "gen")
        notes_dir = ws / "notes_data"
        files = sorted(notes_dir.glob("*.json")) if notes_dir.exists() else []
        if not files:
            pytest.skip("no per-note JSON to introspect (DB backend); schema checked via probe")
        note = json.loads(files[-1].read_text())
        assert spec["field"] in note, f"new note missing field '{spec['field']}'"

    elif kind == "backend_file":
        assert (ws / spec["file"]).exists(), \
            f"expected migrated storage file '{spec['file']}' to exist"

    elif kind == "smoke":
        for c in spec.get("commands", []):
            r = _run_notes(ws, c)
            assert r.returncode == 0, f"smoke command `{c}` broke: {r.stderr[:200]}"

    else:
        pytest.skip(f"unknown generated spec kind: {kind}")
