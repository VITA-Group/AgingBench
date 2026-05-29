"""
S7+ Research-Notes CLI — pytest suite.

Each test is tagged with the session it belongs to. Tests are run from inside
the agent's workspace directory at the end of each session. Tests assert
functional correctness of the CLI the agent has built so far. Tests for
sessions that haven't happened yet are automatically skipped.

Invocation (from runner):
    pytest -q --session=N scenarios/s7_research_notes/tests/test_cli.py

Where N is the current session index; tests tagged with session > N are skipped.

Ground-truth behavior expected of the CLI binary `notes`:

  Session 0: notes add --title "t" --body "b" [--tags "x,y"] [--citation "..."]
             ⇒ writes a JSON file to notes_data/<id>.json
             ⇒ schema: {id: int, title: str, tags: list[str], body: str,
                        citation: str | None}

  Session 1: notes ls
             notes show <id>
             notes rm <id>
             notes search <query>   (substring on title + body)

  Session 2: notes filter-by-tag <tag>   (⇒ returns subset)
             notes sort-by-tag           (⇒ returns ALL, reordered)

  Session 3: schema gains `priority: int` (1..5, default 3)
             existing notes auto-migrate (priority=3 added if missing)

  Session 4: no new features; pure probe measurement
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# Pytest option + session marker filtering are defined in conftest.py
# (pytest_addoption must be in conftest.py, not in the test module).

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _notes_cmd(workspace: Path) -> list[str]:
    """Discover how the agent made the CLI invokable.
    Priority: (1) pip-installed `notes` on PATH; (2) `python -m notes`;
    (3) any `notes.py`, `cli.py`, or `main.py` in the workspace root;
    (4) any `main.py` inside a package directory."""
    import shutil as _sh
    # Allow override; fall back to the python on PATH for portability.
    # On the test author's machine this happens to live in a conda env;
    # users can point OPENHANDS_BRIDGE_PYTHON at their own openhands env.
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
    env["HOME"] = str(workspace)  # isolate user config
    env["PYTHONPATH"] = f"{workspace}:{env.get('PYTHONPATH', '')}"
    cmd = [*_notes_cmd(workspace), *args]
    return subprocess.run(
        cmd, cwd=workspace, capture_output=True, text=True, env=env, timeout=30,
    )


def _workspace() -> Path:
    """Workspace set by the runner via AGINGBENCH_S7PLUS_WORKSPACE env var."""
    p = os.environ.get("AGINGBENCH_S7PLUS_WORKSPACE")
    if not p:
        pytest.skip("AGINGBENCH_S7PLUS_WORKSPACE env var not set")
    ws = Path(p)
    if not ws.exists():
        pytest.skip(f"workspace {ws} does not exist")
    return ws


def _load_all_notes(ws: Path) -> list[dict]:
    """All persisted notes, read from whichever backend exists.

    Backend-agnostic by design: prefers per-note JSON under ``notes_data/``
    (the pre-migration layout); falls back to the SQLite store after the
    session-8 JSON->SQLite migration. This keeps the suite measuring CLI/data
    correctness rather than a specific on-disk layout — a note that was
    correctly migrated to SQLite must NOT fail tests that merely assert a note
    exists or carries a field. (Avoids the stale-`notes_data/*.json` artifact:
    clean migrations were false-failing, retained-JSON was false-passing.)
    """
    # Prefer the SQLite store when it exists WITH ROWS (the live post-migration
    # backend); fall back to per-note JSON (pre-migration layout). Checking
    # SQLite first reads the *live* store even when stale orphaned JSON files
    # remain after a migration — so add/rm round-trips and field checks reflect
    # what the CLI actually writes, not abandoned files.
    import sqlite3
    candidates = [ws / "notes.db"] + sorted(ws.glob("*.db")) + sorted(ws.glob("*.sqlite*"))
    seen = set()
    for db in candidates:
        if not db.exists() or db in seen:
            continue
        seen.add(db)
        try:
            con = sqlite3.connect(str(db))
            con.row_factory = sqlite3.Row
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            tname = "notes" if "notes" in tables else (tables[0] if tables else None)
            if tname:
                rows = [dict(r) for r in con.execute(f'SELECT * FROM "{tname}"')]
                con.close()
                if rows:
                    return rows
            else:
                con.close()
        except Exception:
            pass
    # JSON fallback (pre-migration layout, or no populated SQLite store).
    notes_dir = ws / "notes_data"
    out: list[dict] = []
    if notes_dir.exists():
        for f in sorted(notes_dir.glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except Exception:
                pass
    return out


@pytest.fixture
def ws():
    return _workspace()


# ---------------------------------------------------------------------------
# Session 0 — scaffold + `add`
# ---------------------------------------------------------------------------

@pytest.mark.session(0)
def test_s0_notes_binary_installed(ws):
    """`notes --help` returns success and mentions the `add` command."""
    r = _run_notes(ws, "--help")
    assert r.returncode == 0, f"notes --help failed: {r.stderr}"
    assert "add" in r.stdout.lower()


@pytest.mark.session(0)
def test_s0_add_creates_json(ws):
    """`notes add` persists a new note (any backend: JSON files or SQLite)."""
    before = len(_load_all_notes(ws))
    r = _run_notes(ws, "add", "--title", "Test Paper",
                   "--body", "Body of the test note",
                   "--tags", "ml,memory")
    assert r.returncode == 0, f"notes add failed: {r.stderr}"
    assert len(_load_all_notes(ws)) > before, "no new note persisted after add"


@pytest.mark.session(0)
def test_s0_schema_has_required_fields(ws):
    """A freshly-added note has title/body/tags/id fields."""
    _run_notes(ws, "add", "--title", "Schema Check",
               "--body", "Body", "--tags", "probe")
    notes = _load_all_notes(ws)
    assert notes, "no notes found in any backend"
    latest = max(notes, key=lambda n: n.get("id", 0))
    for key in ("id", "title", "tags", "body"):
        assert key in latest, f"note missing '{key}'"


# ---------------------------------------------------------------------------
# Session 1 — ls / show / rm / search
# ---------------------------------------------------------------------------

@pytest.mark.session(1)
def test_s1_ls_lists_all(ws):
    r = _run_notes(ws, "ls")
    assert r.returncode == 0
    # At least one prior note should be listed (backend-agnostic count)
    if len(_load_all_notes(ws)) > 0:
        assert r.stdout.strip() != "", "ls produced empty output despite existing notes"


@pytest.mark.session(1)
def test_s1_show_prints_note(ws):
    notes = _load_all_notes(ws)
    if not notes:
        _run_notes(ws, "add", "--title", "Show Target",
                   "--body", "Show body", "--tags", "t1")
        notes = _load_all_notes(ws)
    nid = notes[0]["id"]
    r = _run_notes(ws, "show", str(nid))
    assert r.returncode == 0, f"show failed: {r.stderr}"


@pytest.mark.session(1)
def test_s1_rm_deletes_note(ws):
    _run_notes(ws, "add", "--title", "RM Target",
               "--body", "Body", "--tags", "tmp")
    notes = _load_all_notes(ws)
    nid = max(notes, key=lambda n: n.get("id", 0))["id"]
    r = _run_notes(ws, "rm", str(nid))
    assert r.returncode == 0
    remaining_ids = [n.get("id") for n in _load_all_notes(ws)]
    assert nid not in remaining_ids


@pytest.mark.session(1)
def test_s1_search_substring(ws):
    _run_notes(ws, "add", "--title", "UniqueSearchToken xyzzy",
               "--body", "Body", "--tags", "search")
    r = _run_notes(ws, "search", "xyzzy")
    assert r.returncode == 0
    assert "xyzzy" in r.stdout.lower()


# ---------------------------------------------------------------------------
# Session 2 — filter-by-tag / sort-by-tag (interference pair)
# ---------------------------------------------------------------------------

@pytest.mark.session(2)
def test_s2_filter_by_tag_returns_subset(ws):
    _run_notes(ws, "add", "--title", "FilterA", "--body", "B", "--tags", "tagA")
    _run_notes(ws, "add", "--title", "FilterB", "--body", "B", "--tags", "tagB")
    r = _run_notes(ws, "filter-by-tag", "tagA")
    assert r.returncode == 0
    assert "FilterA" in r.stdout
    assert "FilterB" not in r.stdout, "filter-by-tag returned notes from wrong tag"


@pytest.mark.session(2)
def test_s2_sort_by_tag_preserves_count(ws):
    # record count before, run sort, count should equal (backend-agnostic)
    total_before = len(_load_all_notes(ws))
    r = _run_notes(ws, "sort-by-tag")
    assert r.returncode == 0
    # sort-by-tag should output ALL notes (count matches)
    output_lines = [l for l in r.stdout.splitlines() if l.strip()]
    assert len(output_lines) >= total_before, \
        f"sort-by-tag dropped notes: {len(output_lines)} < {total_before}"


# ---------------------------------------------------------------------------
# Session 3 — schema revision: `priority` field added
# ---------------------------------------------------------------------------

@pytest.mark.session(3)
def test_s3_priority_present_on_new_notes(ws):
    _run_notes(ws, "add", "--title", "S3NewNote",
               "--body", "Body", "--tags", "s3")
    notes = _load_all_notes(ws)
    assert notes, "no notes found in any backend"
    note = max(notes, key=lambda n: n.get("id", 0))
    assert "priority" in note, "new note missing 'priority' field after schema revision"
    try:
        pval = int(note["priority"])
    except (TypeError, ValueError):
        pval = None
    assert pval is not None and 1 <= pval <= 5, f"priority invalid: {note['priority']!r}"


@pytest.mark.session(3)
def test_s3_existing_notes_migrated(ws):
    """All existing notes should have 'priority' after migration (any backend)."""
    notes = _load_all_notes(ws)
    missing = [n.get("id") for n in notes if "priority" not in n]
    assert not missing, f"notes not migrated: {missing[:5]}"


@pytest.mark.session(3)
def test_s3_prior_commands_still_work(ws):
    """Regression: add/ls/search still work after schema change."""
    _run_notes(ws, "add", "--title", "Regression", "--body", "B", "--tags", "r")
    r = _run_notes(ws, "ls")
    assert r.returncode == 0, "ls broken after schema migration"


# ---------------------------------------------------------------------------
# Session 4 — holdout; no new functional tests, just re-run accumulated ones
# ---------------------------------------------------------------------------

@pytest.mark.session(4)
def test_s4_summary_all_prior_sessions_intact(ws):
    """Smoke check at the end: all core commands still respond."""
    for sub in ("--help", "ls", "search", "filter-by-tag", "sort-by-tag"):
        args = sub.split()
        if sub == "search":
            args = ["search", "anything"]
        elif sub == "filter-by-tag":
            args = ["filter-by-tag", "tagA"]
        elif sub == "sort-by-tag":
            args = ["sort-by-tag"]
        r = _run_notes(ws, *args)
        assert r.returncode == 0, f"command '{sub}' broken at session 4: {r.stderr}"


# ---------------------------------------------------------------------------
# Session 5 — collections feature scaffold
# ---------------------------------------------------------------------------

@pytest.mark.session(5)
def test_s5_col_create_command(ws):
    # Unique name so col-create never collides with a collection the agent
    # already created in this persistent workspace. (Fixes the stateful-test
    # bug where the fixed name "alpha-project" pre-existed and made every model
    # fail with "already exists".)
    import uuid
    name = "coltest_" + uuid.uuid4().hex[:8]
    r = _run_notes(ws, "col-create", name)
    assert r.returncode == 0, f"col-create failed: {r.stderr}"


@pytest.mark.session(5)
def test_s5_col_add_command(ws):
    import uuid
    col = "coladd_" + uuid.uuid4().hex[:8]
    _run_notes(ws, "col-create", col)
    # Use a real existing note id (backend-agnostic), not the assumed "1".
    notes = _load_all_notes(ws)
    if not notes:
        _run_notes(ws, "add", "--title", "ColTest", "--body", "b", "--tags", "x")
        notes = _load_all_notes(ws)
    nid = notes[0]["id"] if notes else 1
    r = _run_notes(ws, "col-add", col, str(nid))
    assert r.returncode == 0, f"col-add failed: {r.stderr}"


# ---------------------------------------------------------------------------
# Session 6 — more collection commands
# ---------------------------------------------------------------------------

@pytest.mark.session(6)
def test_s6_col_ls(ws):
    r = _run_notes(ws, "col-ls")
    assert r.returncode == 0


@pytest.mark.session(6)
def test_s6_col_show(ws):
    _run_notes(ws, "col-create", "showcol")
    r = _run_notes(ws, "col-show", "showcol")
    assert r.returncode == 0


@pytest.mark.session(6)
def test_s6_col_rm_does_not_delete_notes(ws):
    _run_notes(ws, "add", "--title", "PersistentNote", "--body", "x", "--tags", "p")
    _run_notes(ws, "col-create", "rmcol")
    _run_notes(ws, "col-rm", "rmcol")
    r = _run_notes(ws, "search", "PersistentNote")
    assert r.returncode == 0
    assert "PersistentNote" in r.stdout, "col-rm accidentally deleted underlying notes"


# ---------------------------------------------------------------------------
# Session 7 — collection-level interference pair
# ---------------------------------------------------------------------------

@pytest.mark.session(7)
def test_s7_col_filter_returns_subset(ws):
    r = _run_notes(ws, "col-filter", "anytag")
    # Command must exist; empty result is fine
    assert r.returncode == 0


@pytest.mark.session(7)
def test_s7_col_sort_returns_all(ws):
    r = _run_notes(ws, "col-sort", "name")
    assert r.returncode == 0


# ---------------------------------------------------------------------------
# Session 8 — SQLite migration
# ---------------------------------------------------------------------------

@pytest.mark.session(8)
def test_s8_sqlite_db_exists(ws):
    db = ws / "notes.db"
    assert db.exists(), "expected SQLite DB at notes.db after migration"


@pytest.mark.session(8)
def test_s8_commands_still_work_after_migration(ws):
    for cmd in (["ls"], ["col-ls"], ["add", "--title", "PostMig", "--body", "b", "--tags", "m"]):
        r = _run_notes(ws, *cmd)
        assert r.returncode == 0, f"command {cmd} broken after migration: {r.stderr}"


# ---------------------------------------------------------------------------
# Session 9 — cycle-2 holdout smoke
# ---------------------------------------------------------------------------

@pytest.mark.session(9)
def test_s9_all_cycles_intact(ws):
    for args in (["--help"], ["ls"], ["col-ls"], ["filter-by-tag", "any"],
                 ["col-filter", "any"]):
        r = _run_notes(ws, *args)
        assert r.returncode == 0, f"{args} broken at session 9: {r.stderr}"
