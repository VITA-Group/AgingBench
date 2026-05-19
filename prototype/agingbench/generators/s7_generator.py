"""
agingbench/generators/s7_generator.py — S7+ Research-Notes CLI generator.

Emits a hand-designed, FactGraph-aware 5-session coding task. Unlike S1-S6
which sample from seeded pools, S7+ uses a fixed script because the target
task (a CLI notes app with specific commands and a schema revision) is
curated to exercise every aging mechanism exactly once.

Output format matches S7Generator: list of session dicts with 'task_prompt',
'eval_probes', plus FactGraph structure in the companion dependency_graph.

See scenarios/s7_research_notes/design.md for the full spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .fact_graph import FactGraph
from .pressure_config import PressureConfig


@dataclass
class S7Probe:
    id: str
    session: int
    prompt: str
    expected_keywords: list[str]       # must appear in agent response
    forbidden_keywords: list[str] = field(default_factory=list)  # must NOT appear
    probe_type: str = "standalone"     # compare / trend / synthesize / standalone
    chain_depth: int = 1
    metric_tag: str = "recall"         # recall | version_accuracy | interference | accumulator


def _session_0():
    task = (
        "Create a Python CLI called `notes` using Click. It should be "
        "installable via `pip install -e .`. Define a note schema with these "
        "fields: id (int, auto-increment starting from 1), title (str), "
        "tags (list of str), body (str), citation (str, optional, BibTeX format). "
        "Store each note as an individual JSON file under a `notes_data/` "
        "directory at the workspace root. Implement the `notes add` command "
        "with flags --title, --body, --tags, --citation. "
        "Save your design decisions and schema to `notes/plan.md`."
    )
    probes = [
        S7Probe(
            id="s0_p1", session=0,
            prompt="List the fields in the current note schema (one per line).",
            expected_keywords=["id", "title", "tags", "body", "citation"],
            probe_type="compare", chain_depth=1, metric_tag="recall",
        ),
        S7Probe(
            id="s0_p2", session=0,
            prompt="What Python library did you choose for CLI command parsing?",
            expected_keywords=["click"],
            probe_type="standalone", chain_depth=1, metric_tag="recall",
        ),
    ]
    return task, probes


def _session_1():
    task = (
        "Add four new commands to the `notes` CLI: "
        "(1) `ls` — list all notes with their id and title; "
        "(2) `show <id>` — print the full contents of one note; "
        "(3) `rm <id>` — delete a note by id; "
        "(4) `search <query>` — substring match against title and body, "
        "case-insensitive. Make sure the existing `add` command still works. "
        "Update `notes/plan.md` with the new command list."
    )
    probes = [
        S7Probe(
            id="s1_p3", session=1,
            prompt="How many CLI commands does the notes tool support so far? "
                   "List them by name.",
            expected_keywords=["5", "five", "add", "ls", "show", "rm", "search"],
            probe_type="synthesize", chain_depth=2,
            metric_tag="accumulator",
        ),
        S7Probe(
            id="s1_p4", session=1,
            prompt="Where on disk are notes stored, and in what file format?",
            expected_keywords=["notes_data", "json"],
            probe_type="standalone", chain_depth=1, metric_tag="recall",
        ),
    ]
    return task, probes


def _session_2():
    task = (
        "Add two commands that both involve tags: "
        "(1) `filter-by-tag <tag>` — returns only the notes whose tags list "
        "contains that tag (a SUBSET of notes); "
        "(2) `sort-by-tag` — returns ALL notes, sorted alphabetically by their "
        "first tag (preserves the full count). "
        "These two are easy to confuse — keep them distinct. "
        "Do not break any existing tests."
    )
    probes = [
        S7Probe(
            id="s2_p5", session=2,
            prompt="Which of your two tag-related commands returns a subset "
                   "of notes, and which returns all notes in a new order? "
                   "Name both explicitly.",
            expected_keywords=["filter-by-tag", "subset", "sort-by-tag", "all"],
            forbidden_keywords=[],
            probe_type="compare", chain_depth=1,
            metric_tag="interference",
        ),
    ]
    return task, probes


def _session_3():
    task = (
        "Refactor the note schema: add a new field `priority` (int, range 1 "
        "through 5, default 3). Every NEW note must have a priority. "
        "All EXISTING notes in `notes_data/` must be migrated: if a note's "
        "JSON file is missing `priority`, add it with value 3. "
        "Update `notes/plan.md` to describe the new schema. "
        "Then run the test suite and fix any regressions."
    )
    probes = [
        S7Probe(
            id="s3_p6", session=3,
            prompt="What are the CURRENT required fields in the note schema? "
                   "List them exactly as they are NOW, not as they were before "
                   "any refactors.",
            expected_keywords=["id", "title", "tags", "body", "priority"],
            forbidden_keywords=[],
            probe_type="compare", chain_depth=2,
            metric_tag="version_accuracy",
        ),
    ]
    return task, probes


def _session_5():
    # Cycle 2 scaffold — "collections" feature (groups of notes)
    task = (
        "Add a new `collections` feature to the notes CLI. A collection groups "
        "several notes under a named label (e.g., 'project-alpha', 'reading-list'). "
        "Store each collection as a JSON file under `collections_data/`. Implement: "
        "(1) `notes col-create <name>` — create an empty collection; "
        "(2) `notes col-add <collection_name> <note_id>` — add a note to a collection. "
        "Update `notes/plan.md` with the collection schema: {name (str), "
        "note_ids (list of int), created (str, ISO timestamp)}."
    )
    probes = [
        S7Probe(
            id="s5_p11", session=5,
            prompt="What are the fields in the collection schema?",
            expected_keywords=["name", "note_ids", "created"],
            probe_type="compare", chain_depth=1, metric_tag="recall",
        ),
        S7Probe(
            id="s5_p12", session=5,
            prompt="Where on disk are collections stored, and in what format?",
            expected_keywords=["collections_data", "json"],
            probe_type="standalone", chain_depth=1, metric_tag="recall",
        ),
    ]
    return task, probes


def _session_6():
    # More collection commands
    task = (
        "Add three more collection commands: "
        "(1) `col-ls` — list all collections with their sizes; "
        "(2) `col-show <name>` — print the notes in a collection; "
        "(3) `col-rm <name>` — delete a collection (but NOT the underlying notes). "
        "Do not break any prior commands. Update `notes/plan.md`."
    )
    probes = [
        S7Probe(
            id="s6_p13", session=6,
            prompt="How many total CLI commands does the tool now have "
                   "(counting ALL commands across the notes and collections "
                   "features)? List them.",
            expected_keywords=["12", "twelve", "col-create", "col-add",
                               "col-ls", "col-show", "col-rm"],
            probe_type="synthesize", chain_depth=3, metric_tag="accumulator",
        ),
    ]
    return task, probes


def _session_7():
    # Cycle 2 interference pair: "col-filter" vs "col-sort"
    task = (
        "Add two collection filter/sort commands: "
        "(1) `col-filter <tag>` — return only collections that contain at "
        "least one note with that tag (a SUBSET of collections); "
        "(2) `col-sort <by>` — sort ALL collections by field (name|size|created). "
        "These pair with the earlier `filter-by-tag` / `sort-by-tag` at the "
        "note level. Keep collection-level and note-level commands distinct."
    )
    probes = [
        S7Probe(
            id="s7_p14", session=7,
            prompt="You now have TWO 'filter' commands (one at the note level, "
                   "one at the collection level) and TWO 'sort' commands. "
                   "Which returns a SUBSET at the collection level, and which "
                   "operates on notes?",
            expected_keywords=["col-filter", "subset", "filter-by-tag", "notes"],
            forbidden_keywords=["col-sort returns subset"],
            probe_type="compare", chain_depth=2, metric_tag="interference",
        ),
    ]
    return task, probes


def _session_8():
    # Cycle 2 major revision — migrate storage from JSON files to SQLite
    # This also touches cycle 1 (notes) AND cycle 2 (collections)
    task = (
        "Perform a major refactor: migrate ALL storage from individual JSON "
        "files (both `notes_data/` and `collections_data/`) to a single SQLite "
        "database at `notes.db`. Create two tables: `notes` and `collections`. "
        "Migrate all existing data into the database. The CLI commands should "
        "continue to work identically from the user's perspective. "
        "Update `notes/plan.md` to describe the new storage layer and retain "
        "the priority field from the earlier schema revision."
    )
    probes = [
        S7Probe(
            id="s8_p15", session=8,
            prompt="What is the CURRENT storage backend for both notes and "
                   "collections? State both the file name and the tables.",
            expected_keywords=["sqlite", "notes.db", "notes", "collections"],
            forbidden_keywords=["notes_data", "collections_data"],
            probe_type="synthesize", chain_depth=3, metric_tag="version_accuracy",
        ),
    ]
    return task, probes


def _session_9():
    # Cycle 2 holdout — multi-mechanism long-horizon probes
    task = (
        "No new feature this session. Answer the following questions using "
        "ONLY your workspace files (notes/plan.md, the code, and the database "
        "schema). Do not guess."
    )
    probes = [
        S7Probe(
            id="s9_p16", session=9,
            prompt="What is the CURRENT note schema, and what is the CURRENT "
                   "storage backend? Provide both exactly as they are NOW.",
            expected_keywords=["id", "title", "tags", "body", "priority",
                               "sqlite", "notes.db"],
            forbidden_keywords=["notes_data", "json files"],
            probe_type="synthesize", chain_depth=4, metric_tag="version_accuracy",
        ),
        S7Probe(
            id="s9_p17", session=9,
            prompt="How many total CLI commands exist across BOTH the notes "
                   "and collections features?",
            expected_keywords=["14", "fourteen"],
            probe_type="synthesize", chain_depth=4, metric_tag="accumulator",
        ),
        S7Probe(
            id="s9_p18", session=9,
            prompt="List ALL 'filter' commands and ALL 'sort' commands you "
                   "have built, and state which operate on notes vs collections.",
            expected_keywords=["filter-by-tag", "sort-by-tag", "col-filter",
                               "col-sort", "notes", "collections"],
            probe_type="compare", chain_depth=3, metric_tag="interference",
        ),
        S7Probe(
            id="s9_p19", session=9,
            prompt="Which came first in your development history — the notes "
                   "schema with priority field, or the SQLite migration? "
                   "Reference the relevant sessions (0–9).",
            expected_keywords=["priority", "session 3", "sqlite", "session 8"],
            probe_type="standalone", chain_depth=5, metric_tag="recall",
        ),
    ]
    return task, probes


def _session_long_horizon(session_idx: int, seed: int = 42):
    """Probe-only session for N > 9.

    No new feature work. Emits a probe set sampled deterministically from
    the canonical session-9 holdout probes, with chain_depth incremented
    by (session_idx - 9) to reflect the longer temporal gap between write
    and recall. Measures pure memory decay past the fully-built substrate.
    """
    import random
    task = (
        "No new feature work this session. Please briefly re-read your "
        "notes/plan.md to refresh yourself on the current state of the "
        "project, then answer the following questions using only your "
        "workspace files. Do not guess."
    )
    _, cycle2_probes = _session_9()
    gap = session_idx - 9
    rng = random.Random(seed)
    # Sample 3 probes from the session-9 pool (keeps API cost bounded)
    sampled = rng.sample(cycle2_probes, min(3, len(cycle2_probes)))
    probes = []
    for orig in sampled:
        probes.append(S7Probe(
            id=f"s{session_idx}_lh_{orig.id}",
            session=session_idx,
            prompt=orig.prompt,
            expected_keywords=list(orig.expected_keywords),
            forbidden_keywords=list(orig.forbidden_keywords),
            probe_type=orig.probe_type,
            chain_depth=orig.chain_depth + gap,
            metric_tag=orig.metric_tag,
        ))
    return task, probes


def _session_4():
    # Holdout session — NO new task; the session is entirely probes.
    task = (
        "No new feature this session. Answer the following questions using "
        "only your workspace files (notes/plan.md and the code you've written). "
        "Do not guess. If a fact is in plan.md or the code, read it and report "
        "it accurately."
    )
    probes = [
        S7Probe(
            id="s4_p7", session=4,
            prompt="What is the CURRENT note schema, and which command returns "
                   "ONLY the notes that match a specific tag?",
            expected_keywords=["id", "title", "tags", "body", "priority",
                               "filter-by-tag"],
            forbidden_keywords=["sort-by-tag"],  # wrong answer
            probe_type="synthesize", chain_depth=3,
            metric_tag="recall",
        ),
        S7Probe(
            id="s4_p8", session=4,
            prompt="How many CLI commands does the notes tool have in total "
                   "right now? Count them from the current command list.",
            expected_keywords=["7", "seven"],
            probe_type="synthesize", chain_depth=3,
            metric_tag="accumulator",
        ),
        S7Probe(
            id="s4_p9", session=4,
            prompt="Which two commands in the CLI contain 'by-tag' in their "
                   "name, and what does each one do?",
            expected_keywords=["filter-by-tag", "sort-by-tag",
                               "subset", "sort"],
            probe_type="compare", chain_depth=2,
            metric_tag="interference",
        ),
        S7Probe(
            id="s4_p10", session=4,
            prompt="In which session (0, 1, 2, 3, or 4) did you originally "
                   "decide to use Click for CLI parsing?",
            expected_keywords=["0", "zero", "first", "session 0"],
            probe_type="standalone", chain_depth=4,
            metric_tag="recall",
        ),
    ]
    return task, probes


class S7Generator:
    """Fixed-script generator for the S7+ research-notes coding task."""

    def __init__(self, seed: int = 42, pressure: PressureConfig | None = None):
        self.seed = seed
        self.pressure = pressure or PressureConfig.medium()

    def generate(self, n_sessions: int = 5) -> dict:
        if n_sessions < 1:
            raise ValueError("S7+ requires at least 1 session")
        # Scripted 10-session program: cycle 1 (notes CLI, sessions 0-4) and
        # cycle 2 (collections + SQLite migration, sessions 5-9). For N > 10
        # we add probe-only "long-horizon" sessions that re-sample probes
        # from the accumulated pool to measure pure memory decay past the
        # fully-built substrate.
        builders = [
            _session_0, _session_1, _session_2, _session_3, _session_4,
            _session_5, _session_6, _session_7, _session_8, _session_9,
        ]
        sessions = []
        for i in range(n_sessions):
            if i < len(builders):
                task, probes = builders[i]()
            else:
                # Long-horizon: probe-only session (no new task). Re-samples
                # holdout probes from session 9's pool with depth incremented
                # to reflect the longer temporal gap.
                task, probes = _session_long_horizon(i, seed=self.seed + i)
            sessions.append({
                "session_idx": i,
                "task_prompt": task,
                "eval_probes": [
                    {
                        "id": p.id,
                        "prompt": p.prompt,
                        "expected_keywords": p.expected_keywords,
                        "forbidden_keywords": p.forbidden_keywords,
                        "probe_type": p.probe_type,
                        "chain_depth": p.chain_depth,
                        "metric_tag": p.metric_tag,
                    } for p in probes
                ],
            })

        # Build the FactGraph canonical structure that sessions operate on
        fg = FactGraph()
        # Session 0 facts
        f1 = fg.register_fact(0, "schema",
                              "note schema = {id, title, tags, body, citation}",
                              ["id", "title", "tags", "body", "citation"],
                              fact_id="f1")
        fg.register_fact(0, "storage",
                         "notes stored as per-note JSON under notes_data/",
                         ["notes_data", "json"], fact_id="f2")
        fg.register_fact(0, "framework", "CLI = Click", ["click"], fact_id="f3")
        fg.register_accumulator("test_count", 3.0, 0, domain="meta")
        # Session 1 adds commands
        fg.register_fact(1, "search",
                         "search = case-insensitive substring on title + body",
                         ["substring", "title", "body"], fact_id="f4")
        fg.register_fact(1, "commands",
                         "CLI commands = {add, ls, show, rm, search}",
                         ["add", "ls", "show", "rm", "search"], fact_id="f5")
        fg.add_delta("test_count", 4.0, 1, description="session 1 tests")
        # Session 2 interference
        f6a = fg.register_fact(2, "filter",
                               "filter-by-tag returns subset of notes matching tag",
                               ["filter-by-tag", "subset"], fact_id="f6a")
        f6b = fg.register_fact(2, "sort",
                               "sort-by-tag returns ALL notes reordered by first tag",
                               ["sort-by-tag", "all", "reorder"], fact_id="f6b")
        fg.add_interference("f6a", "f6b", shared_term="by-tag")
        fg.add_delta("test_count", 2.0, 2)
        # Session 3 version chain + lifecycle event
        fg.update_fact("f1",
                       new_content="note schema = {id, title, tags, body, citation, priority}",
                       new_keywords=["id", "title", "tags", "body", "citation", "priority"],
                       session=3, new_id="f1_v2")
        fg.add_delta("test_count", 2.0, 3)

        # Cycle 2 — sessions 5-9, only registered if requested
        if n_sessions > 5:
            fg.register_fact(5, "collections_schema",
                             "collection = {name, note_ids, created}",
                             ["name", "note_ids", "created"], fact_id="f7")
            fg.register_fact(5, "collections_storage",
                             "collections stored as JSON under collections_data/",
                             ["collections_data", "json"], fact_id="f8")
            fg.add_delta("test_count", 2.0, 5)
        if n_sessions > 6:
            fg.register_fact(6, "col_commands",
                             "collection commands = col-create, col-add, col-ls, col-show, col-rm",
                             ["col-create", "col-add", "col-ls", "col-show", "col-rm"],
                             fact_id="f9")
            fg.add_delta("test_count", 3.0, 6)
        if n_sessions > 7:
            # Second interference pair: col-filter (subset) vs col-sort (all)
            fg.register_fact(7, "col_filter",
                             "col-filter returns subset of collections",
                             ["col-filter", "subset"], fact_id="f10a")
            fg.register_fact(7, "col_sort",
                             "col-sort returns ALL collections sorted",
                             ["col-sort", "all", "sort"], fact_id="f10b")
            fg.add_interference("f10a", "f10b", shared_term="col-")
            fg.add_delta("test_count", 2.0, 7)
        if n_sessions > 8:
            # Major storage revision: JSON files -> SQLite
            fg.update_fact("f2",
                           new_content="notes stored in SQLite at notes.db (notes table)",
                           new_keywords=["sqlite", "notes.db", "notes"],
                           session=8, new_id="f2_v2")
            fg.update_fact("f8",
                           new_content="collections stored in SQLite at notes.db (collections table)",
                           new_keywords=["sqlite", "notes.db", "collections"],
                           session=8, new_id="f8_v2")
            fg.add_delta("test_count", 3.0, 8)

        # Emit dependency edges from probes
        for s in sessions:
            for p in s["eval_probes"]:
                # map probe metric_tag to relevant fact_ids
                facts = []
                tag = p["metric_tag"]
                if tag == "version_accuracy":
                    facts = ["f1_v2"]
                elif tag == "interference":
                    facts = ["f6a", "f6b"]
                elif tag == "accumulator":
                    facts = []  # handled separately
                else:
                    # recall / cross-session
                    facts = [fid for fid in ["f1_v2", "f2", "f3", "f5", "f6a"]
                             if fid in fg.facts]
                if facts:
                    fg.add_dependency(
                        task_id=p["id"],
                        session=s["session_idx"],
                        fact_ids=facts,
                        dep_type=p["probe_type"],
                    )

        return {
            "scenario": "s7_research_notes",
            "seed": self.seed,
            "n_sessions": len(sessions),
            "sessions": sessions,
            "dependency_graph": fg.export(),
            "pressure_config": self.pressure.to_dict(),
            "lifecycle_events": [
                {"session": 3, "event_type": "schema_migration",
                 "description": "added required priority field; migrated existing notes"},
            ],
        }
