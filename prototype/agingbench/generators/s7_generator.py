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


import random as _random

# ---------------------------------------------------------------------------
# Procedural long-horizon extension (blocks >= 10)
# ---------------------------------------------------------------------------
# Reuses the five operation types the curriculum demonstrates — add-CRUD,
# schema-revision, storage-migration, interference-pair, recall-checkpoint —
# applied to FRESH targets in a varied, prerequisite-respecting schedule,
# continuing from the curriculum's end-of-block-9 state. Probes + pytest specs
# are derived from running state so ground truth stays exact. Curriculum
# builders _session_0.._session_9 are NOT touched.

_PROC_ENTITIES = [
    "author", "citation", "reminder", "attachment", "label",
    "folder", "template", "bookmark", "snapshot", "alias",
]
_PROC_FIELDS = [
    ("archived", "bool", "false"), ("pinned", "bool", "false"),
    ("color", "str", "none"), ("due_date", "str", "none"),
    ("word_count", "int", "0"), ("starred", "bool", "false"),
    ("language", "str", "en"), ("source_url", "str", "none"),
]
_PROC_BACKENDS = [
    ("DuckDB", "notes.duckdb"), ("Parquet", "notes.parquet"),
    ("LMDB", "notes.lmdb"), ("MessagePack", "notes.msgpack"),
]
_PROC_INTERFERENCE = [
    ("export-tagged", "export-all", "export"),
    ("archive-stale", "archive-all", "archive"),
    ("pin-recent", "pin-all", "pin"),
    ("link-related", "link-all", "link"),
    ("merge-dupes", "merge-all", "merge"),
    ("compact-old", "compact-all", "compact"),
]

# Varied, prerequisite-respecting cadence over the five reused operation types.
_PROC_CADENCE = [
    "add_crud", "interference", "recall", "schema_revision",
    "add_crud", "add_crud", "recall", "storage_migration",
    "add_crud", "interference", "schema_revision", "recall",
]


class _S7ProceduralExtender:
    """Coherent S7 blocks for session_idx >= 10, continuing the curriculum's
    terminal state. Records FactGraph ops, lifecycle events, and data-driven
    pytest specs so runner/metrics see exact ground truth."""

    # Curriculum end-of-block-9 state (must match _session_0.._session_9 output).
    _BASE_COMMANDS = [
        "add", "ls", "show", "rm", "search",
        "filter-by-tag", "sort-by-tag",
        "col-create", "col-add", "col-ls", "col-show", "col-rm",
        "col-filter", "col-sort",
    ]
    _BASE_SCHEMA = ["id", "title", "tags", "body", "citation", "priority"]

    def __init__(self, seed: int = 42):
        rng = _random.Random(seed)
        self.commands = list(self._BASE_COMMANDS)      # running command list (14)
        self.schema = list(self._BASE_SCHEMA)          # running note schema
        self.backend = ("SQLite", "notes.db")          # current storage backend
        self.schema_fact_id = "f1_v2"                  # current schema fact version
        self.storage_fact_id = "f2_v2"                 # current storage fact version
        self._schema_ver = 2
        self._storage_ver = 2
        self._pf = 0
        self.entities = list(_PROC_ENTITIES); rng.shuffle(self.entities)
        self.fields = list(_PROC_FIELDS); rng.shuffle(self.fields)
        self.backends = list(_PROC_BACKENDS); rng.shuffle(self.backends)
        self.interf = list(_PROC_INTERFERENCE); rng.shuffle(self.interf)
        self.removed_fields = []                       # for forbidden_keywords
        self.event_log = []                            # (session, "kind:name") for provenance
        self.fact_ops = []
        self.lifecycle_events = []
        self.test_specs = []

    def _next_fact_id(self):
        self._pf += 1
        return f"pf{self._pf}"

    def next_block(self, session_idx):
        op = _PROC_CADENCE[(session_idx - 10) % len(_PROC_CADENCE)]
        # Fall back to a recall checkpoint when the relevant pool is exhausted
        # (keeps every emitted block coherent and ground-truth-exact).
        if op == "add_crud" and not self.entities:
            op = "recall"
        elif op == "schema_revision" and not self.fields:
            op = "recall"
        elif op == "storage_migration" and not self.backends:
            op = "recall"
        elif op == "interference" and not self.interf:
            op = "recall"
        return getattr(self, f"_op_{op}")(session_idx)

    # ---- operations -------------------------------------------------------
    def _op_add_crud(self, i):
        e = self.entities.pop(0)
        cmds = [f"{e}-add", f"{e}-ls", f"{e}-show", f"{e}-rm"]
        self.commands += cmds
        self.event_log.append((i, f"entity:{e}"))
        bname, bfile = self.backend
        task = (
            f"Add CRUD commands for a new `{e}` entity to the notes CLI: "
            f"`{e}-add --name <n>`, `{e}-ls`, `{e}-show <id>`, `{e}-rm <id>`. "
            f"Persist {e} records in the current {bname} backend ({bfile}, a new "
            f"`{e}s` table/section). Do NOT break any existing commands. "
            f"Update notes/plan.md with the new commands."
        )
        count = len(self.commands)
        probes = [
            S7Probe(id=f"s{i}_pcmd", session=i,
                    prompt="How many CLI commands does the tool support in total now? "
                           "Also list the names of the commands you just added.",
                    expected_keywords=[str(count)] + cmds,
                    probe_type="synthesize", chain_depth=2, metric_tag="accumulator"),
            S7Probe(id=f"s{i}_prec", session=i,
                    prompt=f"What does the `{e}-show` command do, and in which storage "
                           f"backend are `{e}` records persisted?",
                    expected_keywords=[f"{e}-show", bname.lower()],
                    probe_type="standalone", chain_depth=1, metric_tag="recall"),
        ]
        fid = self._next_fact_id()
        self.fact_ops.append({"kind": "register", "id": fid, "session": i,
                              "domain": f"{e}_commands",
                              "content": f"{e} commands = {', '.join(cmds)}",
                              "keywords": list(cmds)})
        for c in cmds:
            self.test_specs.append({"session": i, "kind": "command_runs", "command": c})
        self.test_specs.append({"session": i, "kind": "command_runs",
                                "command": f"{e}-add", "args": ["--name", f"probe_{e}"]})
        return task, probes

    def _op_schema_revision(self, i):
        field, ftype, default = self.fields.pop(0)
        self.schema.append(field)
        self._schema_ver += 1
        self.event_log.append((i, f"field:{field}"))
        task = (
            f"Refactor the note schema: add a new field `{field}` ({ftype}, default "
            f"{default}). Every NEW note must include it; migrate ALL existing notes "
            f"to add `{field}={default}` if missing. Update notes/plan.md and run the "
            f"test suite, fixing any regressions."
        )
        probes = [
            S7Probe(id=f"s{i}_pver", session=i,
                    prompt="List the CURRENT required fields of the note schema, exactly "
                           "as they are NOW after all refactors (not as they were before).",
                    expected_keywords=list(self.schema),
                    forbidden_keywords=list(self.removed_fields),
                    probe_type="synthesize", chain_depth=3, metric_tag="version_accuracy"),
        ]
        new_id = f"f1_v{self._schema_ver}"
        self.fact_ops.append({"kind": "update", "old_id": self.schema_fact_id,
                              "new_id": new_id, "session": i,
                              "content": f"note schema = {{{', '.join(self.schema)}}}",
                              "keywords": list(self.schema)})
        self.schema_fact_id = new_id
        self.lifecycle_events.append({
            "session": i, "event_type": "schema_migration",
            "description": f"added required {field} field; migrated existing notes"})
        self.test_specs.append({"session": i, "kind": "schema_field", "field": field})
        return task, probes

    def _op_storage_migration(self, i):
        new_name, new_file = self.backends.pop(0)
        old_name, old_file = self.backend
        self.backend = (new_name, new_file)
        self._storage_ver += 1
        self.event_log.append((i, f"backend:{new_name}"))
        task = (
            f"Perform a storage migration: move ALL data from {old_name} ({old_file}) "
            f"to {new_name} at `{new_file}`. Migrate existing notes and any other "
            f"entities. All CLI commands must continue to work identically. Update "
            f"notes/plan.md to describe the new storage layer and retain all schema fields."
        )
        probes = [
            S7Probe(id=f"s{i}_pbk", session=i,
                    prompt="What is the CURRENT storage backend (name and file) for the "
                           "notes tool, exactly as it is NOW?",
                    expected_keywords=[new_name.lower(), new_file],
                    forbidden_keywords=[old_name.lower(), old_file],
                    probe_type="synthesize", chain_depth=3, metric_tag="version_accuracy"),
        ]
        new_id = f"f2_v{self._storage_ver}"
        self.fact_ops.append({"kind": "update", "old_id": self.storage_fact_id,
                              "new_id": new_id, "session": i,
                              "content": f"storage backend = {new_name} at {new_file}",
                              "keywords": [new_name.lower(), new_file]})
        self.storage_fact_id = new_id
        self.lifecycle_events.append({
            "session": i, "event_type": "storage_migration",
            "description": f"migrated storage {old_name} -> {new_name}"})
        self.test_specs.append({"session": i, "kind": "backend_file", "file": new_file})
        self.test_specs.append({"session": i, "kind": "smoke", "commands": ["ls"]})
        return task, probes

    def _op_interference(self, i):
        a, b, shared = self.interf.pop(0)
        self.commands += [a, b]
        self.event_log.append((i, f"interference:{shared}"))
        task = (
            f"Add two commands that are easy to confuse: `{a}` — returns only the notes "
            f"matching a {shared} criterion (a SUBSET); `{b}` — applies the {shared} "
            f"operation to ALL notes (preserves the full set). Keep them clearly "
            f"distinct. Do not break any existing commands."
        )
        probes = [
            S7Probe(id=f"s{i}_pintf", session=i,
                    prompt=f"You now have `{a}` and `{b}`. Which returns a SUBSET of notes, "
                           f"and which operates on ALL notes? Name both explicitly.",
                    expected_keywords=[a, "subset", b, "all"],
                    probe_type="compare", chain_depth=1, metric_tag="interference"),
        ]
        ida, idb = self._next_fact_id(), self._next_fact_id()
        self.fact_ops.append({"kind": "register", "id": ida, "session": i, "domain": shared,
                              "content": f"{a} returns subset", "keywords": [a, "subset"]})
        self.fact_ops.append({"kind": "register", "id": idb, "session": i, "domain": shared,
                              "content": f"{b} returns all", "keywords": [b, "all"]})
        self.fact_ops.append({"kind": "interference", "a": ida, "b": idb, "shared": shared})
        self.test_specs.append({"session": i, "kind": "command_runs", "command": a})
        self.test_specs.append({"session": i, "kind": "command_runs", "command": b})
        return task, probes

    def _op_recall(self, i):
        task = (
            "No new feature this session. Re-read your notes/plan.md and the code, then "
            "answer the following questions using ONLY your workspace files. Do not guess."
        )
        count = len(self.commands)
        probes = [
            S7Probe(id=f"s{i}_rc_schema", session=i,
                    prompt="List the CURRENT note schema fields exactly as they are NOW.",
                    expected_keywords=list(self.schema),
                    forbidden_keywords=list(self.removed_fields),
                    probe_type="synthesize", chain_depth=4, metric_tag="version_accuracy"),
            S7Probe(id=f"s{i}_rc_count", session=i,
                    prompt="How many CLI commands exist in total right now?",
                    expected_keywords=[str(count)],
                    probe_type="synthesize", chain_depth=4, metric_tag="accumulator"),
            S7Probe(id=f"s{i}_rc_backend", session=i,
                    prompt="What is the current storage backend (name and file)?",
                    expected_keywords=[self.backend[0].lower(), self.backend[1]],
                    probe_type="standalone", chain_depth=3, metric_tag="recall"),
        ]
        if self.event_log:
            sess, what = self.event_log[0]
            kind, _, name = what.partition(":")
            label = "entity" if kind == "entity" else kind
            probes.append(S7Probe(
                id=f"s{i}_rc_prov", session=i,
                prompt=f"In which session did you first introduce the `{name}` {label}? "
                       f"Reference the session index.",
                expected_keywords=[str(sess)],
                probe_type="standalone", chain_depth=max(1, i - sess), metric_tag="recall"))
        self.test_specs.append({"session": i, "kind": "smoke", "commands": ["--help", "ls"]})
        return task, probes

    # ---- post-loop application -------------------------------------------
    def apply_to_factgraph(self, fg):
        for op in self.fact_ops:
            if op["kind"] == "register":
                fg.register_fact(op["session"], op["domain"], op["content"],
                                 op["keywords"], fact_id=op["id"])
            elif op["kind"] == "update":
                if op["old_id"] in fg.facts:
                    fg.update_fact(op["old_id"], new_content=op["content"],
                                   new_keywords=op["keywords"], session=op["session"],
                                   new_id=op["new_id"])
            elif op["kind"] == "interference":
                if op["a"] in fg.facts and op["b"] in fg.facts:
                    fg.add_interference(op["a"], op["b"], shared_term=op["shared"])


class S7Generator:
    """Fixed-script generator for the S7+ research-notes coding task."""

    def __init__(self, seed: int = 42, pressure: PressureConfig | None = None):
        self.seed = seed
        self.pressure = pressure or PressureConfig.medium()

    def generate(self, n_sessions: int = 5, extension_mode: str = "procedural") -> dict:
        if n_sessions < 1:
            raise ValueError("S7+ requires at least 1 session")
        # Scripted 10-session program: cycle 1 (notes CLI, sessions 0-4) and
        # cycle 2 (collections + SQLite migration, sessions 5-9). For N > 10,
        # extension_mode selects what the extra blocks are:
        #   "procedural" (default) — genuinely new feature work via the reused
        #       operation templates (add-CRUD / schema-revision / migration /
        #       interference / recall), continuing from the block-9 state.
        #   "holdout" — the original probe-only long-horizon re-sampler.
        # Curriculum blocks 0-9 are byte-identical in both modes.
        builders = [
            _session_0, _session_1, _session_2, _session_3, _session_4,
            _session_5, _session_6, _session_7, _session_8, _session_9,
        ]
        extender = _S7ProceduralExtender(seed=self.seed)
        sessions = []
        for i in range(n_sessions):
            if i < len(builders):
                task, probes = builders[i]()
            elif extension_mode == "holdout":
                # Long-horizon: probe-only session (no new task). Re-samples
                # holdout probes from session 9's pool with depth incremented
                # to reflect the longer temporal gap.
                task, probes = _session_long_horizon(i, seed=self.seed + i)
            else:
                # Procedural: a genuinely new block continuing from block-9 state.
                task, probes = extender.next_block(i)
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

        # Procedural blocks (>= 10) register their own facts/versions/interference.
        if extension_mode != "holdout":
            extender.apply_to_factgraph(fg)

        # Emit dependency edges from curriculum probes (blocks 0-9). Procedural
        # probes carry their own chain_depth; their edges are not routed through
        # this hardcoded curriculum mapping.
        for s in sessions:
            if s["session_idx"] >= len(builders):
                continue
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

        result = {
            "scenario": "s7_research_notes",
            "seed": self.seed,
            "n_sessions": len(sessions),
            "sessions": sessions,
            "dependency_graph": fg.export(),
            "pressure_config": self.pressure.to_dict(),
            "lifecycle_events": [
                {"session": 3, "event_type": "schema_migration",
                 "description": "added required priority field; migrated existing notes"},
            ] + (extender.lifecycle_events if extension_mode != "holdout" else []),
        }
        # Only attach data-driven pytest specs when procedural blocks were
        # emitted, so curriculum-only runs (n <= 10) keep their exact output.
        if extension_mode != "holdout" and extender.test_specs:
            result["generated_tests"] = extender.test_specs
        return result
