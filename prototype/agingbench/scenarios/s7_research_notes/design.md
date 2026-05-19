# S7+ Research-Notes CLI — Design Spec

**Purpose:** Extend S7 (self-planning Tier-2) with a real production-grade coding
task that exercises all four aging mechanisms via the same FactGraph apparatus
used by S1–S6. Produces graded scoring signals that do not saturate.

**Task:** Build a research-notes CLI in Python over 5 sessions. Each session
has a natural task, held-out FactGraph probes, and a pytest suite that verifies
functional correctness.

## Task spec

A CLI tool `notes` that lets a researcher store, search, tag, and cite academic
paper notes. Target: ~400–600 LOC, ~30 pytest tests.

## Per-session structure

Each session has:
  - **User message** — the task to perform
  - **FactGraph deltas** — what facts enter the graph, which get revised, what
    interferes with what, what accumulates
  - **Held-out probes** — questions the benchmark poses AFTER the task is done,
    without letting the agent re-read the conversation; must rely on workspace
    files
  - **Test suite slice** — which pytest cases should now pass

### Session 0 — Scaffold

**Task prompt:**
> Create a Python CLI called `notes` using Click. Define a note schema with
> these fields: `id` (int auto-increment), `title` (str), `tags` (list of str),
> `body` (str), `citation` (str, optional, BibTeX-format). Store notes as
> individual JSON files under `notes_data/`. Implement `notes add` as the
> first command. Save your design decisions to `notes/plan.md` under the
> workspace root.

**FactGraph deltas:**
  - `f1`: schema = {id, title, tags, body, citation}  — *fact*
  - `f2`: storage = per-note JSON files under `notes_data/`  — *fact*
  - `f3`: CLI framework = Click  — *fact*
  - Σ init: `test_count` accumulator = 0

**Held-out probes (end of session):**
  - P1 (compare): "Which fields are in the note schema? List them."
    → tests agent recorded `f1` somewhere readable
  - P2 (standalone): "What library does the CLI use for command parsing?"
    → tests `f3`

**Test slice (written upfront, runs at session end):**
  - `test_session0_schema_fields_exist()` — import, inspect Note dataclass
  - `test_session0_add_command_exists()` — `notes --help` shows `add`
  - `test_session0_add_creates_json()` — `notes add ...` writes to `notes_data/`

### Session 1 — CRUD + Search

**Task prompt:**
> Add four more commands: `ls` (list all notes), `show <id>` (print one note),
> `rm <id>` (delete), and `search <query>` (substring match on title + body).
> Make sure your previous `add` still works. Keep `notes/plan.md` up to date.

**FactGraph deltas:**
  - `f4`: search = substring match on title + body  — *fact*
  - `f5`: CLI commands so far = {add, ls, show, rm, search}  — *fact*
  - Σ: `test_count` += 4 (four new commands → four new tests)

**Held-out probes:**
  - P3 (synthesize): "What is the total count of commands in the CLI now?
    → tests accumulator
  - P4 (dep, d=1): "Where are notes stored on disk?"
    → tests `f2` recall from session 0

**Test slice:**
  - `test_session1_ls_lists_all()`, `test_session1_show_prints()`,
    `test_session1_rm_deletes()`, `test_session1_search_substring()`

### Session 2 — Filtering + interference pair

**Task prompt:**
> Add two commands: `filter-by-tag <tag>` (returns notes with that tag) and
> `sort-by-tag` (sorts notes alphabetically by first tag). These are
> intentionally similar — make sure they are distinct. Do not break prior tests.

**FactGraph deltas:**
  - `f6a`: filter-by-tag → subset of notes  — *fact*
  - `f6b`: sort-by-tag → all notes, reordered  — *fact*
  - I: interference pair (`f6a` ⇆ `f6b`, shared term = "by-tag")
  - Σ: `test_count` += 2

**Held-out probes:**
  - P5 (interference): "Which command returns only the notes matching a tag —
    `filter-by-tag` or `sort-by-tag`? And what does the other do?"
    → tests `interference_resistance`

**Test slice:**
  - `test_session2_filter_returns_subset()`,
    `test_session2_sort_preserves_all()`

### Session 3 — Schema revision (version chain + lifecycle event)

**Task prompt:**
> Refactor the note schema: add a new required field `priority` (int, 1–5,
> default 3). Migrate existing notes (add `priority: 3` to any note missing it).
> Update `plan.md` to reflect the new schema. Run the test suite and fix any
> regressions.

**FactGraph deltas:**
  - `f1 → f1'`: schema now includes `priority`  — *version chain*
  - `e_k`: lifecycle event — schema migration (mirrors S6's recompact)
  - Σ: `test_count` += 2

**Held-out probes:**
  - P6 (version-accuracy): "What are the current required fields in the note
    schema? (Do NOT include fields from older versions that have been removed.)"
    → tests `version_accuracy`

**Test slice:**
  - `test_session3_priority_field_present()`,
    `test_session3_existing_notes_migrated()`,
    all prior tests still pass

### Session 4 — Holdout: dependency-chain probing

**Task prompt:**
> No new feature. Answer the following questions using only your workspace
> files (notes/plan.md and the code). Do not rely on conversation memory.

**FactGraph deltas:**
  - no new facts — pure measurement session

**Held-out probes (the full battery):**
  - P7 (dep, d=3): "What is the CURRENT note schema, and which command returns
    a subset of notes by matching a tag? (requires `f1'` and `f6a`, NOT `f1`
    or `f6b`)"
  - P8 (accumulator-final): "How many CLI commands exist in total, and how many
    tests have been written so far?"
  - P9 (compare): "Which two commands contain 'by-tag' in their name, and what
    does each do?"
  - P10 (source-attribution): "When did you decide to use Click for CLI parsing
    — session 0, 1, 2, 3, or not yet?"

**Test slice:** all accumulated tests run; `session4_all_tests_pass()` summary.

## FactGraph summary

| Element | Count | Sessions active |
|---|---|---|
| Facts ($\mathcal{F}$) | 8 (f1, f2, f3, f4, f5, f6a, f6b, f1') | 0–3 |
| Version chains | 1 (f1 → f1') | revision at session 3 |
| Dependency edges ($\mathcal{E}$) | 4 (P4 d=1, P7 d=3, P8 cross-session, P9 d=2) | 1–4 |
| Interference pairs ($\mathcal{I}$) | 1 (f6a ⇆ f6b) | 2 |
| Accumulators ($\Sigma$) | 1 (`test_count`) | 0–4 |
| Lifecycle events | 1 (schema migration at session 3) | 3 |

## Mechanism coverage (columns in Table 3 filled, not saturated)

| Mechanism | Metric | Source |
|---|---|---|
| Compression | `keyword_m(t)` of plan.md + code | canonical keywords from facts survive in agent's written files |
| Interference | `interference_resistance` on P5 | does agent distinguish f6a from f6b? |
| Revision (explicit) | `version_accuracy` on P6 | does agent cite f1' not f1? |
| Revision (latent) | `accumulator_error` on P3, P8 | |gold_test_count − agent_answer| |
| Maintenance | Δ_{pre/post}(session 3) | recall curve slope before/after e_k |
| Cross-session dep | `chain_recall(d)` on P4 (d=1), P9 (d=2), P7 (d=3) | |

All six aging-diagnostic columns become populated, graded, and non-flat.

## Scoring layers

Three signals combined:

1. **Functional test suite** (pytest) — ground-truth "does the code work?"
2. **FactGraph probe scoring** — probe responses graded by keyword match
   against expected answer (same scorer as S1–S6)
3. **Workspace fidelity** — ratio of canonical keywords (from facts) present
   in `plan.md` + written code

## Implementation plan

1. **Generator** (`generators/s7_generator.py`):
   - Emits per-session `{task_prompt, probes, factgraph_deltas}`
   - Factgraph constructed deterministically from seed
2. **Runner** (`runner/s7_runner.py`):
   - Extends `S7Runner`
   - After each session's task, injects probe prompts via `adapter.send_message`
   - Runs pytest on workspace at session end, captures test outcomes
   - Scores probes + pytest + workspace fidelity, writes per-session metrics
3. **Pytest suite** (`scenarios/s7_research_notes/tests/test_cli.py`):
   - ~30 tests, grouped by session (session-0 tests, session-1 tests, …)
   - Each test auto-skips if its session hasn't run yet
4. **SUT YAML** (`registry/suts/openhands/openhands_gpt4omini_s7.yaml`)
5. **CLI dispatch** (`cli/runners.py::_run_s7`) — copies S7 dispatch structure
