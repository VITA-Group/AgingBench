# S7 — Research-Notes Coding Task (Tier-2)

**Tier:** 2 (production-CLI agent: OpenHands, Claude Code)
**Metric Group:** G1 + G2 + G3 (mixed)
**Exposure Axis:** `t_steps`
**Default Sessions:** 10

## What it measures

A Tier-2 evaluation of a long-running developer agent on a research-notes
codebase. Each session, the agent receives a coding task, modifies files
in a persistent workspace, runs `pytest` against the workspace as
ground-truth verification, then answers held-out probes that force it to
rely on workspace files (not conversation memory) by calling
`adapter.reset_session()` between the task and the probes.

This was previously called S7+ in the codebase; renamed to S7 in v0.2 to
become the canonical Tier-2 anchor for research-notes-style codebases. For
the Tier-2 SWE-bench analogue (real GitHub issues + Django) see
[S8](../s8_swe_bench/).

## What makes this Tier-2 (not Tier-1)

- Real production CLI (OpenHands or Claude Code) wrapped via `AgentAdapter`
- Persistent on-disk workspace across sessions (files survive)
- `pytest` provides functional ground-truth — not LLM-judged
- The agent's only memory between sessions is what it wrote to files

## File layout

```
s7_research_notes/
├── README.md            # this file
├── design.md            # S7+ design rationale (ReadOnly metrics + protocol)
└── tests/               # held-out probe definitions
```

The runner lives at [`agingbench/runner/s7_runner.py`](../../runner/s7_runner.py).
The scenario itself is generated at run time via the `S7Generator`; no
seed manifest is checked in.

## Scoring

Per session:
- **`pytest_pass_rate`** — functional ground-truth signal
- **`probe_score`** — keyword match on held-out probes
- **`workspace_fidelity`** — agreement between agent's workspace files
  and gold reference
- **FactGraph-typed sub-metrics** — `version_accuracy`,
  `interference_resistance`, `accumulator_error` (per-mechanism breakdown)

## Example invocation

OpenHands SUT:

```bash
agingbench run \
  --scenario s7_research_notes \
  --sut agingbench/registry/suts/openhands/openhands_gpt4omini_s7.yaml \
  --generated --sessions 10 \
  --card
```

Claude Code SUT:

```bash
agingbench run \
  --scenario s7_research_notes \
  --sut agingbench/registry/suts/claude_code/claude_code_sonnet46_s7.yaml \
  --generated --sessions 10 \
  --card
```

## See also

- [S8 README](../s8_swe_bench/README.md) — Tier-2 SWE-bench analogue
  with load-bearing synthetic consistency tests
- [design.md](design.md) — ReadOnly metrics design notes
