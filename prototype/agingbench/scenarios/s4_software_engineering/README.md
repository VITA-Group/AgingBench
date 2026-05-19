# S4 — Software Engineering Planning

**Tier:** 1
**Metric Group:** G4 (Operational Efficiency / Code Planning)
**Exposure Axis:** `t_steps`
**Default Sessions:** 8

## What it measures

Code-planning quality as a codebase evolves. The agent is shown progressive
snapshots of a codebase across sessions and asked to plan changes (no
execution). Aging signals appear when the agent's plans reference symbols
from outdated snapshots (revision aging), miss dependencies introduced in
later snapshots (compression / interference), or repeat planning errors
that earlier sessions had corrected (memory-policy failure).

This is a *planning-only* scenario — no Docker, no compilation, no test
runs. For end-to-end software-engineering with real verification see
[S7](../s7_research_notes/) (research-notes CLI) or
[S8](../s8_swe_bench/) (SWE-bench-Verified Django chain).

## File layout

```
s4_software_engineering/
├── README.md           # this file
├── __init__.py
├── tasks.json          # per-session planning prompts
├── snapshots/          # per-session codebase snapshots (text)
├── runner.py / validator.py
└── scenario.yaml       # manifest
```

## Scoring

Per session:
- **Plan-validity score** = fraction of tasks where the agent's plan
  references only symbols present in the latest snapshot.
- **Revision-correctness**: when a function is renamed across snapshots,
  whether the agent's plan uses the current name.

## Example invocation

```bash
agingbench run \
  --scenario s4_software_engineering \
  --sut agingbench/registry/suts/haiku45/haiku45_lossy_growing.yaml \
  --sessions 8 \
  --card
```
