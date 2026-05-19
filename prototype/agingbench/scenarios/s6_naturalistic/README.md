# S6 — Naturalistic Multi-Domain Aging

**Tier:** 1
**Metric Group:** G3 (Memory Quality / Multi-Domain Recall)
**Exposure Axis:** `t_writes`
**Default Sessions:** 15 (scales to 30)

## What it measures

Multi-domain recall under naturalistic memory carryover. Tasks are derived
from WebArena and span travel, shopping, calendar, and email domains.
Memory carryover is *rational* — later sessions reference facts from
earlier ones, mimicking how a real personal assistant accumulates context
across days.

The aging signal is measured via a **recall matrix** `R[t][s]` =
agent's recall of session-`s` facts when probed at session `t`. The
diagonal `R[t][t]` measures fresh recall; off-diagonal `R[t][s]` for
`s < t` measures aged recall as a function of lag.

## File layout

```
s6_naturalistic/
├── README.md            # this file
├── __init__.py
├── session_tasks.json   # per-session primary tasks + recall probes
└── validator.py         # keyword-match scorer
```

The runner lives at [`agingbench/runner/s6_runner.py`](../../runner/s6_runner.py).

## Scoring

Per session `t`:
- **Primary-task accuracy** `task_m(t)` — keyword match against reference
- **Mean recall** `recall_m(t)` — averaged over all probes from sessions
  `0..t`
- **`fresh_recall(t)`** — recall of session-`t`'s own probes (just learned)
- **`aged_recall(t, lag)`** — recall of facts from `lag` sessions ago

## Example invocation

```bash
agingbench run \
  --scenario s6_naturalistic \
  --sut agingbench/registry/suts/haiku45/haiku45_lossy_growing.yaml \
  --sessions 15 \
  --card
```

The recall-matrix visualization is written to `recall_matrix.png` alongside
the standard aging-curve plot.
