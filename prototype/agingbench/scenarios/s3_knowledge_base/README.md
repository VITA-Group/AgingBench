# S3 — Knowledge Base (Project Decisions)

**Tier:** 1
**Metric Group:** G3 (Memory Quality / Decision Fidelity)
**Exposure Axis:** `t_writes`
**Default Sessions:** 12 (scales to 100 for high-pressure variants)

## What it measures

Decision fidelity as a project's decision history accumulates. The agent
ingests meeting transcripts session-by-session and is asked to recall the
current state of past decisions. As the knowledge base grows, the agent
must retrieve the *latest* decision for each topic — not an outdated
superseded version — across an ever-larger set of past sessions.

Aging signals: revision accuracy (latest-wins), interference resistance
(distinguishing similar decisions on different topics), and recall depth
(answering questions about decisions from N sessions ago).

## File layout

```
s3_knowledge_base/
├── README.md           # this file
├── __init__.py
├── transcripts.json    # per-session meeting transcripts
├── gold_timeline.json  # ground-truth decision timeline (which decision is current at each t)
├── queries.json        # held-out queries probing decision recall
├── runner.py / validator.py
└── scenario.yaml       # manifest
```

## Scoring

Per session:
- **Decision-fidelity score** = fraction of queries answered with the
  decision that was current as of that session per `gold_timeline.json`.
- **Latency-to-correction**: when a decision is updated, how many sessions
  pass before the agent stops citing the old version.

## Example invocation

```bash
agingbench run \
  --scenario s3_knowledge_base \
  --sut agingbench/registry/suts/haiku45/haiku45_lossy_growing.yaml \
  --generated --sessions 12 \
  --card
```

For longer-horizon stress tests:

```bash
agingbench run --scenario s3_knowledge_base --generated --sessions 100
```

Aging *pressure* (interference/revision/dependency density) is not a CLI flag — it is set via a `pressure:` key in the SUT YAML (a preset name `none`/`light`/`medium`/`heavy`, or explicit `PressureConfig` overrides), resolved per `cli/loaders.py::_resolve_pressure`. See `agingbench/registry/suites/pressure_sweep.yaml` for examples.
