# S1 — Research Literature Agent

**Tier:** 1
**Metric Group:** G1 (Task Performance / Compression)
**Exposure Axis:** `t_writes`
**Default Sessions:** 8

## What it measures

Fact survival under repeated memory compression. The agent receives a corpus
of ML research-paper facts (titles, authors, dates, contributions, numerical
results) at session 0, and over subsequent sessions answers held-out probes
about those facts using only its (compressed) memory state. The aging curve
shows how rapidly each fact category — names, dates, numbers, technical
claims — survives the agent's memory policy.

## File layout

```
s1_research_literature/
├── README.md           # this file
├── scenario.yaml       # manifest (runner module + data paths)
├── __init__.py
├── source_doc.json     # the seed corpus written to M_0
├── probes.json         # held-out evaluation probes (one per fact)
├── tasks.jsonl         # optional per-session tasks (otherwise generated)
├── validator.py        # probe scorer
└── task_validator.py   # per-task scorer
```

The runner lives at [`agingbench/runner/s1_runner.py`](../../runner/s1_runner.py).

For programmatic generation of sessions and probes (instead of using the
canned JSON), pass `--generated` and a `PressureConfig` preset.

## Scoring

Per session:
- **Probe accuracy** = fraction of held-out probes the agent answers correctly
  using only `M_t`.
- **By fact-category breakdown**: name / date / number / claim accuracy
  separately, since each category compresses at a different rate.

The aging curve `m(t)` is the per-session probe accuracy. Half-life is the
session at which `m(t)` first drops below `m_0 / 2`.

## Example invocation

```bash
agingbench run \
  --scenario s1_research_literature \
  --sut agingbench/registry/suts/haiku45/haiku45_lossy_growing.yaml \
  --sessions 8 \
  --card
```

Outputs `metrics.json`, `dependency_metrics.json`, `aging_card.json`, and
`aging_curve.png` in `experiments/results/<run-id>/`.

## See also

- [s8 SWE-bench-Aging README](../s8_swe_bench/README.md) — Tier-2 analogue
  on real GitHub issues
- [aging_card_schema.json](../../metrics/aging_card_schema.json) — output
  card format
