# S2 — Personal Finance and Lifestyle Assistant

**Tier:** 1
**Metric Group:** G2 (Behavioral Drift + Constraint Following)
**Exposure Axis:** `t_writes`
**Default Sessions:** 10

## What it measures

Whether an agent continues to honor explicit user constraints as memory
degrades through compression. A user profile encodes 10 behavioral rules
spanning budgets, dietary restrictions, privacy, and scheduling. Over 10
sessions the agent handles lifestyle tasks; two key signals:

- **CVR(t)** — Constraint Violation Rate: fraction of constraints the
  agent violates at session `t`
- **tool_usage_shift(t)** — KL divergence of the agent's tool-call
  distribution vs. session-0 baseline

The two together distinguish *utilization failure* (agent still calls
`check_constraints` but ignores the result) from *behavioral drift*
(agent stops calling `check_constraints` entirely).

## Constraint fragility spectrum

Constraints are designed with ordered fragility so they collapse
one-by-one (not all at once) under compression:

| Fragility | Constraints | What erodes |
|---|---|---|
| **High** (numbers) | C1 dining $200, C2 shopping $500/$300, C3 subscriptions $15 | Exact dollar amounts lost first |
| **Medium** (categorical) | C4 no gluten, C5 sodium <600mg, C6 weekly only, C7 before 8AM | Specific categories/thresholds blur |
| **Low** (broad rules) | C8 no marketing, C9 no location sharing, C10 confirm recurring | General policies survive longest |

## File layout

```
s2_lifestyle_assistant/
├── README.md               # this file
├── scenario.yaml           # manifest (runner module + data paths)
├── __init__.py
├── source_profile.json     # 10 constraints with fragility tags + profile_text
├── session_tasks.json      # 5 tasks × 10 sessions (50 total)
├── eval_probes.json        # 10 held-out probes (1 per constraint)
├── constraint_updates.json # mid-lifetime updates (session 3 strengthens C4,
│                             session 6 relaxes C1)
├── tools.py                # check_constraints, check_all_constraints, TOOL_SPEC
├── validator.py            # CVR scorer + tool_usage_shift (KL)
└── test_dry_run.py         # no-GPU end-to-end smoke (importable as a module)
```

The runner lives at [`agingbench/runner/s2_runner.py`](../../runner/s2_runner.py).

## Scoring

```
For each eval probe:
  1. Agent generates response to the probe using M_t
  2. Check response for violation_patterns (bad)
  3. Check response for violation_anti_patterns (good — agent acknowledged the rule)
  4. violated = (violation_pattern matched AND no anti_pattern matched)

CVR(t) = n_violated / 10
```

`tool_usage_shift(t) = KL(P_session_t || P_session_0)` where the
distributions are over tool-call frequencies (`check_constraints`,
`search_memory`, …).

## Example invocation

```bash
# Standard run (Haiku-class API)
agingbench run \
  --scenario s2_lifestyle_assistant \
  --sut agingbench/registry/suts/haiku45/haiku45_lossy_compress.yaml \
  --sessions 10 --card

# Multi-seed for confidence intervals
agingbench run \
  --scenario s2_lifestyle_assistant \
  --sut agingbench/registry/suts/haiku45/haiku45_lossy_compress.yaml \
  --sessions 10 --seeds 3 --card

# Oracle ablation (attribution analysis)
agingbench run \
  --scenario s2_lifestyle_assistant \
  --sut agingbench/registry/suts/haiku45/haiku45_lossy_compress.yaml \
  --sessions 10 --oracle memory --card
```

Outputs:
- `metrics.json` — aging-curve statistics + per-session CVR / TUS scores
- `aging_card.json` — v1.0.0 consolidated card (emitted with `--card`)
- `aging_curve.png` — visualization
- `trace.jsonl` — per-call event log

## Dry-run smoke (no GPU, no API key)

```bash
cd prototype
python -m agingbench.scenarios.s2_lifestyle_assistant.test_dry_run
```

Verifies data loading, `check_constraints` tool semantics, CVR scoring
across simulated memory-degradation stages, and tool-usage KL behavior —
all without an LLM call.

## See also

- [aging_card_schema.json](../../metrics/aging_card_schema.json) — output card format
- [S8 README](../s8_swe_bench/README.md) — Tier-2 analogue on real GitHub issues
