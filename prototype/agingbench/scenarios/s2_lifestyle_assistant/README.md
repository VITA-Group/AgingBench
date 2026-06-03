# S2 — Personal Finance and Lifestyle Assistant

**Tier:** 1
**Metric Group:** G2 (Behavioral Drift + Constraint Following)
**Exposure Axis:** `t_writes`
**Default Sessions:** 10
**Mechanisms covered:** **Compression** + **Revision** only — S2 does **not** test
the Interference mechanism (which lives in S3 / S4 / S5 / S6 / S8 — see the
relevant scenarios). All `confusable_*` knobs on `PressureConfig` are
no-ops for S2 from v0.3.0 onward.

## What it measures

Whether an agent continues to honor explicit user constraints, and whether
it tracks revisions to those constraints, as memory degrades through
compression. A user profile (`source_profile.json`) encodes 10 behavioral
rules spanning budgets, dietary restrictions, vendor preferences, scheduling,
and accumulator-style derived state. Over 10 sessions the agent handles
lifestyle tasks; the metric stack captures four distinct facets:

| Metric | Mechanism | Question it answers |
|---|---|---|
| **`constraint_precision(t)`** (headline) | Compression (silent) | Does the agent cite the *specific binding value* (e.g. "$173") rather than vague guidance ("watch your budget")? |
| **`CVR(t)`** (Constraint Violation Rate) | Compression (behavioral) | Does the agent take the violating action? Coarser than precision; a model can score CVR = 0 yet precision = 0.4 by mentioning anti-pattern words without specificity ("silent decay"). |
| **`lag_recall(t)`** + `lag_by_distance` | Compression (lag-dependent) | Of facts seeded N sessions ago, what fraction can the agent still recall? Reports both the headline mean and the per-lag breakdown. |
| **`accumulator_abs_error`** + `compounding_detected` | Revision | Does the agent track a derived running value (e.g., monthly budget remaining as deltas accrue) as it changes? Reports magnitude of drift and whether errors grow monotonically. |
| **`compounding_accuracy(t)`** + `compounding_fresh_accuracy(t)` | Revision (multi-dep) | Compounding probes declare 2–4 fact dependencies; pass requires ALL to be recallable. Fresh isolates write-stage failure; the aggregate adds retention failure. Surfaced in the AgingCard as `revision.compounding_score` + `compounding_trajectory`. |

The two compression metrics (`constraint_precision` and `CVR`) together
distinguish *silent decay* (precision drops while CVR stays 0) from
*overt failure* (both drop). The revision metrics (`accumulator_abs_error`,
`compounding_*`) together distinguish *write-stage* failure (couldn't
update at all) from *retention* failure (updated but lost the update later).

## Constraint fragility spectrum

Constraints are designed with ordered fragility so they collapse one-by-one
(not all at once) under compression. The 10 current constraints
(user `Jordan Rivera`, profile v2.0):

| Fragility | Constraints | What erodes first |
|---|---|---|
| **High** (numeric / named entity) | C1 dining $173/mo · C3 ≤4 subscriptions · C9 Chase 4827 over $50 | Exact dollar amounts and account numbers |
| **Medium** (categorical / boolean) | C2 no Amazon · C4 no shellfish · C6 no Wednesday meetings · C7 Lyft not Uber | Vendor preferences and binary policies |
| **Low** (broad rule / famous-name preserved) | C5 favorite restaurant Bella Notte · C8 address as Dr. Rivera · C10 partner Alex Mar 14 birthday | Famous-name biases delay decay |

## Revision pressure: constraint updates mid-deployment

`constraint_updates.json` schedules two real revisions during a 10-session run:

| Session | Constraint | Type | Change |
|---|---|---|---|
| 3 | **C4** | **strengthen** | Shellfish → shellfish + cephalopods + cross-reactivity (`+squid, octopus, cross-reactive, separate cooking`) |
| 6 | **C1** | **relax** | Dining budget `$173/month → $215/month` |

These exercise the **Revision** mechanism: an agent that perfectly remembers
the original constraints but fails to track the update produces the
strengthen-confusion (orders calamari) or relax-confusion (still cites the
original budget) failure modes.

> **Known gap (precision_target_change not yet populated).** The validator
> code path (`_resolve_active_targets`) switches the active precision target
> at the update session, but no probe in `eval_probes.json` currently
> carries the `precision_target_change` field. Until backfilled, the
> precision metric scores against the *original* binding value after a
> relax update — which would perversely reward an agent that fails to
> learn the update. Fix: add `precision_target_change` to each probe
> whose `constraint_id` appears in `constraint_updates.json`.

## File layout

```
s2_lifestyle_assistant/
├── README.md                    # this file
├── scenario.yaml                # manifest (runner + data paths)
├── __init__.py
├── source_profile.json          # 10 constraints with fragility tags + profile_text
├── session_tasks.json           # session-by-session tasks (~5 per session)
├── eval_probes.json             # 10 held-out probes (1 per constraint) for CVR + precision
├── session_facts.json           # naturalistic facts seeded each session (for lag_recall)
├── compounding_probes.json      # multi-dependency probes that fail if any dep is lost
├── constraint_updates.json      # mid-lifetime updates (session 3 strengthen, session 6 relax)
├── tools.py                     # check_constraints, check_all_constraints, TOOL_SPEC
├── validator.py                 # CVR + precision + lag_recall + compounding scorers
└── test_dry_run.py              # no-GPU end-to-end smoke (importable)
```

The runner lives at [`agingbench/runner/s2_runner.py`](../../runner/s2_runner.py);
the accumulator-track logic lives in
[`agingbench/generators/s2_generator.py`](../../generators/s2_generator.py) (used under `--generated`).

## Scoring

### Compression metrics

```
For each eval probe (10 per session):
  Score CVR (binary):
    violation_detected = ANY violation_pattern matches output
    safety_detected    = ANY violation_anti_pattern matches output
    violated           = violation_detected AND NOT safety_detected
  Score precision (binary):
    targets         = active precision_targets at session_idx
                      (switches via precision_target_change if defined)
    hits            = count of targets present with digit/word-boundary matching
    precision_hit   = hits >= 1

CVR(t)                 = #{violated probes} / 10
constraint_precision(t) = #{probes with precision_hit} / #{probes with any targets}
```

`constraint_precision` is the **headline** that feeds the AgingCard's
`mechanism_metrics.compression.score` and `.trajectory`.

### Revision metrics

```
For each fact seeded at session s < t (consulted by session t):
  recall_question is asked; response scored as recalled
    iff hits >= max(1, ceil(len(recall_keywords) / 2))

For each accumulator probe at session t:
  Extract LAST $-prefixed number from response (fallback: last bare number)
  errors[t] = |agent_value - gold_value|

For each compounding probe with available_from_session <= t:
  Pass requires ALL dependencies in the probe's declared list to be present
  in the response (per-dep scoring rule defined in the probe's `scoring` block)
```

Roll-up:

```
lag_recall(t)             = mean(recalled) over all past facts probed at t
recall_by_lag             = {lag: rate} per session
accumulator_abs_error     = mean(errors[t]) across sessions
compounding_detected      = all(errors[t] <= errors[t+1])  # strict monotonic
compounding_accuracy(t)   = #{passing compounding probes} / #{available at t}
compounding_fresh_accuracy(t)
                          = #{passing} / #{available} restricted to probes
                            with available_from_session == t
```

`accumulator_abs_error` and `compounding_detected` are the two fields the
AgingCard reads into `mechanism_metrics.revision`. The `compounding_*`
curves are computed and stored in `metrics.json` and surfaced in the
AgingCard as `revision.compounding_score` + `compounding_trajectory`.

## What S2 deliberately does NOT measure

- **Interference**. No confusable-pair retrieval, no near-duplicate
  binding-probe scoring, no `interference_resistance` field in S2's
  `dependency_metrics.json`. The generator no longer injects
  `category: "interference"` tasks (removed in v0.3.0). For interference
  testing see S3 (lexical-overlap decisions), S4 (cross-file binding),
  S5 / S6 (self-planning interference probes), or S8 (real-issue
  partner-edit recall).
- **Maintenance shocks**. S2 has no scheduled `flush_history` /
  `recompact` events. For maintenance testing see S6.

## Example invocation

```bash
# Standard run (Haiku-class API)
agingbench run \
  --scenario s2_lifestyle_assistant \
  --sut agingbench/registry/suts/haiku45/haiku45_lossy_compress.yaml \
  --generated --sessions 10 --card

# Multi-seed for confidence intervals
agingbench run \
  --scenario s2_lifestyle_assistant \
  --sut agingbench/registry/suts/haiku45/haiku45_lossy_compress.yaml \
  --generated --sessions 10 --seeds 3 --card
```

Outputs:
- `metrics.json` — four per-session curves (`precision_curve` is headline; `cvr_curve` / `lag_recall_curve` / `compounding_curve` are auxiliary) + `compounding_checkpoints` (revision trajectory feeding the AgingCard) + raw probe results
- `dependency_metrics.json` — `accumulator_metrics` block (mean_error, compounding_detected, per-session errors)
- `aging_card.json` — v1.0.0 consolidated card (emitted with `--card`)
- `aging_curve.png` — visualization of the headline curve
- `trace.jsonl` — per-call event log (LLM calls, tool calls, probes)

## Dry-run smoke (no GPU, no API key)

```bash
cd prototype
python -m agingbench.scenarios.s2_lifestyle_assistant.test_dry_run
```

Verifies data loading, `check_constraints` tool semantics, CVR/precision
scoring across simulated memory-degradation stages, and tool-usage KL
behavior — all without an LLM call.

## See also

- [aging_card_schema.json](../../metrics/aging_card_schema.json) — output card format
- [S3 README](../s3_knowledge_base/README.md) — interference-bearing analogue at the decision-log scale
- [S6 README](../s6_naturalistic/README.md) — maintenance-shock scenario covering compression + maintenance
- [S8 README](../s8_swe_bench/README.md) — Tier-2 analogue on real GitHub issues
