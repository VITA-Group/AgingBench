# AgingBench Telemetry Mode

Maps production agent traces into AgingCards. The current release
verifies **one production format end-to-end — Claude Code session
files** — and accepts a `generic` JSONL shape (bring-your-own logs
with `session_id` / `role` / `content` / token fields). Adapters for
Langfuse, LangSmith, OpenAI Assistants, OpenHands, and OpenTelemetry
ship and parse-test successfully against shipped fixtures, but their
extraction recipes against current third-party SDKs are not yet
validated end-to-end and are tracked as **future extensions**.

Inference is **behavioral-DAG-based** — tool calls, tool results, and
outcomes form the structural signals; regex over user text is the
final fallback, not the default.

Two callable surfaces:

- **`trace_to_card(...)`** — backward-compatible stub. Cost block +
  `warnings: ["telemetry_partial"]`.
- **`trace_to_card_v11(...)`** — full pipeline: adapter → privacy-scrub
  → session-detect → memory-event reconstruction → outcome-extraction
  → 4-mechanism inference + cross-session consistency (P5) →
  dominant-mechanism arbitration → AgingCard with `trace_audit` block +
  Lifespan-Card surface (`signature`, `repair`, `trace_regime`).

Every signal carries a `derived_from` label from a controlled
vocabulary so consumers can distinguish structural signals from regex
fallbacks: `telemetry`, `tool_distribution_drift`,
`tool_result_update_propagation`, `tool_argument_self_reversion`,
`semantic_anchor_drift`, `cross_session_task_consistency`,
`user_correction_text_patterns_fallback`.

## When to use which mode

| Goal | Mode |
|---|---|
| "Is my deployed agent aging in production?" | telemetry |
| "Which model + memory-policy combo ages slowest under controlled inputs?" | scenarios |
| "Combine real-workload signal + controlled-stress signal" | telemetry + synthetic-probe-augmented |

## Quick start (Claude Code)

Claude Code writes one `.jsonl` per conversation under
`~/.claude/projects/<dir>/`. Concatenate them with `prepare_trace`,
then map to a card:

```python
from pathlib import Path
from agingbench.telemetry import prepare_trace, trace_to_card_v11

trace_path = prepare_trace(
    source=Path.home() / ".claude/projects/<your-project-dir>",
    output=Path("agingbench_trace.jsonl"),
)
# CLI alternative:
#   python -m agingbench.telemetry.prepare_trace ~/.claude/projects/<dir>

result = trace_to_card_v11(
    trace_jsonl=trace_path,
    trace_format="claude_code",
    profile="code_assistant",
    sut_hint={"sut_id": "my-claude-code", "model_id": "claude-sonnet-4-5"},
    extract_outcomes=["claude_session_flags", "record_patterns"],
)
print(result.card["headline"])
print(result.card["trace_audit"])
```

Pass a pre-aggregated `.jsonl` directly to skip Step 1. The pipeline
ignores non-LLM events (`queue-operation`, `file-history-snapshot`).

## Trace-completeness levels

The pipeline degrades gracefully — sparser traces still produce a
usable card, just with fewer fields populated.

| Level | Required trace fields | What you get |
|---|---|---|
| **L0** | timestamp + token counts | cost block |
| **L1** | + `session_id` + `tool_calls[]` | + interference drift (tool KL) |
| **L2** | + `model_id` + `prompt_preview` + `response_preview` | + compression pressure + maintenance shocks + revision proxy |
| **L3** | + outcome signal (`OutcomeEvent` JSONL or `extract_outcomes`) | + headline `m(t)` + outcome-conditional maintenance delta |

Claude Code ships at L2; the `record_patterns` and `claude_session_flags`
extractors push it to L3 without extra integration.

## Trace formats

```python
from agingbench.telemetry import list_supported_formats
list_supported_formats()
# ['claude_code', 'generic', 'langfuse',
#  'openai_assistants', 'openhands', 'otlp']
```

**Verified end-to-end (this release):**

| Format | Source shape |
|---|---|
| `claude_code` | `~/.claude/projects/<proj>/*.jsonl` written by Claude Code |
| `generic` | Any JSONL with `session_id` / `role` / `content` / token fields |

**Parse-tested adapters — extraction recipes pending validation
(future extension):**

| Format | Source shape |
|---|---|
| `openai_assistants` | `thread.message` / `thread.run` / `thread.run.step` objects |
| `openhands` | OpenHands SDK event log (`source`, `action`, `observation`, `llm_metrics`) |
| `langfuse` | Langfuse SDK exports or REST-API JSON (camelCase or snake_case) |
| `otlp` | OTLP JSON spans (`gen_ai.*` semconv + legacy `llm.*` namespace) |

LangSmith run JSON works today via `trace_format="generic"` — its field
shape is covered by the generic adapter's aliasing. A dedicated
`langsmith` format will be added once we ship a fixture + adapter-level
test for it.

Each adapter normalises into the canonical `TelemetryRecord`; all
downstream inference is format-agnostic, so the parse-tested adapters
work today if you already have a JSONL in the expected shape — what's
pending is end-to-end validation of the *extraction* path from each
third-party SDK. Sample fixtures in
[`example_traces/`](example_traces/); contributions of validated
recipes are welcome (see Roadmap).

## Outcome extractors

Derive outcomes from in-trace signals — no separate JSONL required.

| Extractor | Watches | Emits |
|---|---|---|
| `claude_session_flags` | user messages with `/clear`, `/reset`, `/new`, `/end` | `abandoned` |
| `record_patterns` | user-message tone after agent response | `fail` on negation, `success` on positive |
| `git_log:<repo>` | `git log --grep="^Revert"` over past 90 days | `revision_fail` linked to the agent record likely behind the reverted commit |

Spec syntax for `extract_outcomes`:

```python
extract_outcomes=[
    "claude_session_flags",                 # bare name
    "git_log:./my-project",                 # name:positional-arg
    "git_log:./my-project:since_days=30",   # name:arg:k=v
]
```

All extractors are best-effort: missing sources emit a warning and an
empty list, not a crash.

## Deployment profiles

A profile encodes domain conventions (outcome-extraction rules,
default privacy patterns, session-detection defaults):

```python
from agingbench.telemetry import list_profiles, load_profile
list_profiles()                     # ['code_assistant', 'generic']
p = load_profile("code_assistant")
p.outcome_rules                     # {'pr_merged': 'success', ...}
p.privacy_patterns                  # [{'pattern': 'AKIA...', 'replacement': '[AWS_ACCESS_KEY]'}, ...]
```

Override per call:

```python
trace_to_card_v11(
    ...,
    profile="code_assistant",
    overrides={"outcome_rules": {"ci_skipped": "abandoned"}},
)
```

The effective rules hash is emitted on the card (`outcome_rules_hash`)
so two teams using the same rules can compare cards meaningfully.

## Synthetic-probe augmentation

For mechanisms the production workload doesn't stress, run an
AgingBench scenario as a probe against the deployed agent, then merge:

```bash
# Step 1. Run the scenario via the standard CLI:
agingbench run --scenario s1_research_literature \
               --sut <your-sut-yaml> --sessions 8 --card --output ./probes/
```

```python
# Step 2. Mix the probe result into your telemetry card:
result = trace_to_card_v11(
    trace_jsonl=Path("prod_trace.jsonl"),
    ...,
    synthetic_probe_cards=[Path("./probes/aging_card.json")],
)
# result.card["synthetic_probes"]["s1_research_literature"] carries
# the controlled-scenario headline alongside the trace-derived metrics.
```

Only S1–S4 / S6 are injectable (`list_injectable_scenarios()`); S5 /
S7 / S8 require docker / production-CLI conditions that can't be
guaranteed in arbitrary deployments.

> **Probe runs are live** — the scenario actually executes against
> your agent. Not post-hoc trace manipulation. For pure post-hoc
> analysis on archived traces, use the extractors above.

## Mechanism trajectories and verdicts

Every mechanism block ships a per-session trajectory + slope + a
**saturation-aware verdict** (so a signal that collapsed to zero by
session 3 isn't labelled "rising healthy" on residual OLS noise).

| Trajectory | Mechanism | What it captures |
|---|---|---|
| `context_noise_ratio_trajectory` | compression | `input_tokens` / distinct emitted entities. Rising = signal density falling |
| `tool_argument_specificity_trajectory` | compression | Fraction of tool-call args that look specific (UUIDs, ISO timestamps, file paths). Falling = compression eating specificity |
| `goal_anchor_drift_trajectory` | interference | Embedding cosine vs session-0 user prompt (Jaccard fallback). Falling = semantic drift |
| `lineage_continuity_trajectory` | interference | Fraction of prior-session entities still referenced. Falling = interference-style forgetting |
| `value_supersession_trajectory` ≡ `per_session_violation_trajectory` | revision | Agent cites a value the world has superseded. Rising = belief-update failures |
| `intervention_rate_trajectory` | maintenance | Human-steering events / agent actions. Rising = more handholding needed |
| `consistency_drop_trajectory` | consistency (P5) | Cumulative `behavior_drift_at_repeat` across repeat-task clusters |

Each has a matching `<name>_slope` and `<name>_verdict`. Verdict enum:

| Verdict | Meaning |
|---|---|
| `no_signal` | trajectory too short or mostly missing |
| `flat` | slope below epsilon, not at a boundary |
| `rising_degradation` / `rising_healthy` | slope > eps; sign by metric polarity |
| `falling_degradation` / `falling_healthy` | slope < −eps; sign by metric polarity |
| `floor_degradation` / `floor_healthy` | saturated low (late-window mean ≤ floor) |
| `ceiling_degradation` / `ceiling_healthy` | saturated high (late-window mean ≥ ceiling) |

`agingbench.telemetry.inference._verdict.is_degrading(verdict)`
collapses to bool for dashboards / alerting.

## Revision: three-tier ladder

Revision dispatches across three tiers based on what the trace
carries; emits the same trajectory under both canonical
(`value_supersession_*`) and legacy (`per_session_violation_*`) names
for backward-compat.

1. **`tool_result_update_propagation`** (structural, preferred): builds
   an `(entity, attribute) → [(t, value)]` timeline from
   `ToolCall.result_summary`; counts agent args citing superseded values.
2. **`tool_argument_self_reversion`** (structural, universal): tracks
   `(arg_key, arg_value)` across sessions; counts agent reverting to
   stale values. Fires on any adapter that populates `args`.
3. **`user_correction_text_patterns_fallback`** (regex, English-only):
   `correction_repetition_rate`, `stale_value_citation_rate`, etc.

## Headline policy (4-tier)

The headline is selected at card-assembly time based on what evidence
the trace carries. Inspect `result.card["headline"]["source"]` to know
which tier fired.

| Tier | Trigger | Headline | `source` |
|---|---|---|---|
| 1 | `OutcomeEvent`s present | `Half-life: N sessions` | `"outcomes"` |
| 2 | No outcomes, ≥ 1 repeat-task cluster (P5) | `Behavior drift: N% on repeat tasks` | `"behavior_drift_at_repeat"` |
| 3 | No outcomes, no clusters, mechanism severity sum rises over ≥ 3 sessions | `Aging trend: rising (slope N/session)` | `"aging_trend"` |
| 4 | None of the above | `Aging not measurable` | `"not_measurable"` |

This unlocks meaningful headlines on outcome-free traces (the common
production case).

## Output schema

`trace_to_card_v11` returns `TraceToCardV11Result(card, n_records,
n_sessions, n_outcome_events, session_detection_mode, profile_used,
outcome_rules_hash)`.

The card always populates `cost_and_efficiency` (token usage) and
`trace_audit`; `headline` per the 4-tier policy above.

**`trace_audit` top-level keys:**

| Key | What it is |
|---|---|
| `derived_from` | Always `"telemetry"` |
| `deployment_type`, `n_sessions_detected`, `n_outcome_events`, `session_detection_mode`, `outcome_rules_hash` | Mirror the `TraceToCardV11Result` fields |
| `trace_regime` | Chat-only vs tool-using, n_sessions, adapter, outcomes-linked. Used by the card surface to caveat unanswerable claims |
| `compression`, `interference`, `revision`, `maintenance`, `consistency` | Per-mechanism blocks (one per mechanism + P5) |
| `dominant_mechanism` | `{dominant, reason, scores, evidence, compatible}`. `reason ∈ {argmax, no_independent_evidence, no_signal}` |
| `signature` | `W` / `R` / `U` / `S` from the dominant mechanism (`None` if no mechanism passes the gate) |
| `repair` | Recommended repair label paired to the signature |
| `headline` | Mirrors `card["headline"]` for self-containment |

Each per-mechanism block carries `coverage`, `derived_from`, and its
own metric fields. `coverage` is `{n_observations, coverage_fraction,
verdict}` with `verdict ∈ {strong, adequate, weak, underpowered,
no_test_fired}`. Full per-block field lists in the inference module
docstrings (`agingbench/telemetry/inference/<mechanism>.py`).

The card validates against the v1.0.0 AgingCard schema.

## File layout

```
telemetry/
├── trace_to_card.py             # pipeline + 4-tier headline policy
├── schema.py                    # canonical dataclasses
├── prepare_trace.py             # Claude Code .jsonl concatenation (Python + CLI)
├── card_lookups.py              # MECHANISM_TO_STAGE (W/R/U/S) + MECHANISM_TO_REPAIR
├── card_render.py               # ASCII Lifespan-Card renderer
├── session_detection.py         # explicit_id → user_id_split → idle_gap
├── memory_reconstruction.py     # 5-rule shock detection
├── privacy_scrubber.py          # PII redaction + session-id hashing
├── outcome_extractors.py        # 3 built-in extractors
├── synthetic_probe.py           # scenarios-as-probes orchestration
├── adapters/                    # 7 format adapters (claude_code, generic, ...)
├── inference/
│   ├── compression.py           # saturation + tool-arg specificity
│   ├── interference.py          # tool-KL + embedding anchor drift + lineage continuity
│   ├── revision.py              # 3-tier ladder
│   ├── maintenance.py           # pre/post-shock delta + intervention rate
│   ├── consistency.py           # P5: cross-session task consistency
│   ├── _selector.py             # dominant-mechanism arbitration
│   ├── _verdict.py              # saturation-aware verdict mapper
│   └── _text_utils.py           # entity / clustering helpers
├── profiles/                    # generic.yaml, code_assistant.yaml
└── example_traces/              # shipped fixtures
```

## Public API

```python
from agingbench.telemetry import (
    trace_to_card_v11, TraceToCardV11Result,
    prepare_trace,
    trace_to_card, TraceToCardResult, SUPPORTED_TRACE_FORMATS,   # stub
    TelemetryRecord, OutcomeEvent, MemoryEvent, ToolCall,
    CoverageReport, TraceAuditBlock,
    list_supported_formats, list_profiles, list_extractors,
    list_injectable_scenarios,
    load_profile, Profile,
    run_extractor,
    extract_from_claude_session_flags,
    extract_from_record_patterns,
    extract_from_git_log,
    load_probe_result, merge_probe_into_card,
    ProbeSchedule, ProbeResult,
)
```

The four mechanism inference functions are also public when you want
to skip the full pipeline:

```python
from agingbench.telemetry.inference import (
    infer_compression, infer_interference, infer_revision,
    infer_maintenance, infer_consistency,
)
```

Each takes `list[list[TelemetryRecord]]` (sessions × records) and
returns the dict that lands under `trace_audit[<mechanism>]`.

## Status

Pipeline components (schemas, session detection, memory-event
reconstruction, privacy scrubber, 4-mechanism inference + P5
consistency, dominant-mechanism selector, headline policy, outcome
extractors, deployment profiles, synthetic-probe orchestrator, ASCII
card renderer, `prepare_trace` preprocessor) are shipped and covered
by the test suite (~85 telemetry-specific tests across
`test_telemetry_adapters.py`, `test_telemetry_stub.py`,
`test_telemetry_v11.py`; 228+ tests total in `prototype/tests/`).

**Trace-format coverage in this release**: Claude Code is verified
end-to-end on real production traces; the `generic` adapter is
verified against fixture data. The remaining four adapters
(`openai_assistants`, `openhands`, `langfuse`, `otlp`) pass
adapter-level tests against shipped fixtures, but their *extraction
recipes* — the steps needed to dump a JSONL of the right shape from
each live third-party SDK — have not been validated against current
SDK versions and are tracked as future work. LangSmith run JSON is
routable today via `trace_format="generic"`.

## Roadmap

| Milestone | Scope |
|---|---|
| **Next** | End-to-end validation of the four parse-tested adapters against current SDKs (`openai_assistants`, `openhands`, `langfuse`, `otlp`), promoting each to "verified" as it lands. Add a dedicated `langsmith` format (currently routes through `generic`). More outcome extractors (GitHub Actions CI status, Langfuse score events, Slack reactions). Validation correlation study against scenario-derived metrics. |
| **Later** | Cross-tenant aggregation with differential privacy. Streaming ingestion. Native protobuf OTLP. |
| **v2** | Multilingual user-correction detection. Workspace-fidelity inference for self-planning agents (S5). |

Contributing a validated recipe for one of the parse-tested adapters
is the highest-leverage way to widen format coverage. The top-level
`docs/CONTRIBUTING.md` covers SUT YAMLs and integration adapters;
telemetry-adapter recipes can be contributed by adding a fixture under
`example_traces/`, a `normalize()` implementation under `adapters/`,
and registering the format in `adapters/__init__.py`.
