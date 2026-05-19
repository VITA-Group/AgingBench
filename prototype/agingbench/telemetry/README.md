# AgingBench Telemetry Mode

Maps production agent traces (Langfuse, LangSmith, OpenTelemetry,
Claude Code session files, custom JSONL) into AgingCards. Two callable
surfaces:

- **`trace_to_card(...)`** — v1.0 stub. Backward-compatible. Returns a
  partial AgingCard with the cost block + `warnings: ["telemetry_partial"]`.
- **`trace_to_card_v11(...)`** — v1.1 full pipeline. Adapter →
  privacy-scrub → session-detect → memory-event reconstruction →
  outcome-extraction → four-mechanism inference → AgingCard with a
  `trace_audit` block parallel to `mechanism_metrics`.

Telemetry-derived metrics are flagged `derived_from: "telemetry"` so
consumers never confuse them with scenario-derived measurements.

## When to use which mode

| Goal | Mode |
|---|---|
| "Is my deployed agent aging in production?" | telemetry mode |
| "Which model + memory-policy combo ages slowest under controlled inputs?" | scenarios mode |
| "Get one card combining real-workload signal + controlled-stress signal" | telemetry + synthetic-probe-augmented |

## Quick start (Claude Code, plug-and-play)

Claude Code writes structured session JSONL to `~/.claude/projects/` by
default. **No instrumentation needed.** This single call gives you a
full AgingCard with cost block, mechanism inference, and outcome-derived
headline curve:

```python
from pathlib import Path
from agingbench.telemetry import trace_to_card_v11

result = trace_to_card_v11(
    trace_jsonl=Path.home() / ".claude/projects/<hash>/<sess>.jsonl",
    trace_format="claude_code",
    profile="code_assistant",
    sut_hint={"sut_id": "my-claude-code", "model_id": "claude-sonnet-4-5"},
    extract_outcomes=[
        "claude_session_flags",     # /clear, /reset, /new, /end → outcomes
        "record_patterns",          # "no, that's wrong" / "thanks" → outcomes
        # "git_log:./my-repo:since_days=30",  # optional: git revert detection
    ],
)
print(result.card["trace_audit"])
print(result.card["headline"])
print(result.card["cost_and_efficiency"])
```

That's the whole onboarding. The pipeline tolerates the various event
types Claude Code emits (`user`, `assistant`, `tool_use`, `tool_result`,
`system`) and skips non-LLM events (`queue-operation`,
`file-history-snapshot`, etc.) without crashing.

## Trace-completeness levels (what trace fields produce what output)

Distinct from the SUT tier classification (Tier 1 / Tier 2 in the top-level
README): this is about how complete a *trace* is. The pipeline degrades
gracefully — agents that emit less still produce a usable card, just with
fewer fields populated.

| Level | Required trace fields | What you get |
|---|---|---|
| **L0** | timestamp + token counts | cost block only |
| **L1** | + `session_id` + `tool_calls[]` | + interference drift (tool KL) |
| **L2** | + `model_id` + `prompt_preview` + `response_preview` (or rich `prompt_tokens`) | + compression pressure + maintenance shock detection + revision proxy |
| **L3** | + outcome signal: either an `OutcomeEvent` JSONL (`outcomes_jsonl=`) OR run any `extract_outcomes` extractor against the records | + headline aging curve `m(t)` + outcome-conditional maintenance delta |

Claude Code traces ship with L2 fields out of the box; the
`record_patterns` and `claude_session_flags` extractors push them to
L3 without any extra integration.

## Supported trace formats

```python
from agingbench.telemetry import list_supported_formats
list_supported_formats()
# ['claude_code', 'generic', 'langfuse', 'langsmith',
#  'openai_assistants', 'openhands', 'otlp']
```

| Format | When to use |
|---|---|
| `claude_code` | `~/.claude/projects/<proj>/*.jsonl` written by Claude Code |
| `openai_assistants` | OpenAI Assistants API: `thread.message`, `thread.run`, `thread.run.step` objects |
| `openhands` | OpenHands SDK event log (`source`, `action`, `observation`, `llm_metrics`) |
| `langfuse` | Langfuse SDK exports or REST-API JSON downloads (camelCase or snake_case) |
| `langsmith` | LangSmith run JSON exports — routed through the generic adapter's field-aliasing |
| `otlp` | OpenTelemetry JSON spans (recognises both `gen_ai.*` semconv and the legacy `llm.*` namespace) |
| `generic` | Best-effort fallback for any JSONL with `session_id`/`role`/`content` fields |

Each is a thin normaliser into the canonical `TelemetryRecord` shape;
all downstream inference is format-agnostic. Sample fixtures for every
format live in [`example_traces/`](example_traces/).

## Outcome extractors (built-in)

The pipeline can derive outcomes from three sources without you needing
to ship a separate JSONL:

| Extractor | What it watches | OutcomeEvent emitted |
|---|---|---|
| `claude_session_flags` | User messages containing `/clear`, `/reset`, `/new`, `/end` | `abandoned` (the user gave up on the prior task) |
| `record_patterns` | User-message tone immediately after an agent response | `fail` for negation patterns ("no, that's wrong", "try again", "undo"); `success` for positive patterns ("thanks", "perfect", "looks good") |
| `git_log:<repo>` | `git log --grep="^Revert"` over the past 90 days | `revision_fail` linked to the agent record that likely produced the reverted commit |

```python
from agingbench.telemetry import list_extractors
list_extractors()
# ['claude_session_flags', 'git_log', 'record_patterns']
```

Spec syntax (used by `extract_outcomes` parameter):

```python
extract_outcomes=[
    "claude_session_flags",                 # bare name
    "record_patterns",                      # bare name
    "git_log:./my-project",                 # name:positional-arg (repo path)
    "git_log:./my-project:since_days=30",   # name:arg:k=v
]
```

All extractors are best-effort: if a source is unavailable (no git
repo, missing fields, etc.) they emit a warning and an empty list
rather than crashing the pipeline.

## Deployment profiles

A profile encodes domain conventions: outcome-extraction rules,
subject-linkage rules, mechanism weights, default privacy patterns,
and session-detection defaults.

```python
from agingbench.telemetry import list_profiles, load_profile
list_profiles()                     # ['code_assistant', 'generic']
p = load_profile("code_assistant")
p.outcome_rules                     # {'pr_merged': 'success', 'ci_pass': 'success', ...}
p.mechanism_weights                 # {'compression': 0.8, 'revision': 1.5, ...}
```

Override per-call when your team's convention differs from the default:

```python
trace_to_card_v11(
    trace_jsonl=...,
    profile="code_assistant",
    overrides={
        "outcome_rules": {
            "ci_skipped": "abandoned",     # add a rule
            "pr_merged":  "user_rejected", # override a default (unusual but supported)
        },
    },
)
```

The effective `outcome_rules_hash` is emitted on the AgingCard so two
teams using the same rules can compare cards meaningfully.

## Synthetic-probe augmentation

For mechanisms the user's natural workload doesn't stress, run one of
AgingBench's existing scenarios as a probe against the deployed agent
(separately, via `agingbench run`), then merge the resulting card into
your telemetry analysis:

```python
from agingbench.telemetry import (
    list_injectable_scenarios,
    trace_to_card_v11,
)

list_injectable_scenarios()
# ['s1_research_literature', 's2_lifestyle_assistant',
#  's3_knowledge_base', 's4_software_engineering', 's6_naturalistic']
```

```bash
# Step 1. Run the scenario via the standard CLI against your agent:
agingbench run --scenario s1_research_literature \
               --sut <your-sut-yaml> --sessions 8 --card \
               --output ./probes/
```

```python
# Step 2. Mix the probe results into your telemetry card:
result = trace_to_card_v11(
    trace_jsonl=Path("prod_trace.jsonl"),
    trace_format="claude_code",
    profile="code_assistant",
    extract_outcomes=["claude_session_flags", "record_patterns"],
    synthetic_probe_cards=[Path("./probes/aging_card.json")],
)
# result.card["synthetic_probes"]["s1_research_literature"] now carries
# the controlled-scenario headline alongside trace_audit's observational
# metrics from the production trace.
```

S5 / S7 / S8 (Tier-2/3 scenarios) are **intentionally NOT** in the
injectable set: their docker-container / production-CLI requirements
can't be guaranteed in arbitrary deployment contexts. Only S1 / S2 / S3 /
S4 / S6 (memory-policy-controlled scenarios) ship as probe sources.

> **Important:** synthetic-probe augmentation requires a **live probe
> run** against your deployed agent — it is NOT post-hoc trace
> manipulation. The probe results are merged at card-assembly time;
> no new data is injected into the original trace. For pure post-hoc
> analysis on archived traces, use the extractors above instead.

## What telemetry mode CAN and CANNOT measure

| Mechanism | Can measure (telemetry) | Cannot measure (needs scenario) |
|---|---|---|
| Compression | saturation pressure, self-contradiction rate, fact-density slope, **context-noise ratio trajectory** | absolute fact survival (no gold answers) |
| Interference | tool-distribution KL drift over sessions, **goal-anchor drift trajectory** | resistance to constructed confusable pairs |
| Revision | user-correction-repetition rate, stale-value-citation rate, **per-session violation trajectory** | latest-wins accuracy without state-change ground truth |
| Maintenance | shock event detection (model swap, context reset, compression spike, system change, /clear), structural pre/post delta, outcome-conditional delta, **intervention-rate trajectory** | counterfactual ("what would the score have been without the shock?") |
| Headline `m(t)` | computable when extractors fire OR an OutcomeEvent JSONL is supplied | only as accurate as the outcome-extraction quality |

Each mechanism block carries a `coverage` sub-block with verdict
(`strong | adequate | weak | underpowered | no_test_fired`) so consumers
know whether a low score = "agent didn't age" or = "test underpowered."

### Long-horizon degradation trajectories (one per mechanism)

In addition to the per-mechanism aggregate scores, every mechanism block
exposes a **trajectory** that captures how the signal evolves over many
sessions — the form long-horizon degradation actually takes:

| Trajectory | Mechanism | What it captures |
|---|---|---|
| `context_noise_ratio_trajectory` | compression | per-session ratio of `input_tokens carried in` / `distinct entities the agent emits`. Rising = effective signal density falling as accumulated context grows |
| `goal_anchor_drift_trajectory` | interference | per-session Jaccard overlap of agent vocabulary vs the session-0 user-prompt vocabulary. Declining = the agent has drifted from the original task framing (general; no domain knowledge required) |
| `per_session_violation_trajectory` | revision | per-session count of agent responses that contain a previously-corrected OLD value (without the NEW value). Rising = constraint forgetting / belief-update failures accumulating |
| `intervention_rate_trajectory` | maintenance | per-session ratio of `human-steering events (fail/user_rejected/abandoned outcomes)` / `agent actions`. Rising = the agent needs more handholding to stay on track |

Each trajectory ships alongside its slope (`<name>_slope`) **and** a
saturation-aware verdict (`<name>_verdict`), so consumers can ask
"is this metric trending up or down across sessions?" without
re-running OLS themselves. All four are general (no domain assumptions,
no entity-specific knowledge); they fall directly under the existing
four mechanisms rather than introducing a fifth.

#### Saturation-aware verdict labels

Naive `slope > 0 ⇒ rising ⇒ degrading` mis-classifies trajectories that
have already saturated at a boundary (e.g., a goal-anchor signal that
collapsed to ≈0 by session 3 still has tiny positive OLS noise on the
zero floor — it is *not* "rising and healthy"). Each mechanism therefore
emits one of nine verdict strings derived from the late-window mean
**and** the slope:

| Verdict | Meaning |
|---|---|
| `no_signal` | trajectory too short / mostly missing |
| `flat` | slope below epsilon and not at a boundary |
| `rising_degradation` / `rising_healthy` | slope > eps; sign of "rising = bad" set by metric |
| `falling_degradation` / `falling_healthy` | slope < −eps; sign set by metric |
| `floor_degradation` / `floor_healthy` | late-window mean ≤ floor threshold (saturated low) |
| `ceiling_degradation` / `ceiling_healthy` | late-window mean ≥ ceiling threshold (saturated high) |

Per-mechanism verdict fields:

| Field | Mechanism |
|---|---|
| `context_noise_verdict` | compression |
| `goal_anchor_drift_verdict` | interference |
| `violation_trajectory_verdict` | revision |
| `intervention_rate_verdict` | maintenance |

Use `agingbench.telemetry.inference._verdict.is_degrading(verdict)` to
collapse a verdict into a boolean for dashboards / alerting.

## What the user defines vs. what AgingBench defines

| Layer | AgingBench provides | User provides |
|---|---|---|
| Framework (TelemetryRecord, mechanism inference math, AgingCard schema) | ✓ | — |
| Default outcome-extraction rules per deployment profile | ✓ shipped YAMLs | — |
| Built-in outcome extractors | ✓ (claude_session_flags, record_patterns, git_log) | — |
| Probe content (S1–S6 scenarios already ship) | ✓ | — |
| Profile selection | — | one keyword |
| Outcome-rule overrides for non-standard deployments | — | optional dict |
| Trace data | — | from your platform |
| Probe scheduling (when/how often to inject) | — | cron / CI |

For most production users, **picking a profile + enabling one or two
extractors** is sufficient. Custom probe content is only needed for
genuinely unique domains not covered by S1–S6.

## Output schema

### `TraceToCardV11Result` (returned by `trace_to_card_v11`)

| Attribute | Type | What it is |
|---|---|---|
| `card` | `dict` | Full AgingCard with `trace_audit` block (validates against v1.0.0 schema) |
| `n_records` | `int` | Records ingested after the per-format adapter (= rows that produced a `TelemetryRecord`) |
| `n_sessions` | `int` | Distinct sessions detected by `session_detection_mode` |
| `n_outcome_events` | `int` | Outcome events linked into the pipeline (from `outcomes_jsonl` + `extract_outcomes`) |
| `session_detection_mode` | `str` | One of `explicit_id`, `user_id_split`, `idle_gap` |
| `profile_used` | `str` | Deployment profile applied (`generic`, `code_assistant`) |
| `outcome_rules_hash` | `str` | `sha256:<hex>` fingerprint of the outcome-extraction rules — for run reproducibility |

The v1.0 stub returns a smaller `TraceToCardResult` with: `card`, `derived_fields`,
`missing_fields`, `n_calls`.

### Top-level AgingCard fields populated by telemetry mode

```python
result.card["cost_and_efficiency"]   # always populated when records carry token usage
result.card["trace_audit"]           # always populated; nested below
result.card["headline"]              # populated when outcomes events are present
```

`cost_and_efficiency` keys: `total_input_tokens`, `total_output_tokens`,
`total_calls`, `tokens_per_session_mean`, `latency_ms_p50`, `latency_ms_p95`,
`total_cost_usd`.

`headline` keys (only when outcomes are linked): `checkpoints`, `m0`, `m_final`,
`decay_slope`, `half_life`.

### `trace_audit` block — top-level keys

| Key | Type | What it is |
|---|---|---|
| `derived_from` | `str` | Always `"telemetry"` (so consumers can distinguish from scenario-derived blocks) |
| `deployment_type` | `str` | Mirrors `profile_used` |
| `n_sessions_detected` | `int` | Same as `result.n_sessions` |
| `n_outcome_events` | `int` | Same as `result.n_outcome_events` |
| `session_detection_mode` | `str` | Same as `result.session_detection_mode` |
| `outcome_rules_hash` | `str` | Same as `result.outcome_rules_hash` |
| `compression` | `dict` | per-mechanism block, see below |
| `interference` | `dict` | per-mechanism block, see below |
| `revision` | `dict` | per-mechanism block, see below |
| `maintenance` | `dict` | per-mechanism block, see below |
| `headline` | `dict` | mirrors `result.card["headline"]` for self-containment |

### Per-mechanism block fields

Every mechanism block carries `coverage` and `derived_from` plus its own metric
fields. `coverage` is `{n_observations, coverage_fraction, verdict}` where
`verdict ∈ {strong, adequate, weak, underpowered, no_test_fired}`.

**`compression`** (`derived_from: "telemetry"`):
`saturation_session_rate`, `saturation_slope`, `saturation_trajectory`,
`self_contradiction_rate`, `fact_density_slope`,
`context_noise_ratio_trajectory`, `context_noise_slope`, `context_noise_verdict`.

**`interference`** (`derived_from: "tool_distribution_drift"`):
`tool_kl_trajectory`, `tool_kl_mean_post_baseline`, `tool_kl_slope`,
`baseline_window_size`, `n_distinct_tools`, `goal_anchor_drift_trajectory`,
`goal_anchor_drift_slope`, `goal_anchor_drift_verdict`.
The `goal_anchor_drift_*` keys are conditional on having ≥ `baseline_window_n + 1`
sessions; when absent the block omits them.

**`revision`** (`derived_from: "user_correction_text_patterns"`):
`correction_repetition_rate`, `stale_value_citation_rate`,
`median_time_to_adopt_correction_turns`, `n_corrections_detected`,
`n_repeated_corrections`, `per_session_violation_trajectory`,
`violation_trajectory_slope`, `violation_trajectory_verdict`.

**`maintenance`** (`derived_from: "telemetry"`):
`shock_events`, `per_shock_deltas`, `median_outcome_rate_delta`,
`median_latency_p50_delta_ms`, `n_shocks`, `intervention_rate_trajectory`,
`intervention_rate_slope`, `intervention_rate_verdict`.

The card validates against the v1.0.0 AgingCard schema and is interoperable
with the rest of AgingBench (validators, adapters, leaderboard).

## Advanced API: running an inference module directly

The four mechanism inference modules are public and stable; call them on
`list[list[TelemetryRecord]]` (a list of sessions, each session a list of
records in temporal order) when you want sub-mechanism control without the
full `trace_to_card_v11` pipeline:

```python
from agingbench.telemetry.inference import (
    infer_compression, infer_interference, infer_revision, infer_maintenance,
)
from agingbench.telemetry.session_detection import detect_sessions
from agingbench.telemetry.adapters import get_adapter

# Build TelemetryRecord list from any source
adapter = get_adapter("openhands")
records = [r for r in (adapter(ev) for ev in raw_events) if r is not None]

# Bucket into sessions, then run any subset of mechanisms
sessions, mode = detect_sessions(records, idle_gap_minutes=30.0)
compression_block = infer_compression(sessions)
revision_block    = infer_revision(sessions)
```

Each inference function returns the same dict that lands under
`trace_audit[<mechanism>]`. Maintenance and revision additionally take
optional `outcomes` / shock-detection inputs — see the docstrings in
[`agingbench/telemetry/inference/`](inference/) for the full signatures.
The `_verdict.degradation_verdict()` helper underneath the
saturation-aware verdict is also public.

## File layout

```
telemetry/
├── __init__.py                  # public API
├── README.md                    # this file
├── trace_to_card.py             # both v1.0 stub + v1.1 pipeline
├── schema.py                    # canonical dataclasses
├── adapters/
│   ├── generic.py               # best-effort JSONL
│   ├── claude_code.py           # ~/.claude/projects/ session files
│   ├── langfuse_v1.py           # Langfuse spans (camelCase + snake_case)
│   ├── otlp_v1.py               # OTLP — gen_ai.* semconv + legacy llm.*
│   ├── openai_assistants.py     # Threads / Runs / RunSteps
│   ├── openhands.py             # OpenHands SDK event log
│   └── (langsmith routes through generic in v1.1)
├── session_detection.py         # explicit_id → user_id_split → idle_gap
├── memory_reconstruction.py     # 5-rule shock event detection
├── privacy_scrubber.py          # PII redaction + stable session-id hashing
├── outcome_extractors.py        # 3 built-in OutcomeEvent extractors
├── inference/
│   ├── compression.py           # saturation + self-contradiction + fact-density
│   ├── interference.py          # tool-distribution KL drift
│   ├── revision.py              # user-correction-repetition rate
│   └── maintenance.py           # pre/post-shock delta + outcome-conditional
├── profiles/
│   ├── generic.yaml
│   └── code_assistant.yaml
├── synthetic_probe.py           # scenarios-as-probes orchestration
└── example_traces/              # shipped fixtures
```

## Public API

```python
from agingbench.telemetry import (
    # v1.1 pipeline
    trace_to_card_v11, TraceToCardV11Result,
    # v1.0 stub (backward-compat)
    trace_to_card, TraceToCardResult, SUPPORTED_TRACE_FORMATS,
    # Schemas
    TelemetryRecord, OutcomeEvent, MemoryEvent, ToolCall,
    CoverageReport, TraceAuditBlock,
    # Discovery
    list_supported_formats, list_profiles, list_extractors,
    list_injectable_scenarios,
    # Profiles
    load_profile, Profile,
    # Outcome extractors
    run_extractor,
    extract_from_claude_session_flags,
    extract_from_record_patterns,
    extract_from_git_log,
    # Synthetic probes
    load_probe_result, merge_probe_into_card,
    ProbeSchedule, ProbeResult,
)
```

## Status: release-ready as of v1.1

| Component | Status | Test coverage |
|---|---|---|
| Schemas (TelemetryRecord, OutcomeEvent, MemoryEvent, …) | ✅ shipped | covered |
| 7 trace-format adapters (claude_code, generic, langfuse, langsmith, openai_assistants, openhands, otlp) | ✅ shipped | adapter-level + e2e |
| Session detection (explicit_id, idle_gap, user_id_split) | ✅ shipped | covered |
| Memory-event reconstruction (5 detectors) | ✅ shipped | covered |
| Privacy scrubber (7 default PII patterns + session-id hashing) | ✅ shipped | covered |
| 4-mechanism inference (compression, interference, revision, maintenance) | ✅ shipped | covered |
| Cost block aggregation from records | ✅ shipped | covered |
| 3 outcome extractors (claude_session_flags, record_patterns, git_log) | ✅ shipped | covered |
| 2 deployment profiles (generic, code_assistant) | ✅ shipped | covered |
| Synthetic-probe orchestrator | ✅ shipped | covered |
| Headline `m(t)` curve from outcomes | ✅ shipped | covered |
| Coverage verdicts on every mechanism | ✅ shipped | covered |
| Real-trace plug-and-play (Claude Code) | ✅ verified | manual + real-data e2e |

26 telemetry-specific tests pass; 167 total package tests pass / 0 fail / 2 skip.

## Roadmap (post-v1.1)

| Milestone | Scope |
|---|---|
| **v1.2 (planned)** | More outcome extractors: GitHub Actions CI status, Langfuse score events, Slack reaction signals. Validation correlation study against scenario-derived metrics. Better NER (filter code identifiers from self-contradiction detection). |
| **v1.3** | Cross-tenant aggregation with differential privacy. Real-time / streaming ingestion. Native protobuf OTLP support (currently JSON-OTLP only). |
| **v2.0** | Multilingual user-correction detection. Workspace-fidelity inference for self-planning agents (S5). |
