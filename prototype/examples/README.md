# AgingBench Integration Examples

This directory contains reference adapters that translate AgingCard JSON
into the formats expected by popular eval and observability systems, plus
sample cards and a CI template for downstream adoption.

## Available adapters

| Adapter | Target system | Notes |
|---|---|---|
| [`openai_evals_adapter.py`](openai_evals_adapter.py) | OpenAI Evals | Maps AgingCard → an `Eval` registration |
| [`langsmith_adapter.py`](langsmith_adapter.py) | LangChain LangSmith | Maps AgingCard → LangSmith run + tags |
| [`langfuse_adapter.py`](langfuse_adapter.py) | Langfuse / OpenTelemetry | Maps AgingCard → Langfuse trace + scores |
| [`mcp_adapter.py`](mcp_adapter.py) | Model Context Protocol | Translates AgingCard into MCP-style tool / memory event records |

All four are v1 reference skeletons (~100 lines each). They are starting
points — extend for your environment.

## How to use

Each adapter follows the same CLI pattern:

```bash
python examples/<adapter>.py --card <path-to-aging_card.json> --out <output>
```

## Sample cards for testing

[`sample_cards/`](sample_cards/) contains 8 canonical AgingCard JSONs
(one per scenario, all from a Haiku-class SUT) that adapters can use for
development without running a full scenario:

```
s1_research_literature_haiku45_lossy_compress.json
s2_lifestyle_assistant_haiku45_lossy_compress.json
s3_knowledge_base_haiku45_lossy_compress.json
s4_software_engineering_haiku45_lossy_compress.json
s5_self_planning_haiku45_lossy_compress.json
s6_naturalistic_haiku45_lossy_compress.json
s7_research_notes_haiku45_lossy_compress.json
s8_swe_bench_claude_code_s8.json
```

These are fixture data; not real model results. Use them when iterating
on an adapter before pointing it at a real run.

## CI template

[`ci/agingbench-lite-template.yml`](ci/agingbench-lite-template.yml) is
a drop-in GitHub Actions workflow that product teams can copy into their
own repo. It installs `agingbench-lite`, runs the lite suite (S1, S2, S7)
against a Haiku-class SUT on every PR, validates the emitted AgingCards
against the schema, and surfaces the headline metrics in the PR check.

## Authoring a new adapter

Pattern: a ~100-line Python file that

1. reads the AgingCard JSON,
2. translates the fields into the target system's payload shape,
3. writes the output (file or HTTP POST).

See [`openai_evals_adapter.py`](openai_evals_adapter.py) as the reference.

Schema reference:
[`agingbench/metrics/aging_card_schema.json`](../agingbench/metrics/aging_card_schema.json)
documents every field a v1.0.0 AgingCard contains. Validator:
`python -m agingbench.metrics.aging_card_validate <card.json>`.
