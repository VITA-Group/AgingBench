# Contributing to AgingBench

AgingBench accepts contributions in four shapes. The protocol is light and human-reviewed.

## Submitting a card

Run AgingBench against your model:

```bash
cd prototype
uv run agingbench run \
  --scenario s1_research_literature \
  --sut <your-sut.yaml> \
  --sessions 10 --seeds 3 --card
```

Validate:

```bash
python -m agingbench.metrics.aging_card_validate \
  experiments/results/<run-dir>/aging_card.json
```

Open a GitHub issue with the [AgingCard submission template](../.github/ISSUE_TEMPLATE/aging_card_submission.md), attach the card, and note the track. Full submission policy in [LEADERBOARD.md](LEADERBOARD.md).

## Adding a scenario

A scenario consists of a manifest, optional curated data, a generator, a runner, and an entry in the CLI dispatch table.

**1. Manifest** — `prototype/agingbench/scenarios/<sN_name>/scenario.yaml`:

```yaml
scenario_id: sN_my_scenario
display_name: "My Scenario"
metric_group: G2           # G1=task, G2=behavioral, G3=memory-quality, G4=efficiency
exposure_axis: t_steps     # or t_writes
default_sessions: 10
runner:
  module: agingbench.runner.sN_runner
  class: SNRunner
```

**2. Generator** (`prototype/agingbench/generators/sN_generator.py`) — subclass `BaseGenerator`. Use `FactGraph` for the temporal DAG and `PressureConfig` for difficulty knobs.

**3. Runner** (`prototype/agingbench/runner/sN_runner.py`) — subclass `BaseRunner` and implement `run(n_sessions, seed) -> RunResult`. Template: `s7_runner.py` (Tier-2 production-CLI) or `s2_runner.py` (Tier-1 controlled).

**4. CLI dispatch** — register in `prototype/agingbench/cli/runners.py::_SCENARIO_RUNNERS`.

**5. Tests** — add `prototype/tests/test_sN_*.py`. Pattern: `tests/test_s2_runner.py`.

## Adding a SUT (a new model or memory policy)

A SUT (System Under Test) bundles a model + memory policy + run hyperparameters in a single YAML at `prototype/agingbench/registry/suts/<family>/<sut_id>.yaml`. Adding one is dropping one such YAML and running it.

```yaml
sut_id: my_sut
description: "What this configuration tests"

model:
  provider: litellm                    # or: local_hf
  model: claude-haiku-4-5-20251001     # litellm model id, or HF repo id for local_hf
  max_tokens: 500
  temperature: 0.0

memory_policy:
  type: summarize_store                # see core/memory/ for the full set
  compaction_prompt: experiments/prompts/compact_lossy.txt
  word_budget: 200

seed: 42
```

The headline `memory_policy.type` is `summarize_store` (paired with a compaction prompt from [`../prototype/experiments/prompts/`](../prototype/experiments/prompts/) — `compact_lossy.txt` for aggressive "lossy" compression, `compact_medium.txt` for higher-fidelity "careful" compression). Baselines: `growing_history` (no compression) and `no_memory` (frozen). The other policies (episodic, chain-compress, typed-state, workspace, observer, …) extend the same `MemoryPolicy` interface — browse [`../prototype/agingbench/core/memory/`](../prototype/agingbench/core/memory/) for the full set.

For Tier-2 SUTs (S7, S8) where the agent owns its own session loop, use an `adapter:` block in place of `model:`. See [`../prototype/agingbench/registry/suts/openhands/`](../prototype/agingbench/registry/suts/openhands/) and [`../prototype/agingbench/registry/suts/claude_code/`](../prototype/agingbench/registry/suts/claude_code/).

## Adding an integration adapter

Read an `aging_card.json`, emit your eval / observability system's preferred format, drop the result in `prototype/examples/<target>_adapter.py`. The skeletons for OpenAI Evals / LangSmith / Langfuse / MCP are ~100-line starting points; open a PR with your file.

## Reporting a bug or disputing a card

Open a GitHub issue with the appropriate template:

- [`bug_report.md`](../.github/ISSUE_TEMPLATE/bug_report.md)
- [`scenario_request.md`](../.github/ISSUE_TEMPLATE/scenario_request.md)
- A dispute marked `[dispute]` in the title

Disputes are reviewed by two leaderboard operators (process in [LEADERBOARD.md](LEADERBOARD.md)).

## Code style + workflow

- Format: `uv run ruff format`. Lint: `uv run ruff check`.
- Tests: `uv run pytest tests/ -v` from `prototype/`. Don't merge with new failures.
- Branches: feature off `main`; PRs require one reviewer.
- Commit messages: imperative ("Add X", "Fix Y").
