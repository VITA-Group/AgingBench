# AgingBench GitHub Actions workflows

This directory contains CI workflows for the AgingBench upstream repo
itself. Product teams who want to run AgingBench-Lite in their own CI
should use the drop-in template in
[`examples/ci/agingbench-lite-template.yml`](../../examples/ci/agingbench-lite-template.yml),
NOT this upstream workflow.

## Files

- `agingbench-lite-ci.yml` — upstream CI: runs the lite-suite scope
  validation + unit tests on every PR. Does NOT run live LLMs (no API
  keys configured in the upstream repo); product teams hook real keys
  via the template below.

## For product teams: how to adopt AgingBench-Lite in your CI

1. Copy the template:
   ```bash
   curl -O https://raw.githubusercontent.com/AgingBench/AgingBench/main/examples/ci/agingbench-lite-template.yml
   mkdir -p .github/workflows
   mv agingbench-lite-template.yml .github/workflows/agingbench-lite.yml
   ```
2. Add `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) to your repo secrets:
   Settings → Secrets and variables → Actions → New repository secret.
3. Author your SUT YAML (`your-sut-config.yaml`) at the repo root or
   anywhere on the working tree. See
   [`prototype/agingbench/registry/suts/haiku45/`](../../prototype/agingbench/registry/suts/haiku45/)
   for the schema.
4. Edit the workflow's `--sut your-sut-config.yaml` to point at it.
5. Push the PR. The workflow runs the lite suite (S1, S2, S7) at 3
   seeds, emits AgingCards, validates them against the v1.0.0 schema,
   and posts a results summary on the PR.

## What the lite suite tests

- **S1 Research Literature** — compression aging (fact survival under
  write-time abstraction).
- **S2 Lifestyle Assistant** — silent precision loss (constraint
  adherence over time).
- **S7 Research Notes** — production-CLI workspace memory (Tier-2).

If any of these scenarios shows substantially worse aging vs the prior
PR baseline, your CI fails (or comments a warning) before the regression
ships to production.

## Compute budget

The lite suite is pinned to <$5 / <30 min per CI run on Haiku-class
models. If your CI exceeds this, you've likely:
- Run the full suite instead of lite
- Set `--seeds` > 3
- Used a frontier model SUT instead of Haiku

## Troubleshooting

- **"No tests collected"**: check Python ≥ 3.10 + `agingbench-lite`
  installed.
- **"AgingCard validation failed"**: open the failing card; the
  validator prints which fields are missing/invalid.
- **"Suite 'full' not in lite subset"**: you're trying to run the full
  suite via the lite CLI. Switch to the full `agingbench` package or
  pin the workflow to `--suite lite`.

See the [CONTRIBUTING.md](../../docs/CONTRIBUTING.md) for the full
contribution workflow if you want to upstream changes.
