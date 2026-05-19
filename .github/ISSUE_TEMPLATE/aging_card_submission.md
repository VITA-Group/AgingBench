---
name: AgingCard result submission
about: Submit an external model result for the AgingBench leaderboard
title: "[card] <model> on <scenario>"
labels: leaderboard-submission
assignees: ''
---

<!--
  Use this template to submit a third-party AgingCard JSON result for the
  AgingBench leaderboard. A leaderboard operator will review your card and,
  if accepted, merge it into the leaderboard/ directory.

  See docs/LEADERBOARD.md for the full submission policy.
-->

## Model under test

<!-- Provider + model ID, e.g., "Anthropic claude-haiku-4-5-20251001" -->

## Scenario

<!-- Scenario ID (e.g., s1_research_literature, s8_terminal_bench) -->

## SUT configuration

<!-- Attach the SUT YAML you used. -->

## AgingCard JSON

<!-- Attach the aging_card.json file. Multiple cards = multiple issues. -->

## Validation

- [ ] Card validates against `aging_card_schema.json`:
      `python -m agingbench.metrics.aging_card_validate <card.json>` exits 0
- [ ] Card was generated using `agingbench run ... --card` (or `agingbench-lite run`)
- [ ] No manual edits to `aging_card.json` after generation
- [ ] Seeds are explicitly set (no implicit randomness)

## Reproducibility statement

<!-- Compute environment (local Docker, GitHub Actions, cloud, etc.).
     Commit hash of AgingBench used to generate the card. -->

## Disclosures

- [ ] All API costs / compute self-funded; no conflict of interest with model provider
- [ ] OR conflict disclosed below

<!-- Disclose any conflicts (e.g., employee of the model provider). -->

## Additional notes

<!-- Anything reviewers should consider when evaluating this submission. -->
