# AgingBench Leaderboard

We accept AgingCard JSON submissions via GitHub. One card per (model × scenario × seed). Schema validation + reproducibility are the only acceptance criteria — no commercial endorsement, no single-scalar ranking.

## Tracks

| Track | What you submit | Anchor |
|---|---|---|
| **Tier 2 — Autonomous agent (S7)** | Agent that owns its session loop (Claude Code, OpenHands, custom) | Default surface; recall is the headline metric |
| **Tier 1 · A — Model swap** | LLM; `ReferenceAgent` + memory policy + seeds held constant | "Which model ages better under identical scaffolding?" |
| **Tier 1 · B — Memory policy** | `MemoryPolicy` subclass; model held at Haiku-4.5 baseline | Memory-systems research + retrieval/compression studies |
| **Tier 1 · C — Controller** | `ThresholdController` subclass | Opens in v1.1 |

## Submitting

1. **Generate** a card:
   ```bash
   cd prototype
   uv run agingbench run \
     --scenario <s1_…|s7_research_notes> \
     --sut <your-sut.yaml> \
     --sessions 10 --seeds 3 --card
   ```
2. **Validate** against the schema:
   ```bash
   python -m agingbench.metrics.aging_card_validate \
     experiments/results/<run-dir>/aging_card.json
   ```
3. **Open a GitHub issue** with the [AgingCard submission template](../.github/ISSUE_TEMPLATE/aging_card_submission.md), attach the card, and note the track (A / B / C / Tier 2).

If your run uses a forked AgingBench (modified scoring, custom scenarios), declare it on submission — fork cards are tagged and stored separately so the main leaderboard reflects only canonical runs.

## What we commit to

- **Weekly review** of submitted AgingCards.
- **8-week release cadence** for the codebase.
- **Backward-compatible AgingCard schema evolution** across minor versions.
- **Transparent governance**: if a submission turns out to be misrepresented, the card moves to `leaderboard/_retracted/` with the dispute reasoning. Nothing is silently deleted.

## Disputes

Open a `[dispute]` issue if a card misrepresents the model. Two leaderboard operators review; if upheld, the card moves to `leaderboard/_retracted/` with the dispute reasoning attached.

## Eligibility

- Card must validate against [`aging_card_schema.json`](../prototype/agingbench/metrics/aging_card_schema.json).
- Provenance `git_sha` must match a published AgingBench release (or be a disclosed fork).
- Conflict-of-interest disclosure required if the submitter is affiliated with the model provider.
- One card per (model × scenario × seed). Newer cards supersede older ones via a `superseded_by` link rather than replacing them.
