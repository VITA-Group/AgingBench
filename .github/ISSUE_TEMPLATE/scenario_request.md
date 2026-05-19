---
name: New scenario request
about: Propose a new scenario to add to AgingBench
title: "[scenario] "
labels: scenario-request
assignees: ''
---

## Scenario name

<!-- Proposed ID (e.g., s10_my_scenario) and display name. -->

## Deployment archetype

<!-- What real-world agent deployment does this scenario mirror?
     E.g., "Long-running customer-service agent with ticketing workflow." -->

## Which aging mechanisms does it test?

- [ ] Compression — fact loss under write-time abstraction
- [ ] Interference — confusion among similar entries as state grows
- [ ] Revision — failure to track updated facts
- [ ] Maintenance — degradation around lifecycle events (recompaction, flush, schema migration)

## Tier (runner-controlled vs autonomous)

- [ ] Tier-1: runner-managed memory (S1–S6 style)
- [ ] Tier-2: agent-managed workspace (S7, S8 style)
- [ ] Anchored on a community benchmark (Terminal-bench / SWE-bench / τ-bench / etc.)?
      If yes, which one and at what pinned commit hash?

## Task / probe sketch

<!-- 2–3 example tasks and 2–3 example probes the agent would face. -->

## DAG structure

<!-- Will this scenario use FactGraph version chains, dependency edges,
     interference pairs, accumulators? Sketch the rough topology. -->

## Lifecycle events

<!-- Any maintenance-style events to inject (recompaction, flush, schema
     migration, dependency upgrade, etc.)? -->

## Scoring approach

<!-- Headline metric and how it's computed.
     E.g., per-session pass rate, workspace fidelity, downstream recall. -->

## Estimated cost per run

<!-- Roughly: tokens per session, sessions per run, API cost class
     (Haiku-tier vs frontier). -->

## Why this scenario is worth adding

<!-- What new aging-mechanism coverage or deployment realism does it bring
     that existing scenarios don't? -->

## Are you willing to implement it?

- [ ] Yes, I will submit the PR
- [ ] No, requesting from the maintainers
- [ ] Looking for a collaborator

## Additional context

<!-- Links to related research, original benchmarks, or prior art. -->
