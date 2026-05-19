"""AgingBench — Longitudinal Reliability Benchmark for Memory-Enabled Agents.

Public surface:
    agingbench.cli       — command-line entry points (`agingbench run …`)
    agingbench.runner    — per-scenario session state machines
    agingbench.metrics   — aging curves, AgingCard schema, summarization
    agingbench.scenarios — eight scenario packages (s1…s8)
    agingbench.generators — pressure-config + dependency-graph generators
    agingbench.core      — LLM, memory-policy, agent, and adapter ABCs

The benchmark measures how an agent's memory degrades over operational
lifetimes across four mechanisms (compression, interference, revision,
maintenance). See `readme.md` for usage and the AgingCard schema at
`agingbench/metrics/aging_card_schema.json` for the run-output format.
"""

__version__ = "0.3.0"

__all__ = [
    "__version__",
]
