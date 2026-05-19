"""S8 — SWE-bench-Aging.

Tier-2 longitudinal benchmark anchored on a real OSS repository. Each
session = one curated GitHub issue from the repo's PR history. Workspace
(the repo clone) persists across sessions; the agent maintains its own
self-planned memory under `.aging/notes.md` (and may also use
`.claude/` if Claude Code is the SUT).

Aging signal:
  - headline: pass-rate of the repo's actual test suite per session
  - compression / interference / revision / maintenance: probes
    grounded in repo + workspace state

Status: fully implemented as of v0.3 (`django_orm_query` 8-issue chain
with synthetic load-bearing consistency tests, per-session Docker
container, four-mechanism probes). See `README.md` for usage and
`PROVENANCE.md` for SWE-bench attribution.
"""
from pathlib import Path

SCENARIO_DIR = Path(__file__).parent
