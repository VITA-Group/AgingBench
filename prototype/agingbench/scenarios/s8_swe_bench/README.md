# S8 — SWE-bench-Aging (django_orm_query, load-bearing)

Tier-2 longitudinal evaluation of a long-running developer agent on a
real OSS repository. Each session = one curated GitHub issue from the
SWE-bench-Verified set, run inside the SWE-bench-pre-built Docker
container at the issue's pre-resolution commit.

The canonical chain (`django_orm_query`) is 8 SWE-bench-Verified PRs
all touching `django/db/models/sql/query.py` (v3.0 → v4.2). Strong
symbol coupling: `split_exclude` (3 PRs), `resolve_lookup_value`
(3 PRs), `build_filter` (2 PRs), `as_sql` (2 PRs). The capstone PR
combines patterns from 4 prior sessions.

## What makes this load-bearing

S8 augments the upstream Django tests with **synthetic consistency
tests** injected into the f2p list at sessions 3, 4, 6, 7. Each
synthetic test inspects the agent's modified Django source AND/OR
the agent's `.aging/notes.md` for adherence to a chain-state design
decision established by an earlier session. Failures degrade task
pass-rate, so memory of cross-session conventions is load-bearing on
capability — closing the gap that would exist if the agent could
fully bypass memory by re-deriving from /testbed.

## Output artifacts (per session)

- `metrics.json` — aging curve + `mechanism_metrics` block
  (`compression`, `interference`, `revision`, `maintenance`)
- `dependency_metrics.json` — sidecar with per-mechanism details
- `session_results.json` — per-session agent + verification records
- `lifecycle_events.json` — scheduled flushes / dep_pins / dep_bumps
- `aging_card.json` — v1.0.0 schema-validated AgingCard

## Docker image requirement

The 8 chain issues require ~24 GB of Docker image cache (per-issue
SWE-bench-pre-built Django images, layer-shared). Pre-pull once with:

```bash
for n in 11265 11734 12050 13158 13590 15554 16032 16263; do
  docker pull swebench/sweb.eval.x86_64.django_1776_django-${n}:latest
  docker tag swebench/sweb.eval.x86_64.django_1776_django-${n}:latest \
             sweb.eval.x86_64.django__django-${n}:latest
done
```

The re-tag is required so the runner finds the image at the expected name.

## Layout

```
s8_swe_bench/
├── __init__.py
├── scenario.yaml                          # manifest (default chain pinned)
├── PROVENANCE.md                          # SWE-bench + django attribution
├── README.md                              # this file
├── docker_runner.py                       # per-session container lifecycle
├── lifecycle.py                           # PressureConfig -> events; dep_pin
├── agent.py                               # claude_code/openhands/litellm bridge
├── verifier.py                            # apply_diff + run_verification
│                                          # (pytest + django runtests.py routing)
├── probes.py                              # 4-mechanism probes + orthogonal contrasts
├── issue_chains/
│   └── django_orm_query.yaml              # 8-issue chain + state_changes +
│                                          # synthetic_tests references
├── seed_manifests/
│   ├── django_orm_query_seed_42.yaml      # chronological pinning
│   └── django_orm_query_seed_43.yaml      # variance ordering
└── synthetic_tests/                       # Django SimpleTestCase tests
                                           # injected into /testbed at runtime
```

## How to invoke

```bash
agingbench run \
  --scenario s8_swe_bench \
  --sut agingbench/registry/suts/claude_code/claude_code_sonnet46_s8.yaml \
  --generated --sessions 8 \
  --output experiments/results/my_run \
  --card
```

See `PROVENANCE.md` for chain rationale and SWE-bench attribution.
