# S8 SWE-bench-Aging — Provenance and Attribution

## What this scenario contains

S8 layers AgingBench's longitudinal-pressure machinery over a curated
chain of real GitHub issues from a single OSS repository, drawn from
the **SWE-bench-Verified** subset. Each session = one real issue
with the upstream repo's actual test suite as the verifier, run
inside the SWE-bench-pre-built Docker container at the issue's
pre-resolution commit.

The canonical chain (`django_orm_query`) targets the Django ORM
query subsystem. Augmenting layer (Phase 17): synthetic consistency
tests injected into the f2p list make the agent's memory of cross-
session design conventions LOAD-BEARING on task pass-rate.

## Source dataset

- **Dataset:** `princeton-nlp/SWE-bench_Verified` on HuggingFace
- **Citation:** Jimenez et al., *"SWE-bench: Can Language Models
  Resolve Real-World GitHub Issues?"* ICLR 2024.
- **License (data):** MIT.
- **Upstream license (django/django):** BSD-3-Clause.
- **Vendoring:** S8 vendors NO upstream source code in-repo. Each
  session pulls a SWE-bench-pre-built Docker image at run time.
  The image contains the Django repo at the issue's pre-resolution
  commit + a conda env with pinned dependencies.

## Image source

```
docker.io/swebench/sweb.eval.x86_64.django_1776_django-{N}:latest
```

(Pulled by `docker pull` and re-tagged to
`sweb.eval.x86_64.django__django-{N}:latest` for the runner — see
README.md.)

## Chain selection — django_orm_query

The 8-issue chain (`issue_chains/django_orm_query.yaml`) was selected
from the 15 SWE-bench-Verified instances touching
`django/db/models/sql/query.py` based on:

1. **Symbol-level coupling.** Multiple PRs touch the same functions
   (`split_exclude` 3x, `resolve_lookup_value` 3x, `as_sql` 2x,
   `build_filter` 2x). Coupling makes interference + cross-session
   load-bearing measurable.

2. **Chronological reachability.** Issues span v3.0 → v4.2; later
   issues build on or interact with patterns from earlier ones.
   The capstone (django-16263) requires recall of patterns from PRs
   0, 2, 5.

3. **Test-coverage health.** All 8 PRs have well-defined fail-to-pass
   + pass-to-pass test sets in SWE-bench-Verified, with reasonable
   p2p denominators (10–145 tests).

The chain is **not bit-exact reproducible from the open data alone** —
seed manifest order in `seed_manifests/django_orm_query_seed_*.yaml`
is hand-curated.

## Synthetic-test layer (Phase 17)

`synthetic_tests/` contains 4 Django `SimpleTestCase` modules. Each
test inspects the agent's modified Django source AND/OR the agent's
`.aging/notes.md` for adherence to a chain-state convention. Tests
are injected into `/testbed/tests/agingbench_syn/` at session start
and added to the f2p list — failures degrade task pass-rate. Synthetic
tests are AgingBench-original code (not derived from upstream Django);
license follows the AgingBench repo (MIT).

## What S8 measures (cross-reference)

The four mechanisms (compression / interference / revision /
maintenance) are scored from artifacts the agent produces during
the session — `solution.diff`, `.aging/notes.md`, and per-session
`attestation.md` answers — plus the verifier's pass/fail per-test
results. See `probes.py` for the scoring functions; see paper
Table 3 for the canonical Tier-2 column mapping.
