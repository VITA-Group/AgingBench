<!--
  AgingBench pull request template.
  Land the non-interference checklist with every PR that touches shared code
  in agingbench/cli/, agingbench/generators/, agingbench/metrics/, or
  agingbench/runner/.
-->

## Summary

<!-- One or two sentences describing what this PR changes and why. -->

## Linked issue / scenario

<!-- e.g., Closes #123, scenario s8_swe_bench. -->

## Non-Interference Checklist (required for shared-code PRs)

If this PR touches any of the following, every box below must be checked or
the deviation explained:

- `agingbench/cli/runners.py`, `agingbench/cli/loaders.py`, `agingbench/cli/__init__.py`
- `agingbench/generators/` (any file)
- `agingbench/metrics/` (any file)
- `agingbench/runner/` (any file)
- `agingbench/core/` (any file)

```
- [ ] No existing dict key behavior changed (only new keys added)
- [ ] No existing function signature changed (only new functions added)
- [ ] No existing output file modified (only new files added)
- [ ] No existing CLI flag behavior changed (only new optional flags added with default=False)
- [ ] No existing test broken; net new tests added are passing
- [ ] If any of the above is violated, explain why and link to affected scenarios in the section below
```

### Deviation explanation (if any)

<!-- If you had to break additive-only, explain here and link the affected scenarios. -->

## Test plan

<!-- List the tests you ran, expected outcomes, and any tests you added. -->

- [ ] Unit tests pass: `cd prototype && uv run pytest tests/ -v`
- [ ] If touching scenario code: smoke run on one scenario completes
- [ ] If touching AgingCard: `python -m agingbench.metrics.aging_card_validate <card.json>` exits 0

## Reviewer notes

<!-- Anything reviewers should pay extra attention to. -->
