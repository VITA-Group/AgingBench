"""Shared eval-text memory proxy for clean C1-C4 attribution.

The Tier-1 runners compute a per-condition ``eval_text`` (the memory the agent
*should* see this session under the active oracle condition: compacted SUT
memory for C1, full store for C2, clean/gold source for C3, in-context ceiling
for C4). But ``ReferenceAgent`` reads ``memory_policy.read()`` directly when it
builds its system prompt — so unless the agent is constructed with this proxy,
its system prompt shows the SUT's lossy compaction even in the oracle modes,
silently breaking the C1-C4 isolation (an uncontrolled second memory channel).

Wrapping the policy with ``EvalTextMemoryProxy(real_policy, eval_text)`` makes
``.read()`` return the runner's mode-dependent ``eval_text`` (matching the
tool/context channel) while ``.write()`` and every other attribute
(``dump_store``, ``retrieve``, ``reset``, ``last_input_tokens``, …) pass through
to the real policy unchanged.

In C1 ``eval_text == real_policy.read()`` by construction, so wrapping is a
no-op for the baseline; only the oracle conditions change (they get fixed).

This mirrors the ``_EvalTextMemoryProxy`` already used by S1; S2/S3/S4 import
this shared version.
"""

from __future__ import annotations


class EvalTextMemoryProxy:
    """Proxy that returns a runner-supplied ``eval_text`` from ``.read()`` and
    delegates everything else (including ``.write()``) to the real policy."""

    def __init__(self, real_policy, eval_text: str):
        self._real = real_policy
        self._text = eval_text

    def read(self, *args, **kwargs):
        # Accept (and ignore) an optional query arg so callers that pass
        # read(query=...) for retrieval policies still work — the eval_text is
        # already the mode-resolved view for this session.
        return self._text

    def write(self, *args, **kwargs):
        return self._real.write(*args, **kwargs)

    def __getattr__(self, name):
        # Only reached for attributes not defined above (delegates dump_store,
        # retrieve, reset, entry_count, last_input_tokens, retriever, etc.).
        return getattr(self._real, name)
