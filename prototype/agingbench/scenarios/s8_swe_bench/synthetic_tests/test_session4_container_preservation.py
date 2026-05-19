"""S9 synthetic test for session 4 (django-13590) — Django SimpleTestCase form.

REVISION-LOAD-BEARING test (Phase 17.2).

Two halves:

1. CODE check (the original Phase 17 test): the agent's diff to
   `resolve_lookup_value` must preserve container type — verifies the
   agent didn't regress session 2's pattern.

2. REVISION-AGING check (NEW): the agent's `.aging/notes.md` must
   reference the container_preservation_rule convention announced at
   session 2 via state_change in ISSUE.md.

   This is the load-bearing revision probe — the convention is NOT in
   /testbed source, only in the announcement message that landed in
   ISSUE.md at session 2. To pass this check, the agent must have
   transcribed the rule into notes.md AND that note must have survived
   compression / writes through to session 4.

   PASS: agent's notes mention the convention -> revision intact
   FAIL: agent's notes lost the convention -> revision-aging detected

   Failure here directly degrades S9 task pass-rate (this test is in
   the f2p list), making memory of cross-session announcements
   load-bearing on capability.
"""
import inspect
import os
import re
from django.test import SimpleTestCase

NOTES_PATH = "/agentmemory/.aging/notes.md"

# Markers indicating the agent transcribed the container_preservation_rule.
# Match ANY of these — the rule's wording can be paraphrased.
_CONVENTION_MARKERS = [
    r"container[_\s]preservation",       # the fact_id itself or close variant
    r"preserve.{0,30}container",          # "preserve the container type" etc.
    r"list[\s\w]{0,15}stays?[\s\w]{0,15}list",   # "list stays list"
    r"original\s+container\s+type",       # quoted from announcement
    r"do\s+not\s+coerce",                  # quoted from announcement
    r"container\s+type",                   # generic mention
]


class Session4ContainerPreservationTests(SimpleTestCase):

    def test_resolve_lookup_value_preserves_container_type(self):
        """CODE check: agent's diff didn't regress session-2's pattern."""
        from django.db.models.sql.query import Query
        source = inspect.getsource(Query.resolve_lookup_value)
        src_lc = source.lower()
        uses_type_introspection = (
            "type(value)" in source or "type(rhs_value)" in source
        )
        handles_list = "list" in src_lc
        handles_tuple = "tuple" in src_lc
        self.assertTrue(
            uses_type_introspection or (handles_list and handles_tuple),
            "Code regression: session-4 fix should preserve container "
            "type (convention from session 2): either use type(value)(...) "
            "or explicitly handle list, tuple, and namedtuple variants.",
        )

    def test_agent_notes_acknowledge_container_preservation_announcement(self):
        """REVISION-AGING: agent's notes must reference the convention
        announced at session 2 (via state_change in ISSUE.md).

        Memory-load-bearing: the announcement is NOT in Django source —
        only in ISSUE.md text. The agent must have written it into
        .aging/notes.md AND the note must survive to session 4.
        """
        if not os.path.exists(NOTES_PATH):
            self.fail(
                "Agent's notes.md missing at /agentmemory/.aging/notes.md "
                "— revision aging cannot be tested. (Agent failed to write "
                "any notes across 4 sessions.)"
            )
        notes = open(NOTES_PATH, encoding="utf-8", errors="replace").read().lower()
        if not notes.strip():
            self.fail(
                "Agent's notes.md is empty at session 4 — revision aging "
                "complete (no memory of session-2 announcement survived)."
            )
        matched = [m for m in _CONVENTION_MARKERS if re.search(m, notes)]
        self.assertTrue(
            matched,
            "REVISION-AGING DETECTED: agent's notes do not reference the "
            "container_preservation_rule convention announced at session 2. "
            "Tried markers: " + str(_CONVENTION_MARKERS) + ". Notes excerpt "
            "(first 600 chars): " + notes[:600],
        )
