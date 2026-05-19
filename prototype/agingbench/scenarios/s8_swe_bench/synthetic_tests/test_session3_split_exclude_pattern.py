"""S9 synthetic test for session 3 (django-13158) — Django SimpleTestCase form.

See module docstring in synthetic_tests/__init__.py for context.
This test is injected into /testbed/tests/agingbench_syn/ at runtime
and run by Django's tests/runtests.py.
"""
import inspect
from django.test import SimpleTestCase


class Session3SplitExcludePatternTests(SimpleTestCase):
    def test_split_exclude_uses_clone_pattern(self):
        """Session-3 fix must use the clone-and-rewrite pattern from session 0."""
        from django.db.models.sql.query import Query
        source = inspect.getsource(Query.split_exclude)
        src_lc = source.lower()
        has_clone = "clone(" in src_lc
        has_empty_handling = ("set_empty" in src_lc or "is_empty" in src_lc
                              or "empty_query" in src_lc)
        self.assertTrue(
            has_clone,
            "Session-3 fix regressed split_exclude: missing clone() call. "
            "Convention from session 0 (django-11265) requires clone+rewrite.",
        )
        self.assertTrue(
            has_empty_handling,
            "Session-3 fix should handle the combined-empty-query case via "
            "set_empty/is_empty (convention from session 0).",
        )
