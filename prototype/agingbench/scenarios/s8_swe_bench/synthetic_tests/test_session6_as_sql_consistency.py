"""S9 synthetic test for session 6 (django-16032) — Django SimpleTestCase form."""
import inspect
from django.test import SimpleTestCase


class Session6AsSqlConsistencyTests(SimpleTestCase):
    def test_as_sql_preserves_outerref_pattern(self):
        """Session-6 fix must not regress session-1's OuterRef in exclude handling."""
        from django.db.models.sql.query import Query
        source = inspect.getsource(Query)
        src_lc = source.lower()
        has_model_propagation = (
            "self.model" in source
            or "outer_query" in src_lc
            or "filtered_relation" in src_lc
        )
        self.assertTrue(
            has_model_propagation,
            "Session-6 fix appears to regress session-1's OuterRef handling. "
            "Expected explicit model propagation pattern (self.model / "
            "outer_query / filtered_relation references).",
        )
