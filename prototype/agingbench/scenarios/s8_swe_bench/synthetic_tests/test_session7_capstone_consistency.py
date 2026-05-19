"""S9 synthetic test for session 7 (django-16263) — Django SimpleTestCase form."""
import inspect
from django.test import SimpleTestCase


class Session7CapstoneConsistencyTests(SimpleTestCase):
    def test_capstone_combines_split_exclude_and_resolve_lookup(self):
        """Capstone must use patterns from sessions 0, 2, and 5."""
        from django.db.models.sql.query import Query
        # 1. split_exclude clone pattern
        se_source = inspect.getsource(Query.split_exclude)
        self.assertIn("clone(", se_source.lower(),
            "Capstone regressed session 0's split_exclude clone pattern.")
        # 2. resolve_lookup_value container preservation
        rlv_source = inspect.getsource(Query.resolve_lookup_value)
        rlv_lc = rlv_source.lower()
        self.assertTrue(
            "type(value)" in rlv_source
            or ("list" in rlv_lc and "tuple" in rlv_lc),
            "Capstone regressed session 2's container-preservation rule.")
        # 3. FilteredRelation handling
        src = inspect.getsource(Query)
        src_lc = src.lower()
        self.assertTrue(
            "filteredrelation" in src_lc or "filtered_relation" in src_lc,
            "Capstone regressed session 5's FilteredRelation handling.")
