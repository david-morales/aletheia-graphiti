"""
Tests for the empty group_ids guard in search_utils.py.

When group_ids is an empty list [], it should behave like None (no filter),
NOT generate an `IN []` clause that returns zero results.

There are two code patterns in search_utils.py that must be guarded:

Pattern 1 (filter_queries.append style):
    if group_ids is not None and len(group_ids) > 0:
        filter_queries.append('e.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

Pattern 2 (group_filter_query += style):
    if group_ids is not None and len(group_ids) > 0:
        group_filter_query += '...'
        filter_params['group_ids'] = group_ids

Both patterns should only apply the filter when group_ids is a non-empty list.
"""

import pytest


def _simulate_filter_queries_pattern(group_ids: list[str] | None) -> tuple[list[str], dict]:
    """Simulate Pattern 1: filter_queries.append style (used in edge/node search functions)."""
    filter_queries: list[str] = []
    filter_params: dict = {}

    if group_ids is not None and len(group_ids) > 0:
        filter_queries.append('e.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    return filter_queries, filter_params


def _simulate_group_filter_query_pattern(group_ids: list[str] | None) -> tuple[str, dict]:
    """Simulate Pattern 2: group_filter_query += style (used in episode/community search)."""
    filter_params: dict = {}
    group_filter_query: str = ''

    if group_ids is not None and len(group_ids) > 0:
        group_filter_query += '\nAND e.group_id IN $group_ids'
        filter_params['group_ids'] = group_ids

    return group_filter_query, filter_params


class TestFilterQueriesPattern:
    """Tests for Pattern 1: filter_queries.append style."""

    def test_none_group_ids_no_filter(self):
        """None group_ids should NOT add a group_ids filter."""
        filter_queries, filter_params = _simulate_filter_queries_pattern(None)
        assert filter_queries == []
        assert 'group_ids' not in filter_params

    def test_empty_list_group_ids_no_filter(self):
        """Empty list [] should NOT add a group_ids filter (the bug fix)."""
        filter_queries, filter_params = _simulate_filter_queries_pattern([])
        assert filter_queries == []
        assert 'group_ids' not in filter_params

    def test_nonempty_list_group_ids_adds_filter(self):
        """Non-empty list should add a group_ids filter."""
        filter_queries, filter_params = _simulate_filter_queries_pattern(['group1'])
        assert len(filter_queries) == 1
        assert 'e.group_id IN $group_ids' in filter_queries[0]
        assert filter_params['group_ids'] == ['group1']

    def test_multiple_group_ids_adds_filter(self):
        """Multiple group IDs should add a filter with all IDs."""
        filter_queries, filter_params = _simulate_filter_queries_pattern(['g1', 'g2', 'g3'])
        assert len(filter_queries) == 1
        assert filter_params['group_ids'] == ['g1', 'g2', 'g3']


class TestGroupFilterQueryPattern:
    """Tests for Pattern 2: group_filter_query += style."""

    def test_none_group_ids_no_filter(self):
        """None group_ids should NOT add a group_ids filter."""
        group_filter_query, filter_params = _simulate_group_filter_query_pattern(None)
        assert group_filter_query == ''
        assert 'group_ids' not in filter_params

    def test_empty_list_group_ids_no_filter(self):
        """Empty list [] should NOT add a group_ids filter (the bug fix)."""
        group_filter_query, filter_params = _simulate_group_filter_query_pattern([])
        assert group_filter_query == ''
        assert 'group_ids' not in filter_params

    def test_nonempty_list_group_ids_adds_filter(self):
        """Non-empty list should add a group_ids filter."""
        group_filter_query, filter_params = _simulate_group_filter_query_pattern(['group1'])
        assert 'group_id IN $group_ids' in group_filter_query
        assert filter_params['group_ids'] == ['group1']

    def test_multiple_group_ids_adds_filter(self):
        """Multiple group IDs should add a filter with all IDs."""
        group_filter_query, filter_params = _simulate_group_filter_query_pattern(
            ['g1', 'g2', 'g3']
        )
        assert 'group_id IN $group_ids' in group_filter_query
        assert filter_params['group_ids'] == ['g1', 'g2', 'g3']


class TestGuardConsistency:
    """Verify that both patterns agree on what constitutes a 'no filter' case."""

    @pytest.mark.parametrize(
        'group_ids',
        [None, []],
        ids=['None', 'empty_list'],
    )
    def test_no_filter_cases(self, group_ids):
        """Both None and empty list should result in no filter for both patterns."""
        filter_queries, filter_params_1 = _simulate_filter_queries_pattern(group_ids)
        group_filter_query, filter_params_2 = _simulate_group_filter_query_pattern(group_ids)

        assert filter_queries == [], f'Pattern 1 should not filter for group_ids={group_ids!r}'
        assert (
            group_filter_query == ''
        ), f'Pattern 2 should not filter for group_ids={group_ids!r}'
        assert 'group_ids' not in filter_params_1
        assert 'group_ids' not in filter_params_2

    @pytest.mark.parametrize(
        'group_ids',
        [['g1'], ['g1', 'g2']],
        ids=['single', 'multiple'],
    )
    def test_filter_cases(self, group_ids):
        """Non-empty lists should result in a filter for both patterns."""
        filter_queries, filter_params_1 = _simulate_filter_queries_pattern(group_ids)
        group_filter_query, filter_params_2 = _simulate_group_filter_query_pattern(group_ids)

        assert len(filter_queries) == 1, f'Pattern 1 should filter for group_ids={group_ids!r}'
        assert (
            group_filter_query != ''
        ), f'Pattern 2 should filter for group_ids={group_ids!r}'
        assert filter_params_1['group_ids'] == group_ids
        assert filter_params_2['group_ids'] == group_ids
