"""Tests for BFS result deduplication at source in search_utils.py.

Verifies that node_bfs_search and edge_bfs_search deduplicate records by UUID
before constructing domain objects, so duplicate traversal paths don't produce
duplicate results.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_driver(provider: GraphProvider = GraphProvider.FALKORDB):
    """Return a mock GraphDriver with the given provider."""
    driver = MagicMock()
    driver.provider = provider
    driver.search_interface = None
    return driver


def _node_record(uuid: str, name: str = 'n') -> dict:
    """Minimal record dict matching get_entity_node_return_query output."""
    return {
        'uuid': uuid,
        'name': name,
        'group_id': 'g1',
        'created_at': datetime.now(timezone.utc),
        'summary': 'summary',
        'labels': ['Entity'],
        'attributes': {},
    }


def _edge_record(uuid: str, name: str = 'e') -> dict:
    """Minimal record dict matching get_entity_edge_return_query output."""
    return {
        'uuid': uuid,
        'source_node_uuid': 'src-1',
        'target_node_uuid': 'tgt-1',
        'group_id': 'g1',
        'created_at': datetime.now(timezone.utc),
        'name': name,
        'fact': 'some fact',
        'episodes': ['ep1'],
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
        'attributes': {},
    }


# ---------------------------------------------------------------------------
# node_bfs_search tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_bfs_dedup_removes_duplicates():
    """When multiple BFS sub-queries return the same node UUID, only one copy should appear.

    Uses KUZU provider because it generates multiple match_queries (Episodic MENTIONS,
    Entity RELATES_TO, and combined) when bfs_max_depth >= 2, exercising cross-query dedup.
    """
    driver = _make_driver(GraphProvider.KUZU)

    # KUZU with bfs_max_depth=2 generates 3 match_queries
    call_count = 0

    async def fake_execute_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ([_node_record('node-1'), _node_record('node-2')], None, None)
        elif call_count == 2:
            # Second sub-query returns a duplicate of node-1
            return ([_node_record('node-1')], None, None)
        else:
            return ([], None, None)

    driver.execute_query = fake_execute_query

    from graphiti_core.search.search_utils import node_bfs_search
    from graphiti_core.search.search_filters import SearchFilters

    nodes = await node_bfs_search(
        driver=driver,
        bfs_origin_node_uuids=['origin-1'],
        search_filter=SearchFilters(),
        bfs_max_depth=2,
        group_ids=['g1'],
        limit=100,
    )

    uuids = [n.uuid for n in nodes]
    assert len(uuids) == 2, f'Expected 2 unique nodes, got {len(uuids)}: {uuids}'
    assert uuids == ['node-1', 'node-2']


@pytest.mark.asyncio
async def test_node_bfs_preserves_order():
    """The first occurrence of a UUID should be kept, not a later duplicate."""
    driver = _make_driver(GraphProvider.KUZU)

    call_count = 0

    async def fake_execute_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ([_node_record('aaa', 'first-a'), _node_record('bbb', 'first-b')], None, None)
        elif call_count == 2:
            return ([_node_record('bbb', 'second-b'), _node_record('ccc', 'first-c')], None, None)
        else:
            return ([], None, None)

    driver.execute_query = fake_execute_query

    from graphiti_core.search.search_utils import node_bfs_search
    from graphiti_core.search.search_filters import SearchFilters

    nodes = await node_bfs_search(
        driver=driver,
        bfs_origin_node_uuids=['origin-1'],
        search_filter=SearchFilters(),
        bfs_max_depth=2,
        group_ids=['g1'],
        limit=100,
    )

    uuids = [n.uuid for n in nodes]
    names = [n.name for n in nodes]
    assert uuids == ['aaa', 'bbb', 'ccc']
    assert names == ['first-a', 'first-b', 'first-c'], (
        'The first occurrence of bbb should be kept, not the second'
    )


@pytest.mark.asyncio
async def test_node_bfs_no_duplicates_unchanged():
    """When there are no duplicates, all records should be preserved."""
    driver = _make_driver(GraphProvider.KUZU)

    call_count = 0

    async def fake_execute_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ([_node_record('aaa'), _node_record('bbb')], None, None)
        elif call_count == 2:
            return ([_node_record('ccc'), _node_record('ddd')], None, None)
        else:
            return ([], None, None)

    driver.execute_query = fake_execute_query

    from graphiti_core.search.search_utils import node_bfs_search
    from graphiti_core.search.search_filters import SearchFilters

    nodes = await node_bfs_search(
        driver=driver,
        bfs_origin_node_uuids=['origin-1'],
        search_filter=SearchFilters(),
        bfs_max_depth=2,
        group_ids=['g1'],
        limit=100,
    )

    uuids = [n.uuid for n in nodes]
    assert len(uuids) == 4
    assert uuids == ['aaa', 'bbb', 'ccc', 'ddd']


# ---------------------------------------------------------------------------
# edge_bfs_search tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_bfs_dedup_removes_duplicates():
    """When multiple BFS sub-queries return the same edge UUID, only one copy should appear."""
    driver = _make_driver(GraphProvider.KUZU)

    call_count = 0

    async def fake_execute_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ([_edge_record('edge-1'), _edge_record('edge-2')], None, None)
        else:
            return ([_edge_record('edge-1')], None, None)

    driver.execute_query = fake_execute_query

    from graphiti_core.search.search_utils import edge_bfs_search
    from graphiti_core.search.search_filters import SearchFilters

    edges = await edge_bfs_search(
        driver=driver,
        bfs_origin_node_uuids=['origin-1'],
        bfs_max_depth=2,
        search_filter=SearchFilters(),
        group_ids=['g1'],
        limit=100,
    )

    uuids = [e.uuid for e in edges]
    assert len(uuids) == 2, f'Expected 2 unique edges, got {len(uuids)}: {uuids}'
    assert uuids == ['edge-1', 'edge-2']
