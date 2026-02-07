"""Tests for batch community projection queries in get_community_clusters.

Verifies that the batch query approach issues a single query per group_id
instead of O(n) individual queries (one per node).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.community_operations import (
    get_community_clusters,
)


def _make_entity_node(uuid: str, group_id: str = 'g1') -> EntityNode:
    """Create a minimal EntityNode for testing."""
    return EntityNode(
        uuid=uuid,
        name=f'node-{uuid}',
        group_id=group_id,
        labels=['Entity'],
        summary='test summary',
    )


def _make_driver(provider: GraphProvider = GraphProvider.NEO4J) -> MagicMock:
    """Create a mock GraphDriver."""
    driver = MagicMock()
    driver.provider = provider
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock()
    return driver


# ---------------------------------------------------------------------------
# test_batch_query_called_once_per_group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_query_called_once_per_group():
    """For a group with N nodes, execute_query should be called exactly once
    (the single batch query), not N times (one per node)."""

    nodes = [_make_entity_node('n1'), _make_entity_node('n2'), _make_entity_node('n3')]
    driver = _make_driver()

    # Batch query returns edges: n1-n2 (2 edges), n2-n3 (1 edge)
    batch_records = [
        {'source_uuid': 'n1', 'neighbor_uuid': 'n2', 'edge_count': 2},
        {'source_uuid': 'n2', 'neighbor_uuid': 'n1', 'edge_count': 2},
        {'source_uuid': 'n2', 'neighbor_uuid': 'n3', 'edge_count': 1},
        {'source_uuid': 'n3', 'neighbor_uuid': 'n2', 'edge_count': 1},
    ]
    driver.execute_query.return_value = (batch_records, None, None)

    with patch.object(EntityNode, 'get_by_group_ids', new_callable=AsyncMock) as mock_get, \
         patch.object(EntityNode, 'get_by_uuids', new_callable=AsyncMock) as mock_uuids:
        mock_get.return_value = nodes
        # get_by_uuids is called by label_propagation result processing;
        # return nodes grouped by cluster
        mock_uuids.return_value = nodes

        await get_community_clusters(driver, ['g1'])

    # Only ONE call to execute_query for the batch query (not 3 = one per node)
    assert driver.execute_query.call_count == 1

    # Verify the query was parameterized with group_id
    call_kwargs = driver.execute_query.call_args
    assert call_kwargs.kwargs.get('group_id') == 'g1' or \
           (len(call_kwargs.args) > 0 and 'group_id' in str(call_kwargs))


# ---------------------------------------------------------------------------
# test_empty_group_skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_group_skipped():
    """When a group has no nodes, no batch query should be issued and
    no clusters should be returned."""

    driver = _make_driver()

    with patch.object(EntityNode, 'get_by_group_ids', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = []

        clusters = await get_community_clusters(driver, ['empty_group'])

    # No execute_query call should have been made
    driver.execute_query.assert_not_called()
    assert clusters == []


# ---------------------------------------------------------------------------
# test_isolated_nodes_in_projection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_isolated_nodes_in_projection():
    """Nodes with no edges should appear in the projection with empty neighbor
    lists. They should each form their own cluster (isolated communities)."""

    nodes = [_make_entity_node('iso1'), _make_entity_node('iso2'), _make_entity_node('iso3')]
    driver = _make_driver()

    # Batch query returns no records (no edges at all)
    driver.execute_query.return_value = ([], None, None)

    with patch.object(EntityNode, 'get_by_group_ids', new_callable=AsyncMock) as mock_get, \
         patch.object(EntityNode, 'get_by_uuids', new_callable=AsyncMock) as mock_uuids:
        mock_get.return_value = nodes
        # Each isolated node forms its own cluster; get_by_uuids called per cluster
        mock_uuids.side_effect = lambda _driver, uuids: [
            n for n in nodes if n.uuid in uuids
        ]

        clusters = await get_community_clusters(driver, ['g1'])

    # Each isolated node should form its own cluster
    assert len(clusters) == 3
    all_node_uuids = {n.uuid for cluster in clusters for n in cluster}
    assert all_node_uuids == {'iso1', 'iso2', 'iso3'}

    # Only one batch query was issued
    assert driver.execute_query.call_count == 1


# ---------------------------------------------------------------------------
# test_multiple_groups_each_get_one_batch_query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_groups_each_get_one_batch_query():
    """When multiple group_ids are provided, each group should trigger exactly
    one batch query (not one per node)."""

    nodes_g1 = [_make_entity_node('a', 'g1'), _make_entity_node('b', 'g1')]
    nodes_g2 = [_make_entity_node('c', 'g2'), _make_entity_node('d', 'g2')]
    driver = _make_driver()

    # Return different records per call
    batch_records_g1 = [
        {'source_uuid': 'a', 'neighbor_uuid': 'b', 'edge_count': 2},
        {'source_uuid': 'b', 'neighbor_uuid': 'a', 'edge_count': 2},
    ]
    batch_records_g2 = [
        {'source_uuid': 'c', 'neighbor_uuid': 'd', 'edge_count': 3},
        {'source_uuid': 'd', 'neighbor_uuid': 'c', 'edge_count': 3},
    ]
    driver.execute_query.side_effect = [
        (batch_records_g1, None, None),
        (batch_records_g2, None, None),
    ]

    all_nodes = nodes_g1 + nodes_g2

    with patch.object(EntityNode, 'get_by_group_ids', new_callable=AsyncMock) as mock_get, \
         patch.object(EntityNode, 'get_by_uuids', new_callable=AsyncMock) as mock_uuids:
        mock_get.side_effect = [nodes_g1, nodes_g2]
        mock_uuids.side_effect = lambda _driver, uuids: [
            n for n in all_nodes if n.uuid in uuids
        ]

        clusters = await get_community_clusters(driver, ['g1', 'g2'])

    # Exactly 2 batch queries — one per group_id
    assert driver.execute_query.call_count == 2
