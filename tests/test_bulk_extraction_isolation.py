"""Tests for per-episode isolation in bulk extraction.

Verifies that extract_nodes_and_edges_bulk catches per-episode failures
and returns failed_indices instead of propagating exceptions to the caller.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.bulk_utils import extract_nodes_and_edges_bulk


def _make_episode(name: str, group_id: str = 'test-group') -> EpisodicNode:
    return EpisodicNode(
        name=name,
        labels=[],
        source=EpisodeType.text,
        content=f'Content for {name}',
        source_description='test',
        group_id=group_id,
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
    )


def _make_entity_node(name: str, group_id: str = 'test-group') -> EntityNode:
    return EntityNode(
        name=name,
        labels=['Entity'],
        group_id=group_id,
        summary=f'{name} summary',
        created_at=datetime.now(timezone.utc),
    )


def _make_entity_edge(
    source_uuid: str, target_uuid: str, group_id: str = 'test-group'
) -> EntityEdge:
    return EntityEdge(
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        name='RELATES_TO',
        fact=f'{source_uuid} relates to {target_uuid}',
        group_id=group_id,
        episodes=[],
        created_at=datetime.now(timezone.utc),
        expired_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
        invalid_at=datetime.now(timezone.utc),
    )


def _make_clients_mock():
    """Create a minimal mock for GraphitiClients."""
    clients = AsyncMock()
    clients.llm_client = AsyncMock()
    clients.embedder = AsyncMock()
    clients.driver = AsyncMock()
    return clients


@pytest.mark.asyncio
async def test_extraction_isolates_per_episode_failures():
    """3 episodes, middle one fails node extraction. Assert 2 successful results, failed_indices=[1]."""
    ep0 = _make_episode('ep0')
    ep1 = _make_episode('ep1')
    ep2 = _make_episode('ep2')

    node_a = _make_entity_node('Alice')
    node_b = _make_entity_node('Bob')

    edge_a = _make_entity_edge(node_a.uuid, node_b.uuid)
    edge_b = _make_entity_edge(node_b.uuid, node_a.uuid)

    episode_tuples = [
        (ep0, []),
        (ep1, []),
        (ep2, []),
    ]

    call_count = 0

    async def mock_extract_nodes(clients, episode, previous_episodes, **kwargs):
        nonlocal call_count
        idx = call_count
        call_count += 1
        if episode.name == 'ep1':
            raise RuntimeError('LLM extraction failed for ep1')
        if episode.name == 'ep0':
            return [node_a]
        return [node_b]

    async def mock_extract_edges(clients, episode, nodes, previous_episodes, **kwargs):
        if episode.name == 'ep0':
            return [edge_a]
        return [edge_b]

    with patch(
        'graphiti_core.utils.bulk_utils.extract_nodes', side_effect=mock_extract_nodes
    ), patch(
        'graphiti_core.utils.bulk_utils.extract_edges', side_effect=mock_extract_edges
    ):
        nodes_bulk, edges_bulk, failed_indices = await extract_nodes_and_edges_bulk(
            _make_clients_mock(),
            episode_tuples,
            edge_type_map={},
        )

    assert failed_indices == [1]
    assert len(nodes_bulk) == 2
    assert len(edges_bulk) == 2
    # Verify the content: ep0 got node_a, ep2 got node_b
    assert nodes_bulk[0] == [node_a]
    assert nodes_bulk[1] == [node_b]
    assert edges_bulk[0] == [edge_a]
    assert edges_bulk[1] == [edge_b]


@pytest.mark.asyncio
async def test_extraction_all_succeed_returns_empty_failures():
    """2 episodes, both succeed. Assert failed_indices=[]."""
    ep0 = _make_episode('ep0')
    ep1 = _make_episode('ep1')

    node_a = _make_entity_node('Alice')
    node_b = _make_entity_node('Bob')

    edge_a = _make_entity_edge(node_a.uuid, node_b.uuid)
    edge_b = _make_entity_edge(node_b.uuid, node_a.uuid)

    episode_tuples = [
        (ep0, []),
        (ep1, []),
    ]

    async def mock_extract_nodes(clients, episode, previous_episodes, **kwargs):
        if episode.name == 'ep0':
            return [node_a]
        return [node_b]

    async def mock_extract_edges(clients, episode, nodes, previous_episodes, **kwargs):
        if episode.name == 'ep0':
            return [edge_a]
        return [edge_b]

    with patch(
        'graphiti_core.utils.bulk_utils.extract_nodes', side_effect=mock_extract_nodes
    ), patch(
        'graphiti_core.utils.bulk_utils.extract_edges', side_effect=mock_extract_edges
    ):
        nodes_bulk, edges_bulk, failed_indices = await extract_nodes_and_edges_bulk(
            _make_clients_mock(),
            episode_tuples,
            edge_type_map={},
        )

    assert failed_indices == []
    assert len(nodes_bulk) == 2
    assert len(edges_bulk) == 2


@pytest.mark.asyncio
async def test_edge_extraction_failure_also_captured():
    """3 episodes, all node extraction succeeds, but one edge extraction fails.
    Assert that episode is in failed_indices."""
    ep0 = _make_episode('ep0')
    ep1 = _make_episode('ep1')
    ep2 = _make_episode('ep2')

    node_a = _make_entity_node('Alice')
    node_b = _make_entity_node('Bob')
    node_c = _make_entity_node('Charlie')

    edge_a = _make_entity_edge(node_a.uuid, node_b.uuid)
    edge_c = _make_entity_edge(node_b.uuid, node_c.uuid)

    episode_tuples = [
        (ep0, []),
        (ep1, []),
        (ep2, []),
    ]

    async def mock_extract_nodes(clients, episode, previous_episodes, **kwargs):
        if episode.name == 'ep0':
            return [node_a]
        if episode.name == 'ep1':
            return [node_b]
        return [node_c]

    async def mock_extract_edges(clients, episode, nodes, previous_episodes, **kwargs):
        if episode.name == 'ep1':
            raise RuntimeError('Edge extraction failed for ep1')
        if episode.name == 'ep0':
            return [edge_a]
        return [edge_c]

    with patch(
        'graphiti_core.utils.bulk_utils.extract_nodes', side_effect=mock_extract_nodes
    ), patch(
        'graphiti_core.utils.bulk_utils.extract_edges', side_effect=mock_extract_edges
    ):
        nodes_bulk, edges_bulk, failed_indices = await extract_nodes_and_edges_bulk(
            _make_clients_mock(),
            episode_tuples,
            edge_type_map={},
        )

    assert failed_indices == [1]
    # Only ep0 and ep2 succeed
    assert len(nodes_bulk) == 2
    assert len(edges_bulk) == 2
    assert nodes_bulk[0] == [node_a]
    assert nodes_bulk[1] == [node_c]
    assert edges_bulk[0] == [edge_a]
    assert edges_bulk[1] == [edge_c]


@pytest.mark.asyncio
async def test_all_episodes_fail_returns_empty_results():
    """All episodes fail node extraction. Assert empty results and all indices failed."""
    ep0 = _make_episode('ep0')
    ep1 = _make_episode('ep1')

    episode_tuples = [
        (ep0, []),
        (ep1, []),
    ]

    async def mock_extract_nodes(clients, episode, previous_episodes, **kwargs):
        raise RuntimeError(f'LLM extraction failed for {episode.name}')

    with patch(
        'graphiti_core.utils.bulk_utils.extract_nodes', side_effect=mock_extract_nodes
    ):
        nodes_bulk, edges_bulk, failed_indices = await extract_nodes_and_edges_bulk(
            _make_clients_mock(),
            episode_tuples,
            edge_type_map={},
        )

    assert failed_indices == [0, 1]
    assert len(nodes_bulk) == 0
    assert len(edges_bulk) == 0
