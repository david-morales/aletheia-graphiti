"""Test that edges with phantom UUID references are dropped before resolution.
Upstream #1267 (partial)."""

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType


@pytest.mark.asyncio
async def test_phantom_uuid_edges_are_dropped(caplog):
    """Edges referencing nodes not in entity map should be filtered out."""
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edges

    now = datetime.now(timezone.utc)

    entity_a = EntityNode(uuid='a', name='Alice', group_id='test', labels=['Person'], created_at=now)
    entity_b = EntityNode(uuid='b', name='Bob', group_id='test', labels=['Person'], created_at=now)

    valid_edge = EntityEdge(
        source_node_uuid='a', target_node_uuid='b',
        name='KNOWS', fact='Alice knows Bob', group_id='test', created_at=now,
    )
    phantom_edge = EntityEdge(
        source_node_uuid='a', target_node_uuid='phantom-doesnt-exist',
        name='WORKS_WITH', fact='Alice works with ghost', group_id='test', created_at=now,
    )

    episode = EpisodicNode(
        name='ep1', source=EpisodeType.text, source_description='test',
        content='Alice knows Bob', valid_at=now, entity_edges=[], group_id='test',
    )

    mock_llm = AsyncMock()
    mock_llm.generate_response.return_value = {'duplicate_facts': [], 'invalidate_facts': []}

    mock_embedder = AsyncMock()
    # create_batch returns a list of embeddings, one per input text
    mock_embedder.create_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1024 for _ in texts])

    # Mock driver that returns empty results for all queries
    # (phantom UUID truly doesn't exist in the database)
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=([], None, None))
    mock_driver.provider = 'falkordb'

    clients = GraphitiClients.model_construct(
        llm_client=mock_llm,
        driver=mock_driver,
        embedder=mock_embedder,
        cross_encoder=MagicMock(),
        tracer=MagicMock(),
    )

    with caplog.at_level(logging.WARNING):
        resolved, invalidated, new_edges = await resolve_extracted_edges(
            clients,
            [valid_edge, phantom_edge],
            episode,
            [entity_a, entity_b],
            {},
            {},
        )

    # Check that the phantom edge warning was logged
    phantom_warnings = [msg for msg in caplog.messages if 'phantom' in msg.lower() or 'Dropping' in msg]
    assert len(phantom_warnings) > 0, f'Expected phantom UUID warning, got: {caplog.messages}'
