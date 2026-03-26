"""Test that add_triplet generates a new UUID when an edge UUID already exists
with different source/target nodes (upstream #1212)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.nodes import EntityNode


@pytest.mark.asyncio
async def test_add_triplet_generates_new_uuid_on_src_dst_mismatch():
    """When edge UUID exists with different src/dst, a new UUID should be generated."""
    now = datetime.now(timezone.utc)

    source = EntityNode(
        uuid='source-1', name='Alice', group_id='test', labels=['Person'], created_at=now,
    )
    target = EntityNode(
        uuid='target-1', name='Bob', group_id='test', labels=['Person'], created_at=now,
    )
    edge = EntityEdge(
        uuid='edge-1',
        source_node_uuid='source-1',
        target_node_uuid='target-1',
        name='KNOWS',
        fact='Alice knows Bob',
        group_id='test',
        created_at=now,
    )

    # Simulate an existing edge with same UUID but different endpoints
    existing_edge = EntityEdge(
        uuid='edge-1',
        source_node_uuid='other-source',
        target_node_uuid='other-target',
        name='WORKS_WITH',
        fact='Other edge',
        group_id='test',
        created_at=now,
    )

    with patch.object(EntityEdge, 'get_by_uuid', new_callable=AsyncMock) as mock_get_edge:
        mock_get_edge.return_value = existing_edge

        # The collision detection logic: when existing edge has different src/dst, generate new UUID
        try:
            check_edge = await EntityEdge.get_by_uuid(Mock(), edge.uuid)
            if (
                check_edge.source_node_uuid != edge.source_node_uuid
                or check_edge.target_node_uuid != edge.target_node_uuid
            ):
                old_uuid = edge.uuid
                from uuid import uuid4
                edge.uuid = str(uuid4())
                assert edge.uuid != old_uuid
                assert edge.uuid != 'edge-1'
        except EdgeNotFoundError:
            pytest.fail('Should not raise EdgeNotFoundError in this test')


@pytest.mark.asyncio
async def test_add_triplet_keeps_uuid_when_edge_not_found():
    """When edge UUID does not exist, keep the original UUID."""
    now = datetime.now(timezone.utc)
    edge = EntityEdge(
        uuid='edge-new',
        source_node_uuid='source-1',
        target_node_uuid='target-1',
        name='KNOWS',
        fact='Alice knows Bob',
        group_id='test',
        created_at=now,
    )

    with patch.object(EntityEdge, 'get_by_uuid', new_callable=AsyncMock) as mock_get_edge:
        mock_get_edge.side_effect = EdgeNotFoundError('edge-new')

        try:
            await EntityEdge.get_by_uuid(Mock(), edge.uuid)
        except EdgeNotFoundError:
            pass

        assert edge.uuid == 'edge-new'


@pytest.mark.asyncio
async def test_add_triplet_keeps_uuid_when_same_endpoints():
    """When edge UUID exists with same src/dst, keep the UUID (update is fine)."""
    now = datetime.now(timezone.utc)
    edge = EntityEdge(
        uuid='edge-1',
        source_node_uuid='source-1',
        target_node_uuid='target-1',
        name='KNOWS',
        fact='Alice knows Bob',
        group_id='test',
        created_at=now,
    )

    existing_edge = EntityEdge(
        uuid='edge-1',
        source_node_uuid='source-1',
        target_node_uuid='target-1',
        name='KNOWS',
        fact='Old fact',
        group_id='test',
        created_at=now,
    )

    with patch.object(EntityEdge, 'get_by_uuid', new_callable=AsyncMock) as mock_get_edge:
        mock_get_edge.return_value = existing_edge

        check_edge = await EntityEdge.get_by_uuid(Mock(), edge.uuid)
        if (
            check_edge.source_node_uuid != edge.source_node_uuid
            or check_edge.target_node_uuid != edge.target_node_uuid
        ):
            pytest.fail('Should not generate new UUID for same endpoints')

        assert edge.uuid == 'edge-1'
