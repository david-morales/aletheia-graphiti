"""Tests for add_memory sync parameter."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.response_types import EpisodeAddedResponse
from graphiti_mcp_server import add_memory


class FakeNode:
    def __init__(self, uuid: str):
        self.uuid = uuid


class FakeEdge:
    def __init__(self, uuid: str):
        self.uuid = uuid


class FakeEpisode:
    uuid = "ep-1"


class FakeResults:
    def __init__(self):
        self.episode = FakeEpisode()
        self.episodic_edges = []
        self.nodes = [FakeNode("node-1"), FakeNode("node-2")]
        self.edges = [FakeEdge("edge-1")]
        self.communities = []
        self.community_edges = []


def make_mock_services(group_id: str = 'test-group'):
    """Create mock graphiti_service, queue_service, and config objects."""
    mock_client = AsyncMock()
    mock_graphiti_service = AsyncMock()
    mock_graphiti_service.get_client = AsyncMock(return_value=mock_client)
    mock_graphiti_service.entity_types = {}
    mock_graphiti_service._schema_dirty = False

    mock_queue_service = AsyncMock()

    mock_config = MagicMock()
    mock_config.graphiti.group_id = group_id

    return mock_graphiti_service, mock_queue_service, mock_config, mock_client


@pytest.mark.asyncio
async def test_add_memory_sync_returns_uuids():
    """When sync=True, add_memory should bypass queue and return UUIDs."""
    svc, queue, cfg, client = make_mock_services(group_id='test-group')
    client.add_episode = AsyncMock(return_value=FakeResults())

    with (
        patch('graphiti_mcp_server.graphiti_service', svc),
        patch('graphiti_mcp_server.queue_service', queue),
        patch('graphiti_mcp_server.config', cfg, create=True),
    ):
        result = await add_memory(
            name="test-episode",
            episode_body="Some content",
            source="text",
            sync=True,
        )

    assert result['node_uuids'] == ['node-1', 'node-2']
    assert result['edge_uuids'] == ['edge-1']
    assert 'message' in result
    client.add_episode.assert_called_once()
    # Queue should NOT have been called
    queue.add_episode.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_without_sync_uses_queue():
    """Default (no sync) should still use the queue."""
    svc, queue, cfg, client = make_mock_services(group_id='test-group')
    queue.add_episode = AsyncMock(return_value=0)

    with (
        patch('graphiti_mcp_server.graphiti_service', svc),
        patch('graphiti_mcp_server.queue_service', queue),
        patch('graphiti_mcp_server.config', cfg, create=True),
    ):
        result = await add_memory(
            name="test-episode",
            episode_body="Some content",
        )

    assert 'message' in result
    queue.add_episode.assert_called_once()
