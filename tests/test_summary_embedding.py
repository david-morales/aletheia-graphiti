"""Tests for summary embedding on EntityNode."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import NodeSearchMethod


def test_summary_embedding_field_exists():
    """EntityNode can be created with a summary_embedding value."""
    node = EntityNode(
        name='Test Node',
        group_id='test-group',
        summary='A summary of surrounding edges',
        summary_embedding=[0.1, 0.2, 0.3],
    )
    assert node.summary_embedding == [0.1, 0.2, 0.3]


def test_summary_embedding_default_none():
    """summary_embedding defaults to None when not provided."""
    node = EntityNode(
        name='Test Node',
        group_id='test-group',
    )
    assert node.summary_embedding is None


def test_summary_similarity_in_search_methods():
    """NodeSearchMethod has a summary_similarity enum value."""
    assert hasattr(NodeSearchMethod, 'summary_similarity')
    assert NodeSearchMethod.summary_similarity.value == 'summary_similarity'


@pytest.mark.asyncio
async def test_generate_summary_embedding():
    """generate_summary_embedding calls the embedder and sets the field."""
    node = EntityNode(
        name='Test Node',
        group_id='test-group',
        summary='This node is connected to many things',
    )

    mock_embedder = AsyncMock()
    mock_embedder.create.return_value = [0.4, 0.5, 0.6]

    result = await node.generate_summary_embedding(mock_embedder)

    mock_embedder.create.assert_called_once_with(
        input_data=['This node is connected to many things']
    )
    assert node.summary_embedding == [0.4, 0.5, 0.6]
    assert result == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_generate_summary_embedding_replaces_newlines():
    """generate_summary_embedding replaces newlines in summary text."""
    node = EntityNode(
        name='Test Node',
        group_id='test-group',
        summary='Line one\nLine two\nLine three',
    )

    mock_embedder = AsyncMock()
    mock_embedder.create.return_value = [0.1, 0.2]

    await node.generate_summary_embedding(mock_embedder)

    mock_embedder.create.assert_called_once_with(input_data=['Line one Line two Line three'])


@pytest.mark.asyncio
async def test_save_includes_summary_embedding():
    """The save method passes summary_embedding in the entity_data dict."""
    node = EntityNode(
        name='Test Node',
        group_id='test-group',
        summary='A test summary',
        summary_embedding=[0.7, 0.8, 0.9],
    )

    mock_driver = MagicMock()
    mock_driver.graph_operations_interface = None
    mock_driver.provider = MagicMock()
    # Make provider not match KUZU so we go through the else branch
    mock_driver.provider.__eq__ = lambda self, other: False
    mock_driver.provider.name = 'FALKORDB'

    # Mock execute_query to capture the call
    mock_driver.execute_query = AsyncMock(return_value=([], None, None))

    # We need to mock get_entity_node_save_query
    with patch('graphiti_core.nodes.get_entity_node_save_query', return_value='MOCK QUERY'):
        await node.save(mock_driver)

    # Verify execute_query was called with entity_data containing summary_embedding
    call_args = mock_driver.execute_query.call_args
    entity_data = call_args.kwargs.get('entity_data', call_args[1].get('entity_data'))
    assert entity_data is not None, 'entity_data not found in call args'
    assert 'summary_embedding' in entity_data
    assert entity_data['summary_embedding'] == [0.7, 0.8, 0.9]
