"""Test that custom edge attributes are extracted even on the first-episode fast path
(no related or existing edges). Upstream #1242."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode, EpisodeType


class OccurredAtAttributes(BaseModel):
    location: str = ''
    severity: str = ''


@pytest.mark.asyncio
async def test_resolve_extracted_edge_extracts_attributes_on_empty_graph():
    """When no related or existing edges, custom attributes should still be extracted."""
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge

    now = datetime.now(timezone.utc)

    extracted_edge = EntityEdge(
        source_node_uuid='src-1',
        target_node_uuid='tgt-1',
        name='OCCURRED_AT',
        fact='Event occurred at Location X',
        group_id='test',
        created_at=now,
    )

    episode = EpisodicNode(
        name='ep1',
        source=EpisodeType.text,
        source_description='test',
        content='Event occurred at Location X with high severity',
        valid_at=now,
        entity_edges=[],
        group_id='test',
    )

    mock_llm = AsyncMock()
    mock_llm.generate_response.return_value = OccurredAtAttributes(
        location='Location X', severity='high'
    )

    edge_type_candidates = {'OCCURRED_AT': OccurredAtAttributes}

    resolved, duplicates, invalidated = await resolve_extracted_edge(
        mock_llm,
        extracted_edge,
        [],  # no related edges
        [],  # no existing edges
        episode,
        edge_type_candidates=edge_type_candidates,
    )

    # Attributes should have been extracted via LLM
    mock_llm.generate_response.assert_called_once()
    assert resolved.attributes == OccurredAtAttributes(location='Location X', severity='high')
    assert duplicates == []
    assert invalidated == []


@pytest.mark.asyncio
async def test_resolve_extracted_edge_skips_attributes_when_no_model():
    """When no edge_type_candidates match, skip attribute extraction on fast path."""
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge

    now = datetime.now(timezone.utc)

    extracted_edge = EntityEdge(
        source_node_uuid='src-1',
        target_node_uuid='tgt-1',
        name='UNKNOWN_TYPE',
        fact='Something happened',
        group_id='test',
        created_at=now,
    )

    episode = EpisodicNode(
        name='ep1',
        source=EpisodeType.text,
        source_description='test',
        content='Something happened',
        valid_at=now,
        entity_edges=[],
        group_id='test',
    )

    mock_llm = AsyncMock()

    _resolved, duplicates, invalidated = await resolve_extracted_edge(
        mock_llm,
        extracted_edge,
        [],
        [],
        episode,
        edge_type_candidates={'OTHER_TYPE': OccurredAtAttributes},
    )

    # No LLM call — no matching edge type
    mock_llm.generate_response.assert_not_called()
    assert duplicates == []
    assert invalidated == []
