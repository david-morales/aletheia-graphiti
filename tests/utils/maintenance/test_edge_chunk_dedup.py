"""Tests for edge extraction cross-chunk deduplication fix.

The covering algorithm creates overlapping chunks so every node pair appears
in at least one chunk.  Before this fix, each pair was pre-assigned to the
*first* chunk containing it, and edges extracted in other (overlapping) chunks
were silently rejected.  The fix removes pair pre-assignment and instead
accepts edges from any chunk where both entity names are present, with
cross-chunk dedup by (source_name, target_name, normalized_fact).
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
from graphiti_core.utils.maintenance.edge_operations import extract_edges


def _make_node(name: str, uuid: str | None = None, labels: list[str] | None = None) -> EntityNode:
    return EntityNode(
        uuid=uuid or f'uuid-{name.lower().replace(" ", "-")}',
        name=name,
        group_id='g1',
        labels=labels or ['Entity'],
    )


def _make_episode(content: str = 'Episode text') -> EpisodicNode:
    return EpisodicNode(
        uuid='ep-1',
        name='ep',
        group_id='g1',
        source='message',
        source_description='test',
        content=content,
        valid_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Helpers to build LLM mock responses keyed by chunk contents
# ---------------------------------------------------------------------------


def _llm_response_sequence(responses: list[list[dict]]):
    """Return an AsyncMock side-effect that yields successive responses."""
    call_idx = 0

    async def _generate(prompt_messages, response_model=None, **kwargs):
        nonlocal call_idx
        idx = call_idx
        call_idx += 1
        if idx < len(responses):
            return {'edges': responses[idx]}
        return {'edges': []}

    return _generate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_from_overlapping_chunk_not_dropped():
    """An edge extracted in chunk 2 (where both nodes are present) must
    survive even if the pair also appears in chunk 1.

    Before the fix, pair (A,C) would be assigned to chunk 1 and any edge
    extracted in chunk 2 for that pair would be silently rejected.
    """
    node_a = _make_node('Alice')
    node_b = _make_node('Bob')
    node_c = _make_node('Charlie')

    episode = _make_episode('Alice knows Bob. Alice knows Charlie. Bob knows Charlie.')

    # Force two overlapping chunks: [Alice, Bob, Charlie] and [Alice, Charlie]
    # Pair (Alice, Charlie) exists in BOTH chunks.
    fake_chunks = [
        ([node_a, node_b], [0, 1]),
        ([node_a, node_c], [0, 2]),
    ]

    # Chunk 1 extracts Alice→Bob only (misses Alice→Charlie)
    # Chunk 2 extracts Alice→Charlie
    chunk_responses = [
        [  # Chunk 1: [Alice, Bob]
            {
                'source_entity_name': 'Alice',
                'target_entity_name': 'Bob',
                'relation_type': 'KNOWS',
                'fact': 'Alice knows Bob',
                'valid_at': None,
                'invalid_at': None,
            }
        ],
        [  # Chunk 2: [Alice, Charlie]
            {
                'source_entity_name': 'Alice',
                'target_entity_name': 'Charlie',
                'relation_type': 'KNOWS',
                'fact': 'Alice knows Charlie',
                'valid_at': None,
                'invalid_at': None,
            }
        ],
    ]

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=_llm_response_sequence(chunk_responses))

    clients = SimpleNamespace(llm_client=llm_client)

    with patch(
        'graphiti_core.utils.maintenance.edge_operations.generate_covering_chunks',
        return_value=fake_chunks,
    ):
        edges = await extract_edges(
            clients,
            episode,
            [node_a, node_b, node_c],
            [],
            edge_type_map={},
        )

    facts = {e.fact for e in edges}
    assert 'Alice knows Bob' in facts, f'Missing Alice→Bob edge; got {facts}'
    assert 'Alice knows Charlie' in facts, f'Missing Alice→Charlie edge; got {facts}'


@pytest.mark.asyncio
async def test_cross_chunk_dedup_removes_duplicates():
    """When two overlapping chunks both extract the same edge, only one copy
    should survive in the output."""
    node_a = _make_node('Alice')
    node_b = _make_node('Bob')

    episode = _make_episode('Alice knows Bob.')

    # Both chunks contain the same pair
    fake_chunks = [
        ([node_a, node_b], [0, 1]),
        ([node_a, node_b], [0, 1]),
    ]

    edge_payload = {
        'source_entity_name': 'Alice',
        'target_entity_name': 'Bob',
        'relation_type': 'KNOWS',
        'fact': 'Alice knows Bob',
        'valid_at': None,
        'invalid_at': None,
    }

    # Both chunks return the same edge
    chunk_responses = [
        [edge_payload],  # Chunk 1
        [edge_payload],  # Chunk 2
    ]

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=_llm_response_sequence(chunk_responses))

    clients = SimpleNamespace(llm_client=llm_client)

    with patch(
        'graphiti_core.utils.maintenance.edge_operations.generate_covering_chunks',
        return_value=fake_chunks,
    ):
        edges = await extract_edges(
            clients,
            episode,
            [node_a, node_b],
            [],
            edge_type_map={},
        )

    # Should be deduplicated to exactly one edge
    assert len(edges) == 1
    assert edges[0].fact == 'Alice knows Bob'


@pytest.mark.asyncio
async def test_cross_chunk_dedup_keeps_different_facts():
    """Different facts between the same pair should both survive."""
    node_a = _make_node('Alice')
    node_b = _make_node('Bob')

    episode = _make_episode('Alice knows Bob. Alice works with Bob.')

    fake_chunks = [
        ([node_a, node_b], [0, 1]),
        ([node_a, node_b], [0, 1]),
    ]

    two_facts = [
        {
            'source_entity_name': 'Alice',
            'target_entity_name': 'Bob',
            'relation_type': 'KNOWS',
            'fact': 'Alice knows Bob',
            'valid_at': None,
            'invalid_at': None,
        },
        {
            'source_entity_name': 'Alice',
            'target_entity_name': 'Bob',
            'relation_type': 'WORKS_WITH',
            'fact': 'Alice works with Bob',
            'valid_at': None,
            'invalid_at': None,
        },
    ]

    # Both chunks return the same two facts
    chunk_responses = [two_facts, two_facts]

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=_llm_response_sequence(chunk_responses))

    clients = SimpleNamespace(llm_client=llm_client)

    with patch(
        'graphiti_core.utils.maintenance.edge_operations.generate_covering_chunks',
        return_value=fake_chunks,
    ):
        edges = await extract_edges(
            clients,
            episode,
            [node_a, node_b],
            [],
            edge_type_map={},
        )

    # Both chunks return the same two facts → dedup keeps 2
    facts = {e.fact for e in edges}
    assert len(facts) == 2
    assert 'Alice knows Bob' in facts
    assert 'Alice works with Bob' in facts


@pytest.mark.asyncio
async def test_invalid_entity_name_still_rejected():
    """Edges referencing entity names not in the chunk should still be dropped."""
    node_a = _make_node('Alice')
    node_b = _make_node('Bob')

    episode = _make_episode('Alice knows someone.')

    fake_chunks = [
        ([node_a, node_b], [0, 1]),
    ]

    chunk_responses = [
        [
            {
                'source_entity_name': 'Alice',
                'target_entity_name': 'Unknown Person',  # Not in chunk
                'relation_type': 'KNOWS',
                'fact': 'Alice knows Unknown Person',
                'valid_at': None,
                'invalid_at': None,
            }
        ],
    ]

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=_llm_response_sequence(chunk_responses))

    clients = SimpleNamespace(llm_client=llm_client)

    with patch(
        'graphiti_core.utils.maintenance.edge_operations.generate_covering_chunks',
        return_value=fake_chunks,
    ):
        edges = await extract_edges(
            clients,
            episode,
            [node_a, node_b],
            [],
            edge_type_map={},
        )

    assert len(edges) == 0


@pytest.mark.asyncio
async def test_single_chunk_no_regression():
    """When there's only one chunk (few nodes), behaviour is unchanged."""
    node_a = _make_node('Alice')
    node_b = _make_node('Bob')

    episode = _make_episode('Alice knows Bob.')

    fake_chunks = [
        ([node_a, node_b], [0, 1]),
    ]

    chunk_responses = [
        [
            {
                'source_entity_name': 'Alice',
                'target_entity_name': 'Bob',
                'relation_type': 'KNOWS',
                'fact': 'Alice knows Bob',
                'valid_at': None,
                'invalid_at': None,
            }
        ],
    ]

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=_llm_response_sequence(chunk_responses))

    clients = SimpleNamespace(llm_client=llm_client)

    with patch(
        'graphiti_core.utils.maintenance.edge_operations.generate_covering_chunks',
        return_value=fake_chunks,
    ):
        edges = await extract_edges(
            clients,
            episode,
            [node_a, node_b],
            [],
            edge_type_map={},
        )

    assert len(edges) == 1
    assert edges[0].fact == 'Alice knows Bob'
