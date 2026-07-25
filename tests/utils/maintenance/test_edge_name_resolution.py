"""Tests for edge extraction entity-name resolution (fork 64898ee / 53597d5).

graphiti-core v0.29.2 (#1224/#1432) replaced the fork's chunked edge extraction
with a single-pass, multi-episode extractor. The fork's name-resolution recovery
is preserved via ``_resolve_entity_name`` inside ``extract_edges``: LLM-returned
entity names are matched against the node list tolerating case differences and
containment, so a valid edge is not silently dropped when the LLM does not echo
a name verbatim; genuinely unresolvable names are still rejected.

(Within-episode duplicate-edge dedup now lives in ``resolve_extracted_edges``'s
fast path — see ``test_resolve_extracted_edges_fast_path_deduplication`` — so the
old cross-chunk dedup tests were removed with the chunking architecture.)
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.nodes import EntityNode, EpisodicNode
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


def _llm_returning(edges: list[dict]):
    """A single-pass LLM mock: extract_edges calls generate_response once."""
    return AsyncMock(return_value={'edges': edges})


async def _run(nodes: list[EntityNode], edges: list[dict]):
    llm_client = MagicMock()
    llm_client.generate_response = _llm_returning(edges)
    clients = SimpleNamespace(llm_client=llm_client)
    return await extract_edges(
        clients,
        _make_episode(),
        nodes,
        [],
        edge_type_map={},
    )


@pytest.mark.asyncio
async def test_case_insensitive_name_resolution():
    """LLM returning lower-cased names must resolve to the canonical nodes."""
    alice = _make_node('Alice')
    bob = _make_node('Bob')
    edges = await _run(
        [alice, bob],
        [
            {
                'source_entity_name': 'alice',
                'target_entity_name': 'bob',
                'relation_type': 'KNOWS',
                'fact': 'Alice knows Bob',
            }
        ],
    )
    assert len(edges) == 1
    assert edges[0].source_node_uuid == alice.uuid
    assert edges[0].target_node_uuid == bob.uuid


@pytest.mark.asyncio
async def test_containment_name_resolution_abbreviated():
    """LLM returning an abbreviated name (contained in the canonical name)
    must resolve to the full node (e.g. 'RUSAL' -> 'United Company RUSAL')."""
    rusal = _make_node('United Company RUSAL')
    alice = _make_node('Alice')
    edges = await _run(
        [rusal, alice],
        [
            {
                'source_entity_name': 'Alice',
                'target_entity_name': 'RUSAL',
                'relation_type': 'WORKS_AT',
                'fact': 'Alice works at RUSAL',
            }
        ],
    )
    assert len(edges) == 1
    assert edges[0].target_node_uuid == rusal.uuid


@pytest.mark.asyncio
async def test_containment_name_resolution_extended():
    """LLM returning an extended name (containing the canonical name) must
    resolve to the shorter node (e.g. 'United Company RUSAL' -> 'RUSAL')."""
    rusal = _make_node('RUSAL')
    alice = _make_node('Alice')
    edges = await _run(
        [rusal, alice],
        [
            {
                'source_entity_name': 'Alice',
                'target_entity_name': 'United Company RUSAL',
                'relation_type': 'WORKS_AT',
                'fact': 'Alice works at RUSAL',
            }
        ],
    )
    assert len(edges) == 1
    assert edges[0].target_node_uuid == rusal.uuid


@pytest.mark.asyncio
async def test_hallucinated_name_still_rejected():
    """An edge whose entity name matches no node (and is not a containment
    match) must be dropped, not attached to an arbitrary node."""
    alice = _make_node('Alice')
    bob = _make_node('Bob')
    edges = await _run(
        [alice, bob],
        [
            {
                'source_entity_name': 'Zorro',
                'target_entity_name': 'Bob',
                'relation_type': 'KNOWS',
                'fact': 'Zorro knows Bob',
            }
        ],
    )
    assert edges == []
