"""Integration tests for the validator hook inside resolve_extracted_edges.

These tests call the REAL resolve_extracted_edges function with all external
dependencies mocked (driver, embedder, search, LLM). The purpose is to verify
that when a validator drops edges, the downstream parallel lists
(valid_edges_list, related_edges_lists, edge_invalidation_candidates,
edge_types_lst) all stay aligned with extracted_edges, so the
zip(..., strict=True) calls don't raise ValueError.

The unit tests in test_edge_validator_hook.py cover _apply_edge_validators
in isolation. These integration tests catch regressions in the PLACEMENT of
the hook — specifically, the bug where the hook ran AFTER parallel lists were
built, causing a length mismatch (see roadmap S2.2).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.search.search_config import SearchResults
from graphiti_core.validation import EdgeDecision, ValidationContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_graphiti_clients(edge_validators=None):
    """Build a GraphitiClients with mocked internals."""
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.driver import GraphDriver
    from graphiti_core.embedder import EmbedderClient
    from graphiti_core.llm_client import LLMClient
    from graphiti_core.tracer import Tracer

    mock_driver = MagicMock(spec=GraphDriver)
    mock_driver.provider = "falkordb"
    return GraphitiClients(
        driver=mock_driver,
        llm_client=MagicMock(spec=LLMClient),
        embedder=MagicMock(spec=EmbedderClient),
        cross_encoder=MagicMock(spec=CrossEncoderClient),
        tracer=MagicMock(spec=Tracer),
        edge_validators=edge_validators or [],
    )


def _entity_node(uuid: str, label: str, group_id: str = "test") -> EntityNode:
    """Build a minimal real EntityNode."""
    return EntityNode(
        uuid=uuid,
        name=f"{label}-{uuid[:8]}",
        labels=[label, "Entity"],
        group_id=group_id,
        created_at=datetime.now(timezone.utc),
        summary="test node",
    )


def _entity_edge(
    name: str, source_uuid: str, target_uuid: str, group_id: str = "test"
) -> EntityEdge:
    """Build a minimal real EntityEdge."""
    return EntityEdge(
        uuid=str(uuid4()),
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        name=name,
        group_id=group_id,
        fact=f"{name} test fact",
        episodes=["ep-1"],
        created_at=datetime.now(timezone.utc),
    )


def _episodic_node(group_id: str = "test") -> EpisodicNode:
    """Build a minimal real EpisodicNode."""
    return EpisodicNode(
        uuid="ep-1",
        group_id=group_id,
        name="test episode",
        source_description="test",
        content="test content",
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
        source=EpisodeType.message,
    )


# ---------------------------------------------------------------------------
# Validators for testing
# ---------------------------------------------------------------------------

class _KeepAllValidator:
    name = "keep-all"

    def validate_edge(self, edge, source_node, target_node, context):
        return EdgeDecision(action="keep")


class _DropAllValidator:
    name = "drop-all"

    def validate_edge(self, edge, source_node, target_node, context):
        return EdgeDecision(action="drop", reason="test drop")


class _DropFirstValidator:
    """Drops the first edge it sees, keeps the rest."""
    name = "drop-first"

    def __init__(self):
        self._seen = 0

    def validate_edge(self, edge, source_node, target_node, context):
        self._seen += 1
        if self._seen == 1:
            return EdgeDecision(action="drop", reason="first edge dropped")
        return EdgeDecision(action="keep")


class _DropAllButLastValidator:
    """Drops all edges except the last one."""
    name = "drop-all-but-last"

    def __init__(self, total: int):
        self._total = total
        self._seen = 0

    def validate_edge(self, edge, source_node, target_node, context):
        self._seen += 1
        if self._seen < self._total:
            return EdgeDecision(action="drop", reason="not the last")
        return EdgeDecision(action="keep")


class _BrokenValidator:
    name = "broken"

    def validate_edge(self, edge, source_node, target_node, context):
        raise RuntimeError("validator crash")


# ---------------------------------------------------------------------------
# Shared mock setup
# ---------------------------------------------------------------------------

_MODULE = "graphiti_core.utils.maintenance.edge_operations"


async def _mock_resolve_extracted_edge(
    llm_client, extracted_edge, related_edges, existing_edges, episode,
    edge_type_candidates=None,
):
    """Stub for resolve_extracted_edge: returns the edge unchanged, no invalidations."""
    return (extracted_edge, [], [])


@pytest.fixture
def three_edges_and_nodes():
    """Three edges between three node pairs."""
    nodes = [
        _entity_node("n1", "Persona"),
        _entity_node("n2", "Persona"),
        _entity_node("n3", "Detencion"),
        _entity_node("n4", "Investigacion"),
        _entity_node("n5", "Identificacion"),
        _entity_node("n6", "Persona"),
    ]
    edges = [
        _entity_edge("ES_DETENIDO", "n1", "n3"),
        _entity_edge("ES_INVESTIGADO", "n2", "n4"),
        _entity_edge("ES_IDENTIFICADO", "n6", "n5"),
    ]
    return edges, nodes


@pytest.fixture
def episode():
    return _episodic_node()


@pytest.fixture
def edge_type_map():
    """Minimal edge_type_map that allows all test edge types."""
    return {
        ("Persona", "Detencion"): ["ES_DETENIDO"],
        ("Persona", "Investigacion"): ["ES_INVESTIGADO"],
        ("Persona", "Identificacion"): ["ES_IDENTIFICADO"],
        ("Entity", "Entity"): ["ES_DETENIDO", "ES_INVESTIGADO", "ES_IDENTIFICADO"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(f"{_MODULE}.resolve_extracted_edge", side_effect=_mock_resolve_extracted_edge)
@patch(f"{_MODULE}.search", new_callable=AsyncMock, return_value=SearchResults())
@patch(f"{_MODULE}.create_entity_edge_embeddings", new_callable=AsyncMock)
@patch.object(EntityEdge, "get_between_nodes", new_callable=AsyncMock, return_value=[])
@patch.object(EntityNode, "get_by_uuids", new_callable=AsyncMock, return_value=[])
async def test_no_validators_completes_normally(
    _mock_get_by_uuids,
    _mock_get_between,
    _mock_embed,
    _mock_search,
    _mock_resolve,
    three_edges_and_nodes,
    episode,
    edge_type_map,
):
    """With no validators, resolve_extracted_edges completes and returns all edges."""
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edges

    edges, nodes = three_edges_and_nodes
    clients = _mock_graphiti_clients(edge_validators=[])

    resolved, invalidated, new = await resolve_extracted_edges(
        clients, edges, episode, nodes, edge_types={}, edge_type_map=edge_type_map,
    )
    assert len(resolved) == 3
    assert len(invalidated) == 0


@pytest.mark.asyncio
@patch(f"{_MODULE}.resolve_extracted_edge", side_effect=_mock_resolve_extracted_edge)
@patch(f"{_MODULE}.search", new_callable=AsyncMock, return_value=SearchResults())
@patch(f"{_MODULE}.create_entity_edge_embeddings", new_callable=AsyncMock)
@patch.object(EntityEdge, "get_between_nodes", new_callable=AsyncMock, return_value=[])
@patch.object(EntityNode, "get_by_uuids", new_callable=AsyncMock, return_value=[])
async def test_keep_all_validator_passes_all_through(
    _mock_get_by_uuids,
    _mock_get_between,
    _mock_embed,
    _mock_search,
    _mock_resolve,
    three_edges_and_nodes,
    episode,
    edge_type_map,
):
    """A keep-all validator doesn't change the outcome."""
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edges

    edges, nodes = three_edges_and_nodes
    clients = _mock_graphiti_clients(edge_validators=[_KeepAllValidator()])

    resolved, invalidated, new = await resolve_extracted_edges(
        clients, edges, episode, nodes, edge_types={}, edge_type_map=edge_type_map,
    )
    assert len(resolved) == 3


@pytest.mark.asyncio
@patch(f"{_MODULE}.resolve_extracted_edge", side_effect=_mock_resolve_extracted_edge)
@patch(f"{_MODULE}.search", new_callable=AsyncMock, return_value=SearchResults())
@patch(f"{_MODULE}.create_entity_edge_embeddings", new_callable=AsyncMock)
@patch.object(EntityEdge, "get_between_nodes", new_callable=AsyncMock, return_value=[])
@patch.object(EntityNode, "get_by_uuids", new_callable=AsyncMock, return_value=[])
async def test_drop_one_no_zip_error(
    _mock_get_by_uuids,
    _mock_get_between,
    _mock_embed,
    _mock_search,
    _mock_resolve,
    three_edges_and_nodes,
    episode,
    edge_type_map,
):
    """Dropping one edge doesn't cause zip(strict=True) length mismatch."""
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edges

    edges, nodes = three_edges_and_nodes
    clients = _mock_graphiti_clients(edge_validators=[_DropFirstValidator()])

    resolved, invalidated, new = await resolve_extracted_edges(
        clients, edges, episode, nodes, edge_types={}, edge_type_map=edge_type_map,
    )
    # 3 edges in, 1 dropped -> 2 resolved
    assert len(resolved) == 2


@pytest.mark.asyncio
@patch(f"{_MODULE}.resolve_extracted_edge", side_effect=_mock_resolve_extracted_edge)
@patch(f"{_MODULE}.search", new_callable=AsyncMock, return_value=SearchResults())
@patch(f"{_MODULE}.create_entity_edge_embeddings", new_callable=AsyncMock)
@patch.object(EntityEdge, "get_between_nodes", new_callable=AsyncMock, return_value=[])
@patch.object(EntityNode, "get_by_uuids", new_callable=AsyncMock, return_value=[])
async def test_drop_all_but_one_no_zip_error(
    _mock_get_by_uuids,
    _mock_get_between,
    _mock_embed,
    _mock_search,
    _mock_resolve,
    three_edges_and_nodes,
    episode,
    edge_type_map,
):
    """Dropping all but one edge still produces a valid result."""
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edges

    edges, nodes = three_edges_and_nodes
    clients = _mock_graphiti_clients(edge_validators=[_DropAllButLastValidator(total=3)])

    resolved, invalidated, new = await resolve_extracted_edges(
        clients, edges, episode, nodes, edge_types={}, edge_type_map=edge_type_map,
    )
    assert len(resolved) == 1


@pytest.mark.asyncio
@patch(f"{_MODULE}.resolve_extracted_edge", side_effect=_mock_resolve_extracted_edge)
@patch(f"{_MODULE}.search", new_callable=AsyncMock, return_value=SearchResults())
@patch(f"{_MODULE}.create_entity_edge_embeddings", new_callable=AsyncMock)
@patch.object(EntityEdge, "get_between_nodes", new_callable=AsyncMock, return_value=[])
@patch.object(EntityNode, "get_by_uuids", new_callable=AsyncMock, return_value=[])
async def test_drop_all_returns_empty(
    _mock_get_by_uuids,
    _mock_get_between,
    _mock_embed,
    _mock_search,
    _mock_resolve,
    three_edges_and_nodes,
    episode,
    edge_type_map,
):
    """Dropping all edges returns an empty result without errors."""
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edges

    edges, nodes = three_edges_and_nodes
    clients = _mock_graphiti_clients(edge_validators=[_DropAllValidator()])

    resolved, invalidated, new = await resolve_extracted_edges(
        clients, edges, episode, nodes, edge_types={}, edge_type_map=edge_type_map,
    )
    assert len(resolved) == 0
    assert len(invalidated) == 0
    assert len(new) == 0


@pytest.mark.asyncio
@patch(f"{_MODULE}.resolve_extracted_edge", side_effect=_mock_resolve_extracted_edge)
@patch(f"{_MODULE}.search", new_callable=AsyncMock, return_value=SearchResults())
@patch(f"{_MODULE}.create_entity_edge_embeddings", new_callable=AsyncMock)
@patch.object(EntityEdge, "get_between_nodes", new_callable=AsyncMock, return_value=[])
@patch.object(EntityNode, "get_by_uuids", new_callable=AsyncMock, return_value=[])
async def test_broken_validator_fails_open(
    _mock_get_by_uuids,
    _mock_get_between,
    _mock_embed,
    _mock_search,
    _mock_resolve,
    three_edges_and_nodes,
    episode,
    edge_type_map,
):
    """A validator that raises doesn't crash ingestion -- edges are kept (fail-open)."""
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edges

    edges, nodes = three_edges_and_nodes
    clients = _mock_graphiti_clients(edge_validators=[_BrokenValidator()])

    resolved, invalidated, new = await resolve_extracted_edges(
        clients, edges, episode, nodes, edge_types={}, edge_type_map=edge_type_map,
    )
    # All 3 edges kept despite validator crash
    assert len(resolved) == 3
