"""Unit tests for MCP server tool functions.

Tests cover: resolve_search_config, search, explore_entity, get_episode_context,
build_communities, and add_memory.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flavours.falkordb import FalkorDbFlavour
from graphiti_mcp_server import (
    add_memory,
    build_communities,
    explore_entity,
    explore_ontology,
    get_episode_context,
    resolve_search_config,
    search,
    search_ontology,
)


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def make_mock_node(
    uuid: str = 'node-uuid-1',
    name: str = 'TestNode',
    labels: list[str] | None = None,
    summary: str = 'A test node summary',
    group_id: str = 'test-group',
    created_at: datetime | None = None,
    attributes: dict | None = None,
):
    """Create a mock EntityNode-like object."""
    node = MagicMock()
    node.uuid = uuid
    node.name = name
    node.labels = labels or ['Entity']
    node.summary = summary
    node.group_id = group_id
    node.created_at = created_at or datetime(2025, 1, 1, tzinfo=timezone.utc)
    node.attributes = attributes if attributes is not None else {'key': 'value'}
    return node


def make_mock_edge(
    uuid: str = 'edge-uuid-1',
    name: str = 'RELATES_TO',
    fact: str = 'A is related to B',
    source_node_uuid: str = 'node-uuid-1',
    target_node_uuid: str = 'node-uuid-2',
    group_id: str = 'test-group',
    created_at: datetime | None = None,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
):
    """Create a mock EntityEdge-like object."""
    edge = MagicMock()
    edge.uuid = uuid
    edge.name = name
    edge.fact = fact
    edge.source_node_uuid = source_node_uuid
    edge.target_node_uuid = target_node_uuid
    edge.group_id = group_id
    edge.created_at = created_at or datetime(2025, 1, 1, tzinfo=timezone.utc)
    edge.valid_at = valid_at
    edge.invalid_at = invalid_at
    return edge


def make_mock_community(
    uuid: str = 'community-uuid-1',
    name: str = 'TestCommunity',
    summary: str = 'A cluster of related entities',
    group_id: str = 'test-group',
):
    """Create a mock CommunityNode-like object."""
    community = MagicMock()
    community.uuid = uuid
    community.name = name
    community.summary = summary
    community.group_id = group_id
    return community


def make_mock_search_results(
    nodes: list | None = None,
    edges: list | None = None,
    communities: list | None = None,
):
    """Create a mock SearchResults-like object."""
    results = MagicMock()
    results.nodes = nodes if nodes is not None else []
    results.edges = edges if edges is not None else []
    results.communities = communities if communities is not None else []
    return results


def make_mock_services(group_id: str = 'test-group'):
    """Create mock graphiti_service, queue_service, and config objects.

    Returns (mock_graphiti_service, mock_queue_service, mock_config, mock_client).
    The mock_client is the object returned by graphiti_service.get_client().
    """
    mock_client = AsyncMock()
    mock_graphiti_service = AsyncMock()
    mock_graphiti_service.get_client = AsyncMock(return_value=mock_client)
    mock_graphiti_service.entity_types = None
    mock_graphiti_service.ontology_client = None
    # A REAL flavour: the census/ontology query TEXT is flavour-owned, so a mock
    # here never yields Cypher. On this AsyncMock service the failure is
    # especially quiet — `flavour.ontology_queries()` returns a COROUTINE, and
    # subscripting it raises inside the tool's own try/except, so the tool
    # answers `{'error': ...}` instead of crashing. Same hazard as
    # test_ontology_resilience.py and test_ontology_tiers.py.
    mock_graphiti_service.flavour = FalkorDbFlavour()
    # Default: no ontology configured — matches ontology_client=None semantics.
    # Tests that exercise the happy path must override this with AsyncMock(return_value=True).
    mock_graphiti_service._ensure_ontology_client = AsyncMock(return_value=False)

    mock_queue_service = AsyncMock()

    mock_config = MagicMock()
    mock_config.graphiti.group_id = group_id

    return mock_graphiti_service, mock_queue_service, mock_config, mock_client


# ---------------------------------------------------------------------------
# TestResolveSearchConfig — pure function tests
# ---------------------------------------------------------------------------

class TestResolveSearchConfig:
    """Tests for the resolve_search_config pure function."""

    def test_valid_combined_rrf(self):
        config = resolve_search_config('combined', 'rrf', 15)
        assert config.limit == 15
        # Should have both edge and node configs (combined search)
        assert config.edge_config is not None
        assert config.node_config is not None

    def test_valid_edges_node_distance(self):
        config = resolve_search_config('edges', 'node_distance', 20)
        assert config.limit == 20
        assert config.edge_config is not None
        # edges-only mode should not have node_config
        assert config.node_config is None

    def test_case_insensitive(self):
        config_lower = resolve_search_config('combined', 'rrf', 10)
        config_upper = resolve_search_config('Combined', 'RRF', 10)
        config_mixed = resolve_search_config('COMBINED', 'Rrf', 10)
        # All should resolve without error and produce configs with same limit
        assert config_lower.limit == 10
        assert config_upper.limit == 10
        assert config_mixed.limit == 10

    def test_invalid_combo_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid search_mode='nodes' \\+ reranker='nonexistent'"):
            resolve_search_config('nodes', 'nonexistent', 10)


# ---------------------------------------------------------------------------
# TestSearch
# ---------------------------------------------------------------------------

class TestSearch:
    """Tests for the search tool function."""

    @pytest.mark.asyncio
    async def test_service_not_initialized(self):
        with patch('graphiti_mcp_server.graphiti_service', None):
            result = await search(query='test query')
        assert 'error' in result
        assert 'not initialized' in result['error']

    @pytest.mark.asyncio
    async def test_basic_search_defaults(self):
        svc, queue, cfg, client = make_mock_services()
        node = make_mock_node()
        edge = make_mock_edge()
        client.search_ = AsyncMock(return_value=make_mock_search_results(
            nodes=[node], edges=[edge],
        ))

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await search(query='test query')

        assert 'error' not in result
        assert 'nodes' in result
        assert 'edges' in result
        assert len(result['nodes']) == 1
        assert len(result['edges']) == 1
        assert result['nodes'][0]['uuid'] == 'node-uuid-1'
        # execution_ms timing is included on the search result (ADR-019 diagnostics)
        assert 'execution_ms' in result
        assert isinstance(result['execution_ms'], float)

    @pytest.mark.asyncio
    async def test_invalid_recipe_returns_error(self):
        svc, queue, cfg, client = make_mock_services()

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await search(
                query='test', search_mode='nodes', reranker='nonexistent',
            )

        assert 'error' in result
        assert 'Invalid' in result['error']

    @pytest.mark.asyncio
    async def test_entity_types_filter(self):
        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            await search(query='test', entity_types=['Person', 'Organization'])

        call_kwargs = client.search_.call_args.kwargs
        search_filter = call_kwargs['search_filter']
        assert search_filter.node_labels == ['Person', 'Organization']

    @pytest.mark.asyncio
    async def test_edge_types_filter(self):
        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            await search(query='test', edge_types=['OWNERSHIP', 'SANCTION'])

        call_kwargs = client.search_.call_args.kwargs
        search_filter = call_kwargs['search_filter']
        assert search_filter.edge_types == ['OWNERSHIP', 'SANCTION']

    @pytest.mark.asyncio
    async def test_valid_at_temporal_filter(self):
        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            await search(query='test', valid_at='2025-06-15T00:00:00')

        call_kwargs = client.search_.call_args.kwargs
        search_filter = call_kwargs['search_filter']
        # valid_at and invalid_at should be populated (not None/empty)
        assert search_filter.valid_at is not None
        assert len(search_filter.valid_at) > 0
        assert search_filter.invalid_at is not None
        assert len(search_filter.invalid_at) > 0

    @pytest.mark.asyncio
    async def test_center_node_uuid_passed(self):
        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            await search(
                query='test',
                search_mode='edges',
                reranker='node_distance',
                center_node_uuid='center-uuid-123',
            )

        call_kwargs = client.search_.call_args.kwargs
        assert call_kwargs['center_node_uuid'] == 'center-uuid-123'

    @pytest.mark.asyncio
    async def test_group_ids_default(self):
        svc, queue, cfg, client = make_mock_services(group_id='default-group')
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            await search(query='test')

        call_kwargs = client.search_.call_args.kwargs
        assert call_kwargs['group_ids'] == ['default-group']

    @pytest.mark.asyncio
    async def test_group_ids_explicit(self):
        svc, queue, cfg, client = make_mock_services(group_id='default-group')
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            await search(query='test', group_ids=['graph-a', 'graph-b'])

        call_kwargs = client.search_.call_args.kwargs
        assert call_kwargs['group_ids'] == ['graph-a', 'graph-b']

    @pytest.mark.asyncio
    async def test_node_attributes_exclude_embeddings(self):
        node = make_mock_node(attributes={
            'key': 'value',
            'name_embedding': [0.1, 0.2, 0.3],
            'Description_Embedding': [0.4, 0.5],
        })
        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(return_value=make_mock_search_results(nodes=[node]))

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await search(query='test')

        attrs = result['nodes'][0]['attributes']
        assert 'key' in attrs
        assert attrs['key'] == 'value'
        # Anything with 'embedding' (case-insensitive) in the key should be excluded
        assert 'name_embedding' not in attrs
        assert 'Description_Embedding' not in attrs


# ---------------------------------------------------------------------------
# TestExploreNode
# ---------------------------------------------------------------------------

class TestExploreNode:
    """Tests for the explore_entity tool function."""

    @pytest.mark.asyncio
    async def test_service_not_initialized(self):
        with patch('graphiti_mcp_server.graphiti_service', None):
            result = await explore_entity(node_name='Test')
        assert 'error' in result
        assert 'not initialized' in result['error']

    @pytest.mark.asyncio
    async def test_neither_name_nor_uuid(self):
        svc, queue, cfg, client = make_mock_services()

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity()

        assert 'error' in result
        assert 'node_name or node_uuid' in result['error']

    @pytest.mark.asyncio
    async def test_name_resolution_found(self):
        svc, queue, cfg, client = make_mock_services()

        # First call: name resolution search returns a node
        resolved_node = make_mock_node(uuid='resolved-uuid', name='FoundEntity')
        resolve_results = make_mock_search_results(nodes=[resolved_node])

        # Second call: neighborhood search returns nodes and edges
        neighbor_node = make_mock_node(uuid='neighbor-uuid', name='Neighbor')
        neighbor_edge = make_mock_edge()
        explore_results = make_mock_search_results(
            nodes=[neighbor_node], edges=[neighbor_edge],
        )

        client.search_ = AsyncMock(side_effect=[resolve_results, explore_results])

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity(node_name='FoundEntity')

        assert 'error' not in result
        assert result['center_node'] is not None
        assert result['center_node']['uuid'] == 'resolved-uuid'
        assert len(result['nodes']) == 1
        assert len(result['edges']) == 1

        # Verify the second search used the resolved UUID
        second_call_kwargs = client.search_.call_args_list[1].kwargs
        assert second_call_kwargs['center_node_uuid'] == 'resolved-uuid'
        assert second_call_kwargs['bfs_origin_node_uuids'] == ['resolved-uuid']

    @pytest.mark.asyncio
    async def test_name_not_found(self):
        svc, queue, cfg, client = make_mock_services()

        # Name resolution returns no nodes
        client.search_ = AsyncMock(
            return_value=make_mock_search_results(nodes=[]),
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity(node_name='NonExistent')

        # Should return an ExploreResponse with empty results, not an error
        assert 'error' not in result
        assert 'No node found' in result['message']
        assert result['center_node'] is None
        assert result['nodes'] == []

    @pytest.mark.asyncio
    async def test_uuid_provided_directly(self):
        svc, queue, cfg, client = make_mock_services()
        target_node = make_mock_node(uuid='direct-uuid', name='Direct')
        client.search_ = AsyncMock(return_value=make_mock_search_results(
            nodes=[target_node],
        ))

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch(
                'graphiti_mcp_server.EntityNode.get_by_uuid',
                AsyncMock(return_value=target_node),
            ),
        ):
            result = await explore_entity(node_uuid='direct-uuid')

        assert 'error' not in result
        # Should only call search_ once (no name resolution needed)
        assert client.search_.call_count == 1
        call_kwargs = client.search_.call_args.kwargs
        assert call_kwargs['center_node_uuid'] == 'direct-uuid'
        assert call_kwargs['bfs_origin_node_uuids'] == ['direct-uuid']

    @pytest.mark.asyncio
    async def test_uuid_path_populates_center_node(self):
        """Regression: the uuid path returned `center_node: null`, always.

        `center_node` was only ever filled from the neighbourhood results, so a
        centre that is not among its own neighbours — or a traversal that comes
        back thin, as on the AGE flavour — produced a confidently null centre.
        The lookup is direct now, so the returned centre does not depend on what
        the traversal happens to find.
        """
        svc, queue, cfg, client = make_mock_services()
        centre = make_mock_node(uuid='hub-uuid', name='KHADIJA DAOUD', labels=['Entity', 'Persona'])
        # The neighbourhood comes back with an UNRELATED node and no edges —
        # exactly the shape the AGE connector returned for this hub.
        unrelated = make_mock_node(uuid='other-uuid', name='bolso')
        client.search_ = AsyncMock(return_value=make_mock_search_results(nodes=[unrelated]))

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch(
                'graphiti_mcp_server.EntityNode.get_by_uuid', AsyncMock(return_value=centre)
            ) as get_by_uuid,
        ):
            result = await explore_entity(node_uuid='hub-uuid')

        assert 'error' not in result
        assert result['center_node'] is not None
        assert result['center_node']['uuid'] == 'hub-uuid'
        assert result['center_node']['name'] == 'KHADIJA DAOUD'
        assert result['center_node']['labels'] == ['Entity', 'Persona']
        assert get_by_uuid.await_args.args[1] == 'hub-uuid'

    @pytest.mark.asyncio
    async def test_uuid_center_node_strips_embeddings(self):
        svc, queue, cfg, client = make_mock_services()
        centre = make_mock_node(
            uuid='hub-uuid',
            attributes={'key': 'value', 'name_embedding': [0.1], 'Summary_Embedding': [0.2]},
        )
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch('graphiti_mcp_server.EntityNode.get_by_uuid', AsyncMock(return_value=centre)),
        ):
            result = await explore_entity(node_uuid='hub-uuid')

        attrs = result['center_node']['attributes']
        assert attrs == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_uuid_not_found_says_so(self):
        """A uuid that is genuinely absent is an answer, not an empty neighbourhood."""
        from graphiti_core.errors import NodeNotFoundError

        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch(
                'graphiti_mcp_server.EntityNode.get_by_uuid',
                AsyncMock(side_effect=NodeNotFoundError('ghost-uuid')),
            ),
        ):
            result = await explore_entity(node_uuid='ghost-uuid')

        assert 'error' not in result
        assert 'No node found' in result['message']
        assert 'ghost-uuid' in result['message']
        assert result['center_node'] is None
        assert result['nodes'] == []
        # and it must not have gone on to run the traversal
        assert client.search_.call_count == 0

    @pytest.mark.asyncio
    async def test_uuid_lookup_backend_failure_is_an_error_not_a_missing_node(self):
        """A dropped pool or a backend error must NOT read as "no such node".

        Reporting an infrastructure failure as an absent uuid tells the agent the
        entity does not exist — the same confidently-wrong shape as the original
        bug, one layer up.
        """
        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch(
                'graphiti_mcp_server.EntityNode.get_by_uuid',
                AsyncMock(side_effect=ConnectionError('pool is closed')),
            ),
        ):
            result = await explore_entity(node_uuid='hub-uuid')

        assert 'error' in result
        assert 'pool is closed' in result['error']
        assert 'No node found' not in result.get('error', '')
        assert client.search_.call_count == 0

    @pytest.mark.asyncio
    async def test_name_path_center_node_does_not_lookup_by_uuid(self):
        """The name branch already has the resolved node; no second round-trip."""
        svc, queue, cfg, client = make_mock_services()
        resolved = make_mock_node(uuid='resolved-uuid', name='FoundEntity')
        client.search_ = AsyncMock(
            side_effect=[
                make_mock_search_results(nodes=[resolved]),
                make_mock_search_results(),
            ]
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch('graphiti_mcp_server.EntityNode.get_by_uuid', AsyncMock()) as get_by_uuid,
        ):
            result = await explore_entity(node_name='FoundEntity')

        assert result['center_node']['uuid'] == 'resolved-uuid'
        get_by_uuid.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_depth_capped_at_4(self):
        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch(
                'graphiti_mcp_server.EntityNode.get_by_uuid',
                AsyncMock(return_value=make_mock_node(uuid='some-uuid')),
            ),
        ):
            await explore_entity(node_uuid='some-uuid', depth=10)

        call_kwargs = client.search_.call_args.kwargs
        config = call_kwargs['config']
        # depth is capped at min(depth, 4) = 4
        assert config.edge_config.bfs_max_depth == 4
        assert config.node_config.bfs_max_depth == 4

    @pytest.mark.asyncio
    async def test_edge_types_filter(self):
        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch(
                'graphiti_mcp_server.EntityNode.get_by_uuid',
                AsyncMock(return_value=make_mock_node(uuid='some-uuid')),
            ),
        ):
            await explore_entity(node_uuid='some-uuid', edge_types=['OWNERSHIP'])

        call_kwargs = client.search_.call_args.kwargs
        search_filter = call_kwargs['search_filter']
        assert search_filter.edge_types == ['OWNERSHIP']


# ---------------------------------------------------------------------------
# TestExploreCentreResolution — BUG-100
# ---------------------------------------------------------------------------

class TestExploreCentreResolution:
    """Regression tests for BUG-100: relevance rank used as identity lookup.

    `explore_entity(node_name=...)` resolved its centre by taking
    `resolve_results.nodes[0]` — the top hit of a hybrid (BM25 + embedding, RRF)
    search. A ranking answers "what is most relevant", not "which node IS this".
    Measured on the live FalkorDB arm (policia_partes_bench_v1, 2026-08-18):
    query 'KHADIJA DAOUD' ranked the *different person* 'KHADIJA NASRE EDDINE'
    at 0 and the exact-named node at 1, so the tool centred on the wrong person.
    """

    @pytest.mark.asyncio
    async def test_exact_name_match_beats_higher_ranked_fuzzy_neighbour(self):
        """BUG-100: the exact-named node wins even when ranked below a neighbour."""
        svc, queue, cfg, client = make_mock_services()

        # The live shape: a same-first-name different person outranks the exact node.
        fuzzy = make_mock_node(uuid='wrong-uuid', name='KHADIJA NASRE EDDINE')
        exact = make_mock_node(uuid='right-uuid', name='KHADIJA DAOUD')
        client.search_ = AsyncMock(
            side_effect=[
                make_mock_search_results(nodes=[fuzzy, exact]),
                make_mock_search_results(),
            ]
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity(node_name='KHADIJA DAOUD')

        assert result['center_node']['name'] == 'KHADIJA DAOUD'
        assert result['center_node']['uuid'] == 'right-uuid'
        # The neighbourhood expansion must follow the corrected centre, not the
        # top hit — a right centre with a wrong traversal is still the wrong answer.
        second_call_kwargs = client.search_.call_args_list[1].kwargs
        assert second_call_kwargs['center_node_uuid'] == 'right-uuid'
        assert second_call_kwargs['bfs_origin_node_uuids'] == ['right-uuid']

    @pytest.mark.asyncio
    async def test_no_exact_match_keeps_top_ranked_node(self):
        """Without an exact name, ranking still decides — behaviour unchanged."""
        svc, queue, cfg, client = make_mock_services()

        first = make_mock_node(uuid='first-uuid', name='KHADIJA NASRE EDDINE')
        second = make_mock_node(uuid='second-uuid', name='KHADIJA ABDELKADER')
        client.search_ = AsyncMock(
            side_effect=[
                make_mock_search_results(nodes=[first, second]),
                make_mock_search_results(),
            ]
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity(node_name='KHADIJA')

        assert result['center_node']['uuid'] == 'first-uuid'

    @pytest.mark.asyncio
    async def test_exact_match_is_case_and_whitespace_insensitive(self):
        """Callers type names as prose; the graph stores them uppercased."""
        svc, queue, cfg, client = make_mock_services()

        fuzzy = make_mock_node(uuid='wrong-uuid', name='KHADIJA NASRE EDDINE')
        exact = make_mock_node(uuid='right-uuid', name='KHADIJA DAOUD')
        client.search_ = AsyncMock(
            side_effect=[
                make_mock_search_results(nodes=[fuzzy, exact]),
                make_mock_search_results(),
            ]
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity(node_name='  khadija   daoud ')

        assert result['center_node']['uuid'] == 'right-uuid'

    @pytest.mark.asyncio
    async def test_first_exact_match_wins_when_several_share_the_name(self):
        """Ties among exact matches keep search order — deterministic, not arbitrary."""
        svc, queue, cfg, client = make_mock_services()

        other = make_mock_node(uuid='other-uuid', name='KHADIJA NASRE EDDINE')
        exact_a = make_mock_node(uuid='exact-a', name='KHADIJA DAOUD')
        exact_b = make_mock_node(uuid='exact-b', name='khadija daoud')
        client.search_ = AsyncMock(
            side_effect=[
                make_mock_search_results(nodes=[other, exact_a, exact_b]),
                make_mock_search_results(),
            ]
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity(node_name='KHADIJA DAOUD')

        assert result['center_node']['uuid'] == 'exact-a'

    @pytest.mark.asyncio
    async def test_uuid_branch_untouched_by_name_matching(self):
        """A uuid is already an identity; exact-name preference must not apply."""
        svc, queue, cfg, client = make_mock_services()
        target = make_mock_node(uuid='direct-uuid', name='KHADIJA NASRE EDDINE')
        client.search_ = AsyncMock(return_value=make_mock_search_results())

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch(
                'graphiti_mcp_server.EntityNode.get_by_uuid',
                AsyncMock(return_value=target),
            ),
        ):
            result = await explore_entity(
                node_uuid='direct-uuid', node_name='KHADIJA DAOUD',
            )

        assert result['center_node']['uuid'] == 'direct-uuid'
        assert result['center_node']['name'] == 'KHADIJA NASRE EDDINE'

    @pytest.mark.asyncio
    async def test_replays_the_measured_live_ranking(self):
        """The real BUG-100 ranking, verbatim, driven through the tool.

        Captured 2026-08-18 from the FalkorDB arm (connector mcp-v2.4.0, graph
        `policia_partes_bench_v1`) via MCP `search` with search_mode='nodes',
        reranker='rrf' — SEARCH_RECIPES[('nodes','rrf')] is the very
        NODE_HYBRID_SEARCH_RRF config the centre resolution uses, so this is the
        ordering explore_entity actually saw when it answered with the wrong
        person. Held as data, not prose, so the fix stays pinned to the
        observation that motivated it.
        """
        live_ranking = [
            ('a3bb5a1f-0593-49bb-acd1-5231e027dbc2', 'KHADIJA NASRE EDDINE'),
            ('c08eb5e2-bc36-451e-91ef-c657b1e61f7f', 'KHADIJA DAOUD'),
            ('b043b0a9-adf3-4145-ae7b-1c3d1fd0188d', 'KHADIJA ABDELKADER'),
            ('711ceaf7-1c76-428b-a146-d444da2a13e8', 'KHADIJA BEN AISA'),
            ('b2dcec4a-6bdb-42ec-98d1-d5cbf8a1e331', 'KHADIJA EL YOUSFI'),
            ('69f80556-455e-4886-a4d8-9a6183a31cc8', 'MOHAMED DAOUD'),
            ('4437f6e6-2db6-40e3-80af-28bd24bcfe70', 'IBRAHIM DAOUD'),
            ('85a5d53a-c3f7-4dce-9553-deb38481fc6c',
             'Identificacion de KHADIJA DAOUD en 20260000100001'),
            ('a22d03a9-4769-454d-8033-d37c2068f0dc',
             'Identificacion de KHADIJA DAOUD en 20260000100018'),
            ('33155c38-ff47-4396-be1a-e3fb96343643',
             'Testimonio de KHADIJA DAOUD en 20260000100089'),
        ]

        svc, queue, cfg, client = make_mock_services()
        client.search_ = AsyncMock(
            side_effect=[
                make_mock_search_results(
                    nodes=[make_mock_node(uuid=u, name=n) for u, n in live_ranking],
                ),
                make_mock_search_results(),
            ]
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity(node_name='KHADIJA DAOUD')

        # The Persona, not the rank-0 different person and not one of the role
        # nodes whose names merely CONTAIN the query string.
        assert result['center_node']['name'] == 'KHADIJA DAOUD'
        assert result['center_node']['uuid'] == 'c08eb5e2-bc36-451e-91ef-c657b1e61f7f'

    @pytest.mark.asyncio
    async def test_exact_match_survives_nfc_nfd_disagreement(self):
        """Same letters, different Unicode form, still the same name.

        The graph holds 'JOSÉ MARÍA' composed (U+00C9 / U+00CD); the caller
        sends it decomposed (E + U+0301, I + U+0301). Byte-unequal, casefold
        does not reconcile them, so without NFC folding the exact match is lost
        and resolution drops back to the ranking — the wrong person, which is
        BUG-100 all over again on the accent-carrying half of a Spanish corpus.
        """
        # Built with unicodedata rather than written as two source literals: a
        # formatter or editor that normalizes this file would quietly collapse
        # the literals into one form and leave the test asserting nothing.
        name = 'JOSÉ MARÍA GARCIA'
        composed = unicodedata.normalize('NFC', name)
        decomposed = unicodedata.normalize('NFD', name)
        assert composed != decomposed  # different bytes...
        # ...which casefold alone does NOT reconcile - the reason NFC is needed.
        assert composed.casefold() != decomposed.casefold()

        svc, queue, cfg, client = make_mock_services()
        fuzzy = make_mock_node(uuid='wrong-uuid', name='JOSE MARIA GARRIDO')
        exact = make_mock_node(uuid='right-uuid', name=composed)
        client.search_ = AsyncMock(
            side_effect=[
                make_mock_search_results(nodes=[fuzzy, exact]),
                make_mock_search_results(),
            ]
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity(node_name=decomposed)

        assert result['center_node']['uuid'] == 'right-uuid'

    @pytest.mark.asyncio
    async def test_nameless_node_in_results_does_not_break_matching(self):
        """A node whose name is None must not blow up the comparison."""
        svc, queue, cfg, client = make_mock_services()

        nameless = make_mock_node(uuid='nameless-uuid', name='KHADIJA NASRE EDDINE')
        nameless.name = None
        exact = make_mock_node(uuid='right-uuid', name='KHADIJA DAOUD')
        client.search_ = AsyncMock(
            side_effect=[
                make_mock_search_results(nodes=[nameless, exact]),
                make_mock_search_results(),
            ]
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_entity(node_name='KHADIJA DAOUD')

        assert result['center_node']['uuid'] == 'right-uuid'


# ---------------------------------------------------------------------------
# TestGetEpisodeContext
# ---------------------------------------------------------------------------

class TestGetEpisodeContext:
    """Tests for the get_episode_context tool function."""

    @pytest.mark.asyncio
    async def test_service_not_initialized(self):
        with patch('graphiti_mcp_server.graphiti_service', None):
            result = await get_episode_context(episode_uuids=['ep-1'])
        assert 'error' in result
        assert 'not initialized' in result['error']

    @pytest.mark.asyncio
    async def test_empty_episode_uuids(self):
        svc, queue, cfg, client = make_mock_services()

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await get_episode_context(episode_uuids=[])

        assert 'error' in result
        assert 'at least one episode UUID' in result['error']

    @pytest.mark.asyncio
    async def test_returns_formatted_nodes_and_edges(self):
        """The extraction half, which this tool has always served.

        The message wording moved when the tool started returning the episodes
        THEMSELVES as well (`search` caps episode content and names this tool as
        where the rest lives, so it now reports found-of-requested rather than
        only the request size). The episode half is covered in
        test_episode_search_wire.py::TestTheTruncationRemedyIsReal.
        """
        svc, queue, cfg, client = make_mock_services()
        node = make_mock_node(uuid='ep-node-1', name='EpNode')
        edge = make_mock_edge(uuid='ep-edge-1', fact='EpNode is related to X')
        client.get_nodes_and_edges_by_episode = AsyncMock(
            return_value=make_mock_search_results(nodes=[node], edges=[edge]),
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
            patch(
                'graphiti_mcp_server.EpisodicNode.get_by_uuids',
                AsyncMock(return_value=[]),
            ),
        ):
            result = await get_episode_context(episode_uuids=['ep-uuid-1', 'ep-uuid-2'])

        assert 'error' not in result
        assert len(result['nodes']) == 1
        assert result['nodes'][0]['uuid'] == 'ep-node-1'
        assert len(result['edges']) == 1
        assert result['edges'][0]['uuid'] == 'ep-edge-1'
        assert '2 requested episodes' in result['message']

        # Verify it was called with the right UUIDs
        client.get_nodes_and_edges_by_episode.assert_awaited_once_with(
            ['ep-uuid-1', 'ep-uuid-2'],
        )


# ---------------------------------------------------------------------------
# TestBuildCommunities
# ---------------------------------------------------------------------------

class TestBuildCommunities:
    """Tests for the build_communities tool function."""

    @pytest.mark.asyncio
    async def test_service_not_initialized(self):
        with patch('graphiti_mcp_server.graphiti_service', None):
            result = await build_communities(group_ids=['g1'])
        assert 'error' in result
        assert 'not initialized' in result['error']

    @pytest.mark.asyncio
    async def test_empty_group_ids(self):
        svc, queue, cfg, client = make_mock_services()

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await build_communities(group_ids=[])

        assert 'error' in result
        assert 'at least one group_id' in result['error']

    @pytest.mark.asyncio
    async def test_returns_community_count_and_results(self):
        svc, queue, cfg, client = make_mock_services()
        comm1 = make_mock_community(uuid='c1', name='Cluster A')
        comm2 = make_mock_community(uuid='c2', name='Cluster B')
        # build_communities returns (community_nodes, community_edges)
        mock_community_edges = [MagicMock()]
        client.build_communities = AsyncMock(
            return_value=([comm1, comm2], mock_community_edges),
        )

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await build_communities(group_ids=['g1', 'g2'])

        assert 'error' not in result
        assert result['community_count'] == 2
        assert len(result['communities']) == 2
        assert result['communities'][0]['uuid'] == 'c1'
        assert result['communities'][1]['uuid'] == 'c2'
        assert '2 communities' in result['message']
        assert '2 graphs' in result['message']


# ---------------------------------------------------------------------------
# TestAddMemory
# ---------------------------------------------------------------------------

class TestAddMemory:
    """Tests for the add_memory tool function."""

    @pytest.mark.asyncio
    async def test_services_not_initialized(self):
        with (
            patch('graphiti_mcp_server.graphiti_service', None),
            patch('graphiti_mcp_server.queue_service', None),
        ):
            result = await add_memory(name='Test', episode_body='body')
        assert 'error' in result
        assert 'not initialized' in result['error']

    @pytest.mark.asyncio
    async def test_single_mode_queues_episode(self):
        svc, queue, cfg, client = make_mock_services(group_id='default-grp')
        queue.add_episode = AsyncMock(return_value=1)

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await add_memory(name='MyEpisode', episode_body='Some content')

        assert 'error' not in result
        assert 'queued' in result['message'].lower()
        assert 'MyEpisode' in result['message']
        queue.add_episode.assert_awaited_once()
        call_kwargs = queue.add_episode.call_args.kwargs
        assert call_kwargs['group_id'] == 'default-grp'
        assert call_kwargs['name'] == 'MyEpisode'
        assert call_kwargs['content'] == 'Some content'

    @pytest.mark.asyncio
    async def test_single_mode_missing_name(self):
        svc, queue, cfg, client = make_mock_services()

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await add_memory(episode_body='Some content')

        assert 'error' in result
        assert 'name' in result['error'].lower() or 'episode_body' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_single_mode_missing_body(self):
        svc, queue, cfg, client = make_mock_services()

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await add_memory(name='MyEpisode')

        assert 'error' in result
        assert 'name' in result['error'].lower() or 'episode_body' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_bulk_mode_success(self):
        svc, queue, cfg, client = make_mock_services(group_id='bulk-grp')
        # add_episode_bulk returns an object with .nodes and .edges
        bulk_result = MagicMock()
        bulk_result.nodes = [make_mock_node(), make_mock_node(uuid='n2')]
        bulk_result.edges = [make_mock_edge()]
        client.add_episode_bulk = AsyncMock(return_value=bulk_result)

        episodes = [
            {'name': 'Doc 1', 'content': 'Content 1', 'source': 'text'},
            {'name': 'Doc 2', 'content': 'Content 2', 'source': 'json'},
        ]

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await add_memory(episodes=episodes)

        assert 'error' not in result
        assert '2 episodes' in result['message']
        assert '2 nodes' in result['message']
        assert '1 edges' in result['message']
        client.add_episode_bulk.assert_awaited_once()
        call_kwargs = client.add_episode_bulk.call_args.kwargs
        assert call_kwargs['group_id'] == 'bulk-grp'
        assert len(call_kwargs['bulk_episodes']) == 2

    @pytest.mark.asyncio
    async def test_bulk_mode_empty_list(self):
        svc, queue, cfg, client = make_mock_services()

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await add_memory(episodes=[])

        assert 'error' in result
        assert 'empty' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_bulk_mode_missing_name_key(self):
        svc, queue, cfg, client = make_mock_services()

        episodes = [{'content': 'Some content'}]  # missing 'name'

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await add_memory(episodes=episodes)

        assert 'error' in result
        assert 'name' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_bulk_mode_missing_content_key(self):
        svc, queue, cfg, client = make_mock_services()

        episodes = [{'name': 'Doc 1'}]  # missing 'content'

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await add_memory(episodes=episodes)

        assert 'error' in result
        assert 'content' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_invalid_source_type_defaults_to_text(self):
        svc, queue, cfg, client = make_mock_services()
        queue.add_episode = AsyncMock(return_value=1)

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await add_memory(
                name='Test', episode_body='body', source='invalid_type',
            )

        # Should not error — falls back to EpisodeType.text
        assert 'error' not in result
        assert 'queued' in result['message'].lower()
        # Verify the episode_type passed is EpisodeType.text
        call_kwargs = queue.add_episode.call_args.kwargs
        from graphiti_core.nodes import EpisodeType
        assert call_kwargs['episode_type'] == EpisodeType.text

    @pytest.mark.asyncio
    async def test_group_id_defaults_to_config(self):
        svc, queue, cfg, client = make_mock_services(group_id='config-default')
        queue.add_episode = AsyncMock(return_value=1)

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.queue_service', queue),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await add_memory(
                name='Test', episode_body='body',
            )

        assert 'error' not in result
        call_kwargs = queue.add_episode.call_args.kwargs
        assert call_kwargs['group_id'] == 'config-default'


# ---------------------------------------------------------------------------
# TestOntologyConfig
# ---------------------------------------------------------------------------

class TestOntologyConfig:
    """Tests for ontology_graph configuration field."""

    def test_ontology_graph_defaults_to_none(self):
        from config.schema import GraphitiAppConfig
        app_config = GraphitiAppConfig(group_id='test', user_id='test')
        assert app_config.ontology_graph is None

    def test_ontology_graph_can_be_set(self):
        from config.schema import GraphitiAppConfig
        app_config = GraphitiAppConfig(
            group_id='test',
            user_id='test',
            ontology_graph='my_ontology',
        )
        assert app_config.ontology_graph == 'my_ontology'


# ---------------------------------------------------------------------------
# TestOntologyClientInit
# ---------------------------------------------------------------------------

class TestOntologyClientInit:
    """Tests for ontology client initialization in make_mock_services."""

    def test_make_mock_services_includes_ontology_client(self):
        """Verify the mock services factory exposes ontology_client."""
        svc, queue, cfg, client = make_mock_services()
        # graphiti_service should have ontology_client attribute
        assert hasattr(svc, 'ontology_client')


# ---------------------------------------------------------------------------
# TestSearchOntology
# ---------------------------------------------------------------------------

class TestSearchOntology:
    """Tests for the search_ontology tool function."""

    @pytest.mark.asyncio
    async def test_no_ontology_configured(self):
        svc, queue, cfg, client = make_mock_services()
        svc.ontology_client = None

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await search_ontology(query='AirworthinessDirective')

        assert 'error' in result
        assert 'No ontology graph configured' in result['error']

    @pytest.mark.asyncio
    async def test_basic_ontology_search(self):
        svc, queue, cfg, client = make_mock_services()
        ontology_client = AsyncMock()
        svc.ontology_client = ontology_client
        svc._ensure_ontology_client = AsyncMock(return_value=True)

        node = make_mock_node(uuid='onto-1', name='AirworthinessDirective', labels=['OntologyClass'])
        ontology_client.search_ = AsyncMock(
            return_value=make_mock_search_results(nodes=[node]),
        )

        cfg.graphiti.ontology_graph = 'ad_ontology'

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await search_ontology(query='AirworthinessDirective')

        assert 'error' not in result
        assert len(result['nodes']) == 1
        assert result['nodes'][0]['name'] == 'AirworthinessDirective'

        # Verify group_ids uses ontology_graph name
        call_kwargs = ontology_client.search_.call_args.kwargs
        assert call_kwargs['group_ids'] == ['ad_ontology']

    @pytest.mark.asyncio
    async def test_service_not_initialized(self):
        with patch('graphiti_mcp_server.graphiti_service', None):
            result = await search_ontology(query='test')
        assert 'error' in result
        assert 'not initialized' in result['error']


# ---------------------------------------------------------------------------
# TestExploreOntology
# ---------------------------------------------------------------------------

class TestExploreOntology:
    """Tests for the explore_ontology tool function."""

    @pytest.mark.asyncio
    async def test_no_ontology_configured(self):
        svc, queue, cfg, client = make_mock_services()
        svc.ontology_client = None

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_ontology(node_name='Aircraft')

        assert 'error' in result
        assert 'No ontology graph configured' in result['error']

    @pytest.mark.asyncio
    async def test_neither_name_nor_uuid(self):
        svc, queue, cfg, client = make_mock_services()
        svc.ontology_client = AsyncMock()
        svc._ensure_ontology_client = AsyncMock(return_value=True)

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_ontology()

        assert 'error' in result
        assert 'node_name or node_uuid' in result['error']

    @pytest.mark.asyncio
    async def test_name_resolution_and_explore(self):
        """v1.0.3 class-context payload: center + relationships + hierarchy + neighbors.

        Exact-name resolution happens against the OntologyClass rows; the
        semantic search_ fallback must NOT be needed for an exact name.
        (Full payload coverage lives in test_ontology_tiers.py.)
        """
        svc, queue, cfg, client = make_mock_services()
        svc.config.graphiti.ontology_graph = 'test_ontology'
        svc._ensure_ontology_client = AsyncMock(return_value=True)

        rows = [
            {
                'uuid': 'onto-uuid',
                'name': 'Aircraft',
                'ontology_type': 'class',
                'summary': 'A powered flying machine. Registered with an authority.',
                'alt_labels': [],
                'inherits_from': [],
                'examples': [],
                'source_entity': None,
                'target_entity': None,
                'properties': None,
                'identity': None,
            },
            {
                'uuid': 'airline-uuid',
                'name': 'Airline',
                'ontology_type': 'class',
                'summary': 'An operator of aircraft. Holds an operating certificate.',
                'alt_labels': [],
                'inherits_from': [],
                'examples': [],
                'source_entity': None,
                'target_entity': None,
                'properties': None,
                'identity': None,
            },
            {
                'uuid': 'rel-uuid',
                'name': 'OPERATED_BY',
                'ontology_type': 'relationship_class',
                'summary': 'Links an aircraft to its operator.',
                'alt_labels': [],
                'inherits_from': [],
                'examples': [],
                'source_entity': 'Aircraft',
                'target_entity': 'Airline',
                'properties': None,
                'identity': None,
            },
        ]

        ontology_client = MagicMock()
        ontology_client.driver.execute_query = AsyncMock(return_value=(rows, None, None))
        ontology_client.search_ = AsyncMock(return_value=make_mock_search_results())
        svc.ontology_client = ontology_client

        with (
            patch('graphiti_mcp_server.graphiti_service', svc),
            patch('graphiti_mcp_server.config', cfg, create=True),
        ):
            result = await explore_ontology(node_name='Aircraft')

        assert 'error' not in result
        assert result['center']['name'] == 'Aircraft'
        assert result['relationships']['outgoing'] == [
            {
                'name': 'OPERATED_BY',
                'target': 'Airline',
                'summary': 'Links an aircraft to its operator.',
            }
        ]
        assert [n['name'] for n in result['neighbors']] == ['Airline']
        # Exact name matched in the rows — no semantic resolution needed.
        ontology_client.search_.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_service_not_initialized(self):
        with patch('graphiti_mcp_server.graphiti_service', None):
            result = await explore_ontology(node_name='test')
        assert 'error' in result
        assert 'not initialized' in result['error']


# ---------------------------------------------------------------------------
# TestSchemaConstraints
# ---------------------------------------------------------------------------

class TestSchemaConstraints:
    """Verify that tightened schemas use Literal types."""

    def test_search_mode_is_literal(self):
        from graphiti_mcp_server import search
        import inspect
        sig = inspect.signature(search)
        search_mode_annotation = sig.parameters['search_mode'].annotation
        assert hasattr(search_mode_annotation, '__args__'), 'search_mode should be Literal type'
        assert 'nodes' in search_mode_annotation.__args__
        assert 'edges' in search_mode_annotation.__args__
        assert 'communities' in search_mode_annotation.__args__
        assert 'combined' in search_mode_annotation.__args__

    def test_search_reranker_is_literal(self):
        from graphiti_mcp_server import search
        import inspect
        sig = inspect.signature(search)
        reranker_annotation = sig.parameters['reranker'].annotation
        assert hasattr(reranker_annotation, '__args__'), 'reranker should be Literal type'
        assert 'rrf' in reranker_annotation.__args__
        assert 'mmr' in reranker_annotation.__args__
        assert 'cross_encoder' in reranker_annotation.__args__
        assert 'node_distance' in reranker_annotation.__args__
        assert 'episode_mentions' in reranker_annotation.__args__

    def test_add_memory_source_is_literal(self):
        from graphiti_mcp_server import add_memory
        import inspect
        sig = inspect.signature(add_memory)
        source_annotation = sig.parameters['source'].annotation
        assert hasattr(source_annotation, '__args__'), 'source should be Literal type'
        assert 'text' in source_annotation.__args__
        assert 'json' in source_annotation.__args__
        assert 'message' in source_annotation.__args__

    def test_explore_node_depth_is_literal(self):
        from graphiti_mcp_server import explore_entity
        import inspect
        sig = inspect.signature(explore_entity)
        depth_annotation = sig.parameters['depth'].annotation
        assert hasattr(depth_annotation, '__args__'), 'depth should be Literal type'
        assert 1 in depth_annotation.__args__
        assert 4 in depth_annotation.__args__

    def test_search_ontology_search_mode_is_literal(self):
        from graphiti_mcp_server import search_ontology
        import inspect
        sig = inspect.signature(search_ontology)
        search_mode_annotation = sig.parameters['search_mode'].annotation
        assert hasattr(search_mode_annotation, '__args__'), 'search_mode should be Literal type'

    def test_search_ontology_reranker_is_literal(self):
        from graphiti_mcp_server import search_ontology
        import inspect
        sig = inspect.signature(search_ontology)
        reranker_annotation = sig.parameters['reranker'].annotation
        assert hasattr(reranker_annotation, '__args__'), 'reranker should be Literal type'
        # search_ontology only supports rrf, mmr, cross_encoder (no node_distance/episode_mentions)
        assert 'rrf' in reranker_annotation.__args__
        assert 'mmr' in reranker_annotation.__args__
        assert 'cross_encoder' in reranker_annotation.__args__


# ---------------------------------------------------------------------------
# TestDynamicRegistration
# ---------------------------------------------------------------------------

class TestDynamicRegistration:
    """Verify that dynamic tool registration and resources work.

    CONTRACT-marked at CLASS scope, not module scope, and deliberately so. This
    module is a behavioural unit suite — most of it asserts what arguments the
    tools pass DOWN to `search_`/`add_episode`, which is implementation, not the
    announced surface. Marking the whole file would widen `contract` from
    "ADR-015/019 surface guards" to "any unit test", diluting the one marker the
    CI job selects on. What registration puts into `tools/list` IS that surface,
    so the class carries the marker and the rest of the module does not.
    """

    pytestmark = pytest.mark.contract

    def test_register_dynamic_tools_adds_tools_to_mcp(self, monkeypatch):
        import graphiti_mcp_server as srv
        from domain_profile import DomainProfile, EntityTypeInfo, EdgeTypeInfo
        from graphiti_mcp_server import register_dynamic_tools, mcp

        profile = DomainProfile(
            group_id='test_graph',
            entity_types={
                'Aircraft': EntityTypeInfo('Aircraft', 10, 'Test aircraft', ['PH-KZB']),
            },
            edge_types={
                'OPERATED_BY': EdgeTypeInfo('OPERATED_BY', 5, 'Test link', 'Aircraft -> Airline'),
            },
            time_range=None,
        )

        # The ontology tools are served only where an ontology graph is
        # configured (M11) — configure one, since this asserts they register.
        monkeypatch.setattr(
            srv,
            'config',
            type(
                'C',
                (),
                {
                    'graphiti': type(
                        'G', (), {'ontology_graph': 'onto_v1', 'group_id': 'test_graph'}
                    )
                },
            ),
            raising=False,
        )
        register_dynamic_tools(profile)

        # Verify tools are registered with dynamic descriptions
        tools = mcp._tool_manager._tools
        assert 'search' in tools
        assert 'Aircraft' in tools['search'].description
        assert 'explore_entity' in tools
        assert 'search_ontology' in tools
        assert 'explore_ontology' in tools

    def test_register_resources_adds_three_resources(self):
        from domain_profile import DomainProfile, EntityTypeInfo
        from graphiti_mcp_server import register_resources, mcp

        profile = DomainProfile(
            group_id='test_graph',
            entity_types={
                'Aircraft': EntityTypeInfo('Aircraft', 10, 'Test', ['PH-KZB']),
            },
            edge_types={},
            time_range=None,
        )

        register_resources(profile)

        resources = mcp._resource_manager._resources
        assert any('domain_summary' in str(uri) for uri in resources)
        assert any('entity_catalog' in str(uri) for uri in resources)
        assert any('relationship_types' in str(uri) for uri in resources)
