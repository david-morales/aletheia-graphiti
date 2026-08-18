"""BUG-87: an entity edge stored under a TYPED label must still be reachable.

The bug
-------
BUG-62 fixed the hybrid SEARCH legs. The "an entity edge is a ``RELATES_TO``"
assumption survived everywhere else. The paths fixed here, and what each one
cost on a bulk-ingested graph:

LIVE on FalkorDB — ``FalkorDriver`` sets neither ``search_interface`` nor
``graph_operations_interface``, so nothing intercepts any of these:

* ``edges.py`` — ``Edge.delete`` / ``Edge.delete_by_uuids`` matched
  ``MENTIONS|RELATES_TO|HAS_MEMBER``: a bulk-written ``DETIENE`` edge could not
  be deleted through the model API at all;
* ``search_utils.get_relevant_edges`` / ``get_edge_invalidation_candidates`` —
  a typed edge was never OFFERED as a dedup or temporal-invalidation candidate:
  duplicates never merged, superseded facts never expired;
* ``search_utils.node_distance_reranker`` — every candidate scored as
  unconnected, so ``explore_node``'s "ranked by proximity to the center node"
  was a no-op and truncation then dropped arbitrary results;
* ``search_utils.get_embeddings_for_edges`` — returned ``{}``, so the bulk path
  re-embedded facts it had already embedded and compared dedup against nothing;
* ``utils/maintenance/community_operations.py`` — the neighbour projection came
  back empty, so ``build_communities`` ran label propagation over a graph of
  isolated nodes.

NOT live — staged for an upstream refactor, zero consumers repo-wide today:

* ``driver/falkordb/operations/search_ops.py`` — all three edge methods carried
  the constant, plus a fulltext leg that asked only the ``RELATES_TO`` index.
  Fixed so the refactor does not land BUG-62 again; see
  ``TestTheFalkorOpsModuleEdgeMethods``, which says so in place. **Nothing in
  this module was a live falkor fulltext/similarity/BFS leg** — those live in
  ``search_utils`` and were already untyped from the BUG-62 wave.

How it fails is not uniform, and the difference matters:

* on **FalkorDB** the typed pattern parses and matches ZERO rows. Silent.
* on **AGE** the alternation form is not silent at all — ``MENTIONS|RELATES_TO
  |HAS_MEMBER`` is a hard ``ERROR: syntax error at or near "|"`` (measured in
  review). Loud, and in the branch's favour: it cannot have corrupted anything.

Who has typed edges
-------------------
FalkorDB (``bulk_utils`` MERGEs each edge under the extracted ``name``) and AGE
(``_edge_label()`` always stores the typed name). Neo4j, Kuzu and Neptune really
do write ``RELATES_TO``, and the tests below pin that they are LEFT ALONE — the
fix must not turn a correct, index-served pattern into a scan on the providers
that never had the bug.

How these tests prove it
------------------------
Each test drives the real function against a fake driver that captures the
emitted Cypher, then runs that Cypher's relationship pattern through
``pattern_selects`` — a small model of "would this pattern match an edge of type
T". So the assertions are about what the query WOULD DO to a typed edge, not
about the presence of a string. Reintroduce the constant and they go red.
"""

import re
from datetime import datetime, timezone

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.graph_queries import (
    TYPED_EDGE_LABEL_PROVIDERS,
    entity_edge_pattern_type,
)
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    get_edge_invalidation_candidates,
    get_embeddings_for_edges,
    get_relevant_edges,
    node_distance_reranker,
)
from graphiti_core.utils.maintenance.community_operations import get_community_clusters

# A relationship pattern: `-[e:A|B {…}]->` / `-[e {…}]-` / `-[e]-`.
_REL_PATTERN_RE = re.compile(
    r'-\[(?P<var>\w*)(?P<types>(?::\w+)(?:\|\w+)*)?\s*(?P<props>\{[^}]*\})?\]'
)

# A typed edge as the FalkorDB bulk writer and the AGE writer store one.
TYPED_LABEL = 'DETIENE'


def pattern_selects(cypher: str, edge_type: str, *, var: str = 'e') -> bool:
    """Would the `[{var}…]` relationship pattern in `cypher` match `edge_type`?

    An untyped pattern matches anything; a typed one matches only the types it
    names. That is the whole of the bug: the patterns named three types and the
    graph held hundreds.
    """
    for match in _REL_PATTERN_RE.finditer(cypher):
        if match.group('var') != var:
            continue
        types = match.group('types')
        if not types:
            return True
        return edge_type in types.lstrip(':').split('|')
    raise AssertionError(f'no `[{var}…]` relationship pattern found in:\n{cypher}')


class _FakeDriver:
    """Captures the Cypher a core function emits, and answers nothing useful."""

    def __init__(self, provider: GraphProvider):
        self.provider = provider
        self.graph_operations_interface = None
        self.search_interface = None
        self.queries: list[str] = []
        self.fulltext_syntax = ''

    async def execute_query(self, cypher: str, **kwargs):
        self.queries.append(cypher)
        return [], None, None


def _edge() -> EntityEdge:
    return EntityEdge(
        uuid='edge-1',
        source_node_uuid='src',
        target_node_uuid='tgt',
        group_id='g',
        name=TYPED_LABEL,
        fact='X detiene Y',
        fact_embedding=[0.1, 0.2, 0.3, 0.4],
        created_at=datetime.now(timezone.utc),
    )


# ------------------------------------------------------- the shared helper


class TestTheSharedPatternHelper:
    """One helper, so the legs cannot drift apart again."""

    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    def test_typed_label_providers_get_an_untyped_pattern(self, provider):
        assert entity_edge_pattern_type(provider) == ''

    @pytest.mark.parametrize(
        'provider', [GraphProvider.NEO4J, GraphProvider.KUZU, GraphProvider.NEPTUNE]
    )
    def test_every_other_provider_keeps_the_constant(self, provider):
        assert entity_edge_pattern_type(provider) == ':RELATES_TO'

    def test_falkordb_and_age_are_exactly_the_typed_set(self):
        assert frozenset({GraphProvider.FALKORDB, GraphProvider.AGE}) == TYPED_EDGE_LABEL_PROVIDERS

    def test_the_model_of_a_pattern_is_itself_right(self):
        assert pattern_selects('MATCH (n)-[e]->(m)', TYPED_LABEL)
        assert pattern_selects('MATCH (n)-[e {uuid: $uuid}]->(m)', TYPED_LABEL)
        assert not pattern_selects('MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER]->(m)', TYPED_LABEL)
        assert pattern_selects('MATCH (n)-[e:MENTIONS|DETIENE]->(m)', TYPED_LABEL)
        assert pattern_selects('MATCH (n)-[e:RELATES_TO]->(m)', 'RELATES_TO')


# ------------------------------------------------------------ the delete path


class TestDeleteReachesATypedEdge:
    """`EntityEdge.delete` could not delete a bulk-written edge (BUG-87)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_delete_matches_a_typed_edge(self, provider):
        driver = _FakeDriver(provider)
        await _edge().delete(driver)
        assert len(driver.queries) == 1
        assert pattern_selects(driver.queries[0], TYPED_LABEL)
        assert 'uuid: $uuid' in driver.queries[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_delete_by_uuids_matches_a_typed_edge(self, provider):
        driver = _FakeDriver(provider)
        await EntityEdge.delete_by_uuids(driver, ['edge-1', 'edge-2'])
        assert len(driver.queries) == 1
        assert pattern_selects(driver.queries[0], TYPED_LABEL)
        assert 'e.uuid IN $uuids' in driver.queries[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_delete_still_reaches_the_structural_types(self, provider):
        """Widening must not lose what the old list DID reach."""
        driver = _FakeDriver(provider)
        await _edge().delete(driver)
        for structural in ('MENTIONS', 'RELATES_TO', 'HAS_MEMBER'):
            assert pattern_selects(driver.queries[0], structural)

    @pytest.mark.asyncio
    async def test_neo4j_keeps_its_typed_pattern(self):
        """Neo4j genuinely writes RELATES_TO — do not turn its index into a scan."""
        driver = _FakeDriver(GraphProvider.NEO4J)
        await _edge().delete(driver)
        assert 'MENTIONS|RELATES_TO|HAS_MEMBER' in driver.queries[0]
        assert not pattern_selects(driver.queries[0], TYPED_LABEL)

    @pytest.mark.asyncio
    async def test_kuzu_keeps_its_two_statement_shape(self):
        """Kuzu reifies entity edges as `RelatesToNode_` nodes — untouched."""
        driver = _FakeDriver(GraphProvider.KUZU)
        await _edge().delete(driver)
        assert len(driver.queries) == 2
        assert 'RelatesToNode_' in driver.queries[1]


# --------------------------------------------------- dedup + invalidation


class TestDedupCandidatesReachATypedEdge:
    """`get_relevant_edges` never OFFERED a typed edge, so duplicates stayed."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_a_typed_edge_is_a_dedup_candidate(self, provider):
        driver = _FakeDriver(provider)
        await get_relevant_edges(driver, [_edge()], SearchFilters())
        assert pattern_selects(driver.queries[0], TYPED_LABEL)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_the_endpoints_still_scope_it_to_entity_edges(self, provider):
        driver = _FakeDriver(provider)
        await get_relevant_edges(driver, [_edge()], SearchFilters())
        cypher = driver.queries[0]
        assert '(n:Entity {uuid: edge.source_node_uuid})' in cypher
        assert '(m:Entity {uuid: edge.target_node_uuid})' in cypher

    @pytest.mark.asyncio
    async def test_neo4j_is_untouched(self):
        driver = _FakeDriver(GraphProvider.NEO4J)
        await get_relevant_edges(driver, [_edge()], SearchFilters())
        assert 'e:RELATES_TO {group_id: edge.group_id}' in driver.queries[0]


class TestInvalidationCandidatesReachATypedEdge:
    """`get_edge_invalidation_candidates` — the temporal half of the same bug."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_a_typed_edge_is_an_invalidation_candidate(self, provider):
        driver = _FakeDriver(provider)
        await get_edge_invalidation_candidates(driver, [_edge()], SearchFilters())
        assert pattern_selects(driver.queries[0], TYPED_LABEL)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_the_endpoints_still_scope_it_to_entity_edges(self, provider):
        driver = _FakeDriver(provider)
        await get_edge_invalidation_candidates(driver, [_edge()], SearchFilters())
        cypher = driver.queries[0]
        assert '(n:Entity)-[e' in cypher
        assert '(m:Entity)' in cypher

    @pytest.mark.asyncio
    async def test_neo4j_is_untouched(self):
        driver = _FakeDriver(GraphProvider.NEO4J)
        await get_edge_invalidation_candidates(driver, [_edge()], SearchFilters())
        assert 'e:RELATES_TO {group_id: edge.group_id}' in driver.queries[0]


# ------------------------------- the three pins the first round missed (M2)


class TestTheRerankerReachesATypedEdge:
    """`node_distance_reranker` — the LIVE falkor reranker (no search_interface).

    Scoring every candidate as unconnected made `explore_node`'s documented
    "ranked by proximity to the center node" a no-op, and the truncation that
    follows then dropped arbitrary results rather than the furthest ones.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_a_typed_edge_makes_a_node_adjacent(self, provider):
        driver = _FakeDriver(provider)
        await node_distance_reranker(driver, ['a', 'b'], 'center')
        # The pattern is anonymous here — `-[…]-`, no variable.
        assert pattern_selects(driver.queries[0], TYPED_LABEL, var='')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_the_endpoints_still_scope_it(self, provider):
        driver = _FakeDriver(provider)
        await node_distance_reranker(driver, ['a'], 'center')
        assert '(center:Entity {uuid: $center_uuid})' in driver.queries[0]
        assert '(n:Entity {uuid: node_uuid})' in driver.queries[0]

    @pytest.mark.asyncio
    async def test_neo4j_keeps_its_typed_pattern(self):
        driver = _FakeDriver(GraphProvider.NEO4J)
        await node_distance_reranker(driver, ['a'], 'center')
        assert ')-[:RELATES_TO]-(' in driver.queries[0]
        assert not pattern_selects(driver.queries[0], TYPED_LABEL, var='')


class TestBulkEmbeddingLoadReachesATypedEdge:
    """`get_embeddings_for_edges` returned {} for every bulk-written edge."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_a_typed_edge_is_loadable(self, provider):
        driver = _FakeDriver(provider)
        await get_embeddings_for_edges(driver, [_edge()])
        assert pattern_selects(driver.queries[0], TYPED_LABEL)
        assert 'e.uuid IN $edge_uuids' in driver.queries[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_the_endpoints_still_scope_it(self, provider):
        driver = _FakeDriver(provider)
        await get_embeddings_for_edges(driver, [_edge()])
        assert '(n:Entity)-[e' in driver.queries[0]
        assert '(m:Entity)' in driver.queries[0]

    @pytest.mark.asyncio
    async def test_neo4j_keeps_its_typed_pattern(self):
        driver = _FakeDriver(GraphProvider.NEO4J)
        await get_embeddings_for_edges(driver, [_edge()])
        assert '(n:Entity)-[e:RELATES_TO]-(m:Entity)' in driver.queries[0]


@pytest.fixture
def _group_has_nodes(monkeypatch):
    """`get_community_clusters` skips a group with no nodes before it projects.

    The projection query is what is under test, so the node fetch is stubbed
    rather than modelled — the fake driver answers every query with no rows.
    """
    import graphiti_core.utils.maintenance.community_operations as mod

    async def _nodes(driver, group_ids):
        return [
            EntityNode(
                uuid='n1',
                name='X',
                group_id=group_ids[0],
                labels=['Entity'],
                created_at=datetime.now(timezone.utc),
            )
        ]

    monkeypatch.setattr(mod.EntityNode, 'get_by_group_ids', _nodes)


def _projection(driver) -> str:
    matches = [q for q in driver.queries if 'neighbor_uuid' in q]
    assert matches, f'the neighbour projection was never issued:\n{driver.queries}'
    return matches[0]


@pytest.mark.usefixtures('_group_has_nodes')
class TestCommunityProjectionReachesATypedEdge:
    """`community_operations` — a RAW execute_query, intercepted by nothing.

    An empty neighbour projection is not an error either: label propagation
    just runs over isolated nodes and `build_communities` produces nothing.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_a_typed_edge_counts_as_a_neighbour(self, provider):
        driver = _FakeDriver(provider)
        await get_community_clusters(driver, ['g'])
        assert pattern_selects(_projection(driver), TYPED_LABEL)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('provider', sorted(TYPED_EDGE_LABEL_PROVIDERS, key=str))
    async def test_the_endpoints_and_the_group_still_scope_it(self, provider):
        driver = _FakeDriver(provider)
        await get_community_clusters(driver, ['g'])
        projection = _projection(driver)
        assert '(n:Entity {group_id: $group_id})' in projection
        assert '(m:Entity {group_id: $group_id})' in projection

    @pytest.mark.asyncio
    async def test_neo4j_keeps_its_typed_pattern(self):
        driver = _FakeDriver(GraphProvider.NEO4J)
        await get_community_clusters(driver, ['g'])
        assert '-[e:RELATES_TO]-' in _projection(driver)


# ------------------------------------------- the staged per-driver ops module


class _FakeExecutor:
    """A QueryExecutor whose graph holds ONE typed entity edge.

    `driver.search_ops` is not wired up today — nothing calls
    `FalkorSearchOperations` — but the ledger records these three methods
    because they reintroduce BUG-62 the day the staged upstream refactor lands.
    So they are tested through their real signatures, against a graph that
    reports the types a bulk ingest actually produces.
    """

    def __init__(self, edge_types: list[str] | None = None):
        self.edge_types = [TYPED_LABEL] if edge_types is None else edge_types
        self.queries: list[str] = []

    async def execute_query(self, cypher: str, **kwargs):
        self.queries.append(cypher)
        if 'RETURN DISTINCT type(e) AS edge_type' in cypher:
            return [{'edge_type': t} for t in self.edge_types], None, None
        return [], None, None

    @property
    def search_queries(self) -> list[str]:
        return [q for q in self.queries if 'RETURN DISTINCT type(e) AS edge_type' not in q]


@pytest.fixture
def ops():
    from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations

    return FalkorSearchOperations()


class TestTheFalkorOpsModuleEdgeMethods:
    @pytest.mark.asyncio
    async def test_fulltext_asks_the_typed_index_not_only_relates_to(self, ops):
        executor = _FakeExecutor([TYPED_LABEL, 'RELATES_TO'])
        await ops.edge_fulltext_search(executor, 'detencion', SearchFilters(), limit=5)
        indexes = [q for q in executor.search_queries if 'queryRelationships' in q]
        assert len(indexes) == 2, executor.search_queries
        assert any(f"queryRelationships('{TYPED_LABEL}'" in q for q in indexes)
        assert any("queryRelationships('RELATES_TO'" in q for q in indexes)

    @pytest.mark.asyncio
    async def test_fulltext_hydrates_under_the_same_type_it_queried(self, ops):
        executor = _FakeExecutor([TYPED_LABEL])
        await ops.edge_fulltext_search(executor, 'detencion', SearchFilters(), limit=5)
        cypher = executor.search_queries[0]
        assert pattern_selects(cypher, TYPED_LABEL)

    @pytest.mark.asyncio
    async def test_fulltext_survives_a_type_with_no_index(self, ops, caplog):
        """One missing index must not take the other types down with it."""

        class _Flaky(_FakeExecutor):
            async def execute_query(self, cypher: str, **kwargs):
                if f"queryRelationships('{TYPED_LABEL}'" in cypher:
                    self.queries.append(cypher)
                    raise RuntimeError('no such index')
                return await super().execute_query(cypher, **kwargs)

        executor = _Flaky([TYPED_LABEL, 'RELATES_TO'])
        result = await ops.edge_fulltext_search(executor, 'detencion', SearchFilters(), limit=5)
        assert result == []
        assert len([q for q in executor.search_queries if 'queryRelationships' in q]) == 2

    @pytest.mark.asyncio
    async def test_similarity_matches_a_typed_edge(self, ops):
        executor = _FakeExecutor()
        await ops.edge_similarity_search(executor, [0.1] * 4, None, None, SearchFilters(), limit=5)
        cypher = executor.search_queries[0]
        assert pattern_selects(cypher, TYPED_LABEL)
        assert '(n:Entity)-[e]->(m:Entity)' in cypher
        assert 'e.fact_embedding IS NOT NULL' in cypher

    @pytest.mark.asyncio
    async def test_bfs_walks_and_rehydrates_a_typed_edge(self, ops):
        executor = _FakeExecutor()
        await ops.edge_bfs_search(executor, ['origin-1'], 2, SearchFilters(), limit=5)
        cypher = executor.search_queries[0]
        assert 'RELATES_TO|MENTIONS' not in cypher, 'the traversal cannot walk a typed edge'
        assert pattern_selects(cypher, TYPED_LABEL)
        assert '(n:Entity)-[e {uuid: rel.uuid}]-(m:Entity)' in cypher
