"""BUG-87: an entity edge stored under a TYPED label must still be reachable.

The bug
-------
BUG-62 fixed the search legs, but the "an entity edge is a ``RELATES_TO``"
assumption survived in every sibling path:

* ``edges.py`` — ``Edge.delete`` / ``Edge.delete_by_uuids`` matched
  ``MENTIONS|RELATES_TO|HAS_MEMBER``, so a bulk-written ``DETIENE`` edge could
  not be deleted through the model API at all;
* ``search_utils.get_relevant_edges`` / ``get_edge_invalidation_candidates`` —
  the dedup and temporal-invalidation candidate queries pinned
  ``-[e:RELATES_TO]-``, so a typed edge was never OFFERED as a candidate:
  duplicates never merged and a superseded fact was never invalidated;
* ``driver/falkordb/operations/search_ops.py`` — all three edge methods carried
  the same constant, plus a fulltext leg that asked only the ``RELATES_TO``
  index.

Every one of these fails by matching zero rows, which is not an error. The
consequence class is therefore silent: maintenance and temporal invalidation
no-op on every bulk-ingested graph.

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
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    get_edge_invalidation_candidates,
    get_relevant_edges,
)

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
