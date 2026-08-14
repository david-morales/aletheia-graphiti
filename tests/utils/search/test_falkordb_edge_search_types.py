"""Regression tests for BUG-62 — FalkorDB edge search returns nothing.

The FalkorDB bulk writer MERGEs every entity edge under the edge's own ``name``
as the relationship type (``graphiti_core/utils/bulk_utils.py``: ``DETIENE``,
``INTERVIENE_AGENTE``, ``WORKS_AT``, …), while the single-edge writer still uses
``RELATES_TO``.  Edge search used to look the edges back up under a hardcoded
``RELATES_TO``:

* ``edge_fulltext_search`` defaulted ``edge_types`` to ``['RELATES_TO']``, so it
  only ever queried the ``RELATES_TO`` fulltext index — empty in a bulk-ingested
  graph.  Creating per-type fulltext indices did **not** help, because the search
  never asked for them.
* ``edge_similarity_search`` pinned ``-[e:RELATES_TO]->`` into its MATCH with no
  FalkorDB branch at all, so the cosine leg was dead for the same reason.

Both legs feed the default ``combined`` search mode, which is why a default
search on the FalkorDB flavour returned nodes but zero facts.

These tests assert on the **emitted Cypher** at the driver boundary, so they run
offline against no database.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    edge_fulltext_search,
    edge_similarity_search,
    resolve_entity_edge_types,
)

pytestmark = pytest.mark.asyncio

# The relationship types a bulk-ingested police-report graph actually holds:
# custom fact-carrying types plus the structural ones Graphiti always writes.
PRESENT_RELATIONSHIP_TYPES = [
    'DETIENE',
    'INTERVIENE_AGENTE',
    'MENTIONS',
    'HAS_EPISODE',
    'NEXT_EPISODE',
    'RELATES_TO',
]

FACT_CARRYING_TYPES = {'DETIENE', 'INTERVIENE_AGENTE', 'RELATES_TO'}
STRUCTURAL_TYPES = {'MENTIONS', 'HAS_EPISODE', 'NEXT_EPISODE'}


def _edge_record(uuid: str, name: str, fact: str) -> dict:
    """A record shaped like ``get_entity_edge_return_query`` returns."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        'uuid': uuid,
        'source_node_uuid': f'src-{uuid}',
        'target_node_uuid': f'tgt-{uuid}',
        'group_id': 'policia',
        'created_at': now,
        'name': name,
        'fact': fact,
        'episodes': ['ep-1'],
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
        'attributes': {},
    }


# Edges as the bulk writer left them: keyed by the relationship type they were
# MERGEd under, never under RELATES_TO.
WRITTEN_EDGES = {
    'DETIENE': [_edge_record('edge-detiene-1', 'DETIENE', 'El agente detiene al sospechoso')],
    'INTERVIENE_AGENTE': [
        _edge_record('edge-interviene-1', 'INTERVIENE_AGENTE', 'El agente interviene en el robo')
    ],
}


class FakeFalkorQueryLog:
    """Records every Cypher string a search emits, and answers like FalkorDB."""

    def __init__(self, present_types: list[str], written: dict[str, list[dict]]):
        self.present_types = present_types
        self.written = written
        self.queries: list[str] = []

    async def execute_query(self, cypher: str, **kwargs):
        self.queries.append(cypher)

        if 'db.relationshipTypes' in cypher:
            return [{'relationshipType': t} for t in self.present_types], None, None

        if 'queryRelationships' in cypher:
            edge_type = self._queried_type(cypher)
            if edge_type not in self.present_types:
                # FalkorDB raises when the index does not exist.
                raise RuntimeError(f'no such index {edge_type}')
            return list(self.written.get(edge_type, [])), None, None

        # Similarity path: an untyped Entity->Entity match reaches every entity edge.
        if 'MATCH (n:Entity)-[e]->(m:Entity)' in cypher:
            return [r for records in self.written.values() for r in records], None, None

        # A typed similarity match only ever reaches RELATES_TO — nothing was
        # written under it, so it comes back empty. This is the bug, reproduced.
        if 'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)' in cypher:
            return [], None, None

        return [], None, None

    @staticmethod
    def _queried_type(cypher: str) -> str:
        marker = "queryRelationships('"
        start = cypher.index(marker) + len(marker)
        return cypher[start : cypher.index("'", start)]

    @property
    def fulltext_queries(self) -> list[str]:
        return [q for q in self.queries if 'queryRelationships' in q]

    @property
    def queried_edge_types(self) -> set[str]:
        return {self._queried_type(q) for q in self.fulltext_queries}


def _falkor_driver(log: FakeFalkorQueryLog):
    """A real FalkorDriver with its I/O replaced by the query log."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    with patch.object(FalkorDriver, 'build_indices_and_constraints', AsyncMock()):
        driver = FalkorDriver(falkor_db=MagicMock())
    driver.execute_query = log.execute_query  # type: ignore[method-assign]
    return driver


def _plain_driver(provider: GraphProvider):
    """A minimal non-FalkorDB driver stand-in for the parity arm."""
    driver = MagicMock()
    driver.provider = provider
    driver.search_interface = None
    driver.fulltext_syntax = ''
    driver.default_group_id = ''
    return driver


async def test_resolve_entity_edge_types_reads_the_types_present_in_the_graph():
    """The type list comes from the database, never from a constant."""
    log = FakeFalkorQueryLog(PRESENT_RELATIONSHIP_TYPES, WRITTEN_EDGES)
    driver = _falkor_driver(log)

    resolved = await resolve_entity_edge_types(driver)

    assert set(resolved) == FACT_CARRYING_TYPES
    # Structural types carry no facts and have no fulltext index — never search them.
    assert not STRUCTURAL_TYPES & set(resolved)


async def test_resolve_entity_edge_types_accepts_positional_records():
    """FalkorDB has returned db.relationshipTypes() rows as bare sequences."""
    log = FakeFalkorQueryLog(PRESENT_RELATIONSHIP_TYPES, WRITTEN_EDGES)
    driver = _falkor_driver(log)

    async def positional(cypher: str, **kwargs):
        if 'db.relationshipTypes' in cypher:
            return [[t] for t in PRESENT_RELATIONSHIP_TYPES], None, None
        return await log.execute_query(cypher, **kwargs)

    driver.execute_query = positional  # type: ignore[method-assign]

    assert set(await resolve_entity_edge_types(driver)) == FACT_CARRYING_TYPES


async def test_resolve_entity_edge_types_falls_back_when_enumeration_fails():
    """A driver that cannot enumerate degrades to the historical default."""
    driver = _falkor_driver(FakeFalkorQueryLog([], {}))

    async def boom(cypher: str, **kwargs):
        raise RuntimeError('db.relationshipTypes unsupported')

    driver.execute_query = boom  # type: ignore[method-assign]

    assert await resolve_entity_edge_types(driver) == ['RELATES_TO']


async def test_resolve_entity_edge_types_is_relates_to_for_other_providers():
    """Non-FalkorDB providers keep writing RELATES_TO — no enumeration, no query."""
    driver = _plain_driver(GraphProvider.NEO4J)

    assert await resolve_entity_edge_types(driver) == ['RELATES_TO']
    driver.execute_query.assert_not_called()


async def test_edge_fulltext_search_queries_the_types_actually_written():
    """BUG-62: the search asked for RELATES_TO, which the bulk writer never writes."""
    log = FakeFalkorQueryLog(PRESENT_RELATIONSHIP_TYPES, WRITTEN_EDGES)
    driver = _falkor_driver(log)

    edges = await edge_fulltext_search(driver, 'robo', SearchFilters(), ['policia'])

    # The emitted queries must target the present fact-carrying types...
    assert 'DETIENE' in log.queried_edge_types
    assert 'INTERVIENE_AGENTE' in log.queried_edge_types
    # ...and must not be the single RELATES_TO lookup that returned nothing.
    assert log.queried_edge_types != {'RELATES_TO'}
    # Structural relationships have no fulltext index; querying them only logs errors.
    assert not STRUCTURAL_TYPES & log.queried_edge_types

    # And the written edges come back.
    assert {e.uuid for e in edges} == {'edge-detiene-1', 'edge-interviene-1'}


async def test_edge_fulltext_search_honours_explicitly_requested_types():
    """An explicit edge_types list is still respected — no enumeration."""
    log = FakeFalkorQueryLog(PRESENT_RELATIONSHIP_TYPES, WRITTEN_EDGES)
    driver = _falkor_driver(log)

    edges = await edge_fulltext_search(
        driver, 'robo', SearchFilters(), ['policia'], edge_types=['DETIENE']
    )

    assert log.queried_edge_types == {'DETIENE'}
    assert not any('db.relationshipTypes' in q for q in log.queries)
    assert {e.uuid for e in edges} == {'edge-detiene-1'}


async def test_edge_fulltext_search_sanitizes_caller_supplied_edge_types():
    """edge_types is interpolated into Cypher — it must not carry an injection."""
    log = FakeFalkorQueryLog(['DETIENE'], WRITTEN_EDGES)
    driver = _falkor_driver(log)

    await edge_fulltext_search(
        driver,
        'robo',
        SearchFilters(),
        ['policia'],
        edge_types=["DETIENE') YIELD relationship AS rel MATCH (x) DETACH DELETE x //"],
    )

    for query in log.fulltext_queries:
        assert 'DETACH DELETE' not in query


async def test_edge_similarity_search_reaches_typed_edges():
    """BUG-62, cosine leg: a pinned :RELATES_TO match cannot see a typed edge."""
    log = FakeFalkorQueryLog(PRESENT_RELATIONSHIP_TYPES, WRITTEN_EDGES)
    driver = _falkor_driver(log)

    edges = await edge_similarity_search(
        driver, [0.1] * 8, None, None, SearchFilters(), ['policia']
    )

    emitted = '\n'.join(log.queries)
    assert '-[e:RELATES_TO]->' not in emitted
    assert {e.uuid for e in edges} == {'edge-detiene-1', 'edge-interviene-1'}


async def test_edge_similarity_search_keeps_relates_to_for_other_providers():
    """Only FalkorDB writes typed relationships — do not change the others."""
    driver = _plain_driver(GraphProvider.NEO4J)
    driver.execute_query = AsyncMock(return_value=([], None, None))

    await edge_similarity_search(driver, [0.1] * 8, None, None, SearchFilters(), ['policia'])

    emitted = driver.execute_query.call_args[0][0]
    assert '-[e:RELATES_TO]->' in emitted


async def test_combined_mode_edge_leg_returns_facts():
    """The reported symptom: default combined search gave nodes but zero facts.

    COMBINED_HYBRID_SEARCH_RRF's edge_config is bm25 + cosine_similarity with
    edge_types left at None — exactly the two legs fixed here, and exactly the
    edge_config EDGE_HYBRID_SEARCH_RRF ('edges' mode) uses too.
    """
    from graphiti_core.search.search import search
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

    assert COMBINED_HYBRID_SEARCH_RRF.edge_config is not None
    assert COMBINED_HYBRID_SEARCH_RRF.edge_config.edge_types is None

    log = FakeFalkorQueryLog(PRESENT_RELATIONSHIP_TYPES, WRITTEN_EDGES)
    driver = _falkor_driver(log)

    clients = MagicMock()
    clients.driver = driver
    clients.embedder = AsyncMock()
    clients.cross_encoder = AsyncMock()

    results = await search(
        clients,
        'robo',
        ['policia'],
        COMBINED_HYBRID_SEARCH_RRF,
        SearchFilters(),
        query_vector=[0.1] * 8,
    )

    assert {e.uuid for e in results.edges} == {'edge-detiene-1', 'edge-interviene-1'}


async def test_live_probe_bm25_fallback_never_calls_the_embedder():
    """verify_live.py runs without API keys, so it must not reach for an embedder.

    Its cosine leg borrows a fact_embedding from the graph; when the graph has
    none to borrow it strips the vector-dependent methods instead. search() then
    takes its zero-vector path — if a method survived the strip, it would call
    `embedder.create` on the None this probe passes and die at runtime.
    """
    import verify_live
    from graphiti_core.search.search import search
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

    original_methods = list(COMBINED_HYBRID_SEARCH_RRF.edge_config.search_methods)

    log = FakeFalkorQueryLog(PRESENT_RELATIONSHIP_TYPES, WRITTEN_EDGES)
    clients = MagicMock()
    clients.driver = _falkor_driver(log)
    clients.embedder = None  # exactly what the probe passes with no key
    clients.cross_encoder = AsyncMock()

    results = await search(
        clients,
        'robo',
        ['policia'],
        verify_live._bm25_only(COMBINED_HYBRID_SEARCH_RRF),
        SearchFilters(),
    )

    # The bm25 leg still finds the typed edges.
    assert {e.uuid for e in results.edges} == {'edge-detiene-1', 'edge-interviene-1'}
    # And the shared module-level recipe is untouched for every later caller.
    assert COMBINED_HYBRID_SEARCH_RRF.edge_config.search_methods == original_methods


async def test_falkordb_and_default_arms_return_the_same_edges():
    """Two-arm parity: the same written edges are found on either flavour.

    The arms differ only in how the writer stored the edges — typed relationship
    names on FalkorDB, RELATES_TO everywhere else. Search must not care.
    """
    falkor_log = FakeFalkorQueryLog(PRESENT_RELATIONSHIP_TYPES, WRITTEN_EDGES)
    falkor_driver = _falkor_driver(falkor_log)
    falkor_edges = await edge_fulltext_search(falkor_driver, 'robo', SearchFilters(), ['policia'])

    # Default arm: one RELATES_TO index holding the same two edges.
    default_driver = _plain_driver(GraphProvider.NEO4J)
    all_records = [r for records in WRITTEN_EDGES.values() for r in records]
    default_driver.execute_query = AsyncMock(return_value=(all_records, None, None))
    default_edges = await edge_fulltext_search(default_driver, 'robo', SearchFilters(), ['policia'])

    assert {e.uuid for e in falkor_edges} == {e.uuid for e in default_edges}
    assert {e.fact for e in falkor_edges} == {e.fact for e in default_edges}
    assert falkor_edges
