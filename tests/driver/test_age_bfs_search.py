"""Regression: BFS traversal on the AGE flavour returned nothing, silently.

Bug (2026-08-03, q5 residual F3): `explore_node` on the AGE connector reported
"1 nodes, 0 edges" for a hub with 18 proven edges, with no error raised.

Root cause: `AGESearch` never overrode `node_bfs_search` / `edge_bfs_search` /
`node_distance_reranker`, so the base `SearchInterface` raised
`NotImplementedError`, `search_utils` caught it and fell through to the
PROVIDER-GENERIC Cypher — which hardcodes the `:Entity` node label and the
`:RELATES_TO` edge type. On AGE neither exists: `_node_label()` stores every
entity vertex under its LEAF ontology class (`Persona`, `Identificacion`, …)
and `_edge_label()` stores every entity edge under its typed relationship name
(`ES_IDENTIFICADO`, `EN_PARTE`, …). The generic queries are valid AGE Cypher,
match zero rows and raise nothing — silent-wrong.

Each test asserts BOTH halves: the graph really is traversable (unlabelled
probe finds the neighbours) AND the search leg returns them. Without the
second half a future label regression would look the same as a data problem.

Live-gated through the shared `age_driver` fixture (tests/driver/conftest.py):
a per-test throwaway graph on AGE_TEST_DSN, dropped in teardown, skipped when
the store is unreachable.
"""

from datetime import datetime, timezone

import pytest

from graphiti_core.driver.search_interface.age_search import AGESearch
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.search.search_filters import SearchFilters

GROUP = 'agebfs'


@pytest.mark.parametrize('method', ['node_bfs_search', 'edge_bfs_search', 'node_distance_reranker'])
def test_age_search_overrides_the_traversal_methods(method):
    """Offline guard: without an override the caller falls through to the generic
    `:Entity` / `:RELATES_TO` Cypher, which matches nothing on AGE and raises
    nothing. Runs with no live backend so every CI run gates it."""
    assert getattr(AGESearch, method) is not getattr(SearchInterface, method)


def _entity(driver, uuid, name, labels):
    from graphiti_core.nodes import EntityNode

    n = EntityNode(
        uuid=uuid,
        name=name,
        group_id=GROUP,
        labels=list(labels),
        created_at=datetime.now(timezone.utc),
        summary=name,
        attributes={},
    )
    n.name_embedding = [0.1] * driver.embedding_dim
    return n


def _edge(driver, uuid, source, target, name):
    from graphiti_core.edges import EntityEdge

    e = EntityEdge(
        uuid=uuid,
        source_node_uuid=source,
        target_node_uuid=target,
        group_id=GROUP,
        name=name,
        fact=f'{source} {name} {target}',
        created_at=datetime.now(timezone.utc),
    )
    e.fact_embedding = [0.1] * driver.embedding_dim
    return e


async def _chain(driver):
    """centre -[ES_IDENTIFICADO]-> ident -[EN_PARTE]-> parte -[OCURRE_EN]-> ubicacion.

    Every vertex carries a leaf ontology label, i.e. none of them is stored
    under the plain `:Entity` label the generic Cypher filters on — the exact
    shape the policia AGE graphs have.
    """
    ops = driver.graph_operations_interface
    await ops.node_save(_entity(driver, 'c', 'KHADIJA DAOUD', ['Entity', 'Persona']), driver)
    await ops.node_save(
        _entity(
            driver, 'n1', 'Identificacion 1', ['Entity', 'RolInvolucramiento', 'Identificacion']
        ),
        driver,
    )
    await ops.node_save(_entity(driver, 'n2', 'Parte 1', ['Entity', 'ParteDeIntervencion']), driver)
    await ops.node_save(_entity(driver, 'n3', 'LOGRONO', ['Entity', 'Ubicacion']), driver)
    await ops.edge_save(_edge(driver, 'e1', 'c', 'n1', 'ES_IDENTIFICADO'), driver)
    await ops.edge_save(_edge(driver, 'e2', 'n1', 'n2', 'EN_PARTE'), driver)
    await ops.edge_save(_edge(driver, 'e3', 'n2', 'n3', 'OCURRE_EN'), driver)


@pytest.mark.asyncio
async def test_node_bfs_search_finds_typed_neighbours(age_driver):
    from graphiti_core.search.search_utils import node_bfs_search

    await _chain(age_driver)

    # the graph IS traversable — this is the control the bug's symptom lacked
    probe, _, _ = await age_driver.execute_query(
        "MATCH (o)-[*1..3]->(n) WHERE o.uuid = 'c' RETURN count(n) AS c", columns=['c']
    )
    assert probe[0]['c'] == 3

    found = await node_bfs_search(age_driver, ['c'], SearchFilters(), 3, [GROUP], 20)
    assert sorted(n.uuid for n in found) == ['n1', 'n2', 'n3']


@pytest.mark.asyncio
async def test_edge_bfs_search_finds_typed_edges(age_driver):
    from graphiti_core.search.search_utils import edge_bfs_search

    await _chain(age_driver)

    found = await edge_bfs_search(age_driver, ['c'], 3, SearchFilters(), [GROUP], 20)
    assert sorted(e.uuid for e in found) == ['e1', 'e2', 'e3']
    # hydration must survive the uuid round-trip, not just the traversal
    assert {e.uuid: e.name for e in found}['e1'] == 'ES_IDENTIFICADO'


@pytest.mark.asyncio
async def test_bfs_is_scoped_to_the_requested_group(age_driver):
    from graphiti_core.search.search_utils import edge_bfs_search, node_bfs_search

    await _chain(age_driver)

    assert await node_bfs_search(age_driver, ['c'], SearchFilters(), 3, ['other_group'], 20) == []
    assert await edge_bfs_search(age_driver, ['c'], 3, SearchFilters(), ['other_group'], 20) == []


@pytest.mark.asyncio
async def test_edge_bfs_honours_the_edge_types_filter(age_driver):
    """`explore_node(edge_types=[...])` must not silently return every type."""
    from graphiti_core.search.search_utils import edge_bfs_search

    await _chain(age_driver)

    found = await edge_bfs_search(
        age_driver, ['c'], 3, SearchFilters(edge_types=['EN_PARTE']), [GROUP], 20
    )
    assert [e.uuid for e in found] == ['e2']


@pytest.mark.asyncio
async def test_bfs_respects_depth(age_driver):
    from graphiti_core.search.search_utils import node_bfs_search

    await _chain(age_driver)

    assert sorted(
        n.uuid for n in await node_bfs_search(age_driver, ['c'], SearchFilters(), 1, [GROUP], 20)
    ) == ['n1']


@pytest.mark.asyncio
async def test_bfs_with_no_origins_returns_empty(age_driver):
    from graphiti_core.search.search_utils import edge_bfs_search, node_bfs_search

    await _chain(age_driver)

    assert await node_bfs_search(age_driver, [], SearchFilters(), 2, [GROUP], 20) == []
    assert await edge_bfs_search(age_driver, None, 2, SearchFilters(), [GROUP], 20) == []


@pytest.mark.asyncio
async def test_node_distance_reranker_ranks_adjacent_above_distant(age_driver):
    """The generic reranker matches `(:Entity)-[:RELATES_TO]-(:Entity)`, so on AGE
    it scored EVERY candidate 0.0 — explore_node's "ranked by proximity" was a
    no-op, and truncating to `limit` dropped arbitrary results."""
    from graphiti_core.search.search_utils import node_distance_reranker

    await _chain(age_driver)

    uuids, scores = await node_distance_reranker(age_driver, ['n3', 'n1'], 'c')
    assert uuids == ['n1', 'n3']  # adjacent first
    assert scores[0] > scores[1]


@pytest.mark.asyncio
async def test_node_distance_reranker_keeps_the_center_node(age_driver):
    from graphiti_core.search.search_utils import node_distance_reranker

    await _chain(age_driver)

    uuids, _ = await node_distance_reranker(age_driver, ['n3', 'c', 'n1'], 'c')
    assert uuids[0] == 'c'
    assert uuids[1] == 'n1'


@pytest.mark.asyncio
async def test_explore_node_search_config_returns_the_neighbourhood(age_driver_embedding_matched):
    """End-to-end at the search seam: the exact call `explore_node` makes.

    Empty query (a uuid was given, so there is no text to search), bm25 +
    cosine + bfs on both nodes and edges, node_distance rerankers, BFS origin =
    the center. This is what returned "1 nodes, 0 edges" on AGE while the same
    call returned 20/20 on FalkorDB: the two shadow-table legs contribute
    nothing without a query string, so the whole result rides on BFS.
    """
    from graphiti_core.graphiti_types import GraphitiClients
    from graphiti_core.search.search import search
    from graphiti_core.search.search_config import (
        EdgeReranker,
        EdgeSearchConfig,
        EdgeSearchMethod,
        NodeReranker,
        NodeSearchConfig,
        NodeSearchMethod,
        SearchConfig,
    )

    age_driver = age_driver_embedding_matched
    await _chain(age_driver)

    explore_config = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[
                EdgeSearchMethod.bm25,
                EdgeSearchMethod.cosine_similarity,
                EdgeSearchMethod.bfs,
            ],
            reranker=EdgeReranker.node_distance,
            bfs_max_depth=2,
        ),
        node_config=NodeSearchConfig(
            search_methods=[
                NodeSearchMethod.bm25,
                NodeSearchMethod.cosine_similarity,
                NodeSearchMethod.bfs,
            ],
            reranker=NodeReranker.node_distance,
            bfs_max_depth=2,
        ),
        limit=20,
    )
    # model_construct: the LLM/embedder/cross-encoder are unreachable on this
    # path (an empty query skips embedding, node_distance needs no encoder), and
    # stubbing them would only assert that they stay unreachable.
    clients = GraphitiClients.model_construct(
        driver=age_driver, llm_client=None, embedder=None, cross_encoder=None, tracer=None
    )

    results = await search(
        clients,
        '',
        [GROUP],
        explore_config,
        SearchFilters(),
        center_node_uuid='c',
        bfs_origin_node_uuids=['c'],
    )

    # depth 2 from the centre: n1, n2 and the two edges on those paths (e3 is
    # the third hop). Pre-fix this call returned nothing but noise.
    assert sorted(n.uuid for n in results.nodes) == ['n1', 'n2']
    assert sorted(e.uuid for e in results.edges) == ['e1', 'e2']
    assert results.nodes[0].uuid == 'n1'  # node_distance rerank: adjacent first
