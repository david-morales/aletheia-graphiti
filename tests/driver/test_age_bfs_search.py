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

The fixture is built to make every predicate in the implementation LOAD-BEARING
— mutation testing on the first version showed 5 of 6 mutants surviving because
the graph had no Episodic vertex, no structural edge and no incoming edge, so
`'Entity' IN n.labels`, the `MENTIONS`/`HAS_MEMBER` exclusions and both
traversal directions were all untested. Every one of those now has a test that
fails when the predicate is removed.

Live-gated through the shared `age_driver` fixture (tests/driver/conftest.py):
a per-test throwaway graph on AGE_TEST_DSN, dropped in teardown, skipped when
the store is unreachable.
"""

from datetime import datetime, timezone

import pytest

from graphiti_core.driver.graph_operations.age_graph_operations import _cy
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


async def _structural_edge(driver, uuid, source, target, label):
    """Write a Graphiti STRUCTURAL edge (`MENTIONS` / `HAS_MEMBER`) by hand.

    Two reasons this is raw Cypher rather than `episodic_edge_save`:

    1. `episodic_edge_save` matches `(n:Entity {uuid: …})`, and on AGE a typed
       entity vertex is labelled `Persona` / `Identificacion` / … — so its MERGE
       silently writes nothing. (Separate write-path bug, its own lane; the live
       bench graph has 24 MENTIONS edges for 101 episodes and 1358 entities.)
    2. `HAS_MEMBER` has no AGE writer at all — `build_communities` would reach
       AGE through the generic fallback.

    The props mirror an entity edge's so that a MUTANT which stops excluding
    these edges hydrates them cleanly and fails on the assertion, rather than
    blowing up in `_hydrate_edge` and passing for the wrong reason.
    """
    props = {
        'uuid': uuid,
        'group_id': GROUP,
        'source_node_uuid': source,
        'target_node_uuid': target,
        'name': label,
        'fact': f'{source} {label} {target}',
        'episodes': [],
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    assignments = ', '.join(f'r.{k} = {_cy(v)}' for k, v in props.items())
    await driver.execute_query(
        f'MATCH (a), (b) WHERE a.uuid = {_cy(source)} AND b.uuid = {_cy(target)} '
        f'MERGE (a)-[r:{label} {{uuid: {_cy(uuid)}}}]->(b) SET {assignments}'
    )


async def _chain(driver):
    """The policia AGE shape, with every predicate's counter-example present.

        src  -[CONOCE]->        c            (INCOMING to the centre)
        ep   -[MENTIONS]->      c            (structural, episode -> entity)
        com  -[HAS_MEMBER]->    c            (structural, community -> entity)
        c    -[ES_IDENTIFICADO]-> n1 -[EN_PARTE]-> n2 -[OCURRE_EN]-> n3

    Every entity vertex carries a leaf ontology label, i.e. none is stored under
    the plain `:Entity` label the generic Cypher filters on. `ep` is a real
    `:Episodic` vertex and `com` a `:Community` one — neither carries a `labels`
    property, which is what `'Entity' IN n.labels` actually tests.
    """
    from graphiti_core.nodes import EpisodeType, EpisodicNode

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
    await ops.node_save(_entity(driver, 'src', 'JOSE TORRES', ['Entity', 'Persona']), driver)

    await ops.edge_save(_edge(driver, 'e1', 'c', 'n1', 'ES_IDENTIFICADO'), driver)
    await ops.edge_save(_edge(driver, 'e2', 'n1', 'n2', 'EN_PARTE'), driver)
    await ops.edge_save(_edge(driver, 'e3', 'n2', 'n3', 'OCURRE_EN'), driver)
    await ops.edge_save(_edge(driver, 'ein', 'src', 'c', 'CONOCE'), driver)

    await ops.episodic_node_save(
        EpisodicNode(
            uuid='ep',
            name='parte-20260000100001',
            group_id=GROUP,
            source=EpisodeType.text,
            source_description='parte',
            content='…',
            created_at=datetime.now(timezone.utc),
            valid_at=datetime.now(timezone.utc),
            entity_edges=[],
        ),
        driver,
    )
    await _structural_edge(driver, 'me', 'ep', 'c', 'MENTIONS')

    await driver.execute_query(
        f'MERGE (n:Community {{uuid: {_cy("com")}}}) '
        f'SET n.name = {_cy("comunidad")}, n.group_id = {_cy(GROUP)}'
    )
    await _structural_edge(driver, 'hm', 'com', 'c', 'HAS_MEMBER')


# ---------------------------------------------------------------------------
# The original defect: the legs returned nothing at all
# ---------------------------------------------------------------------------


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
    assert sorted(e.uuid for e in found) == ['e1', 'e2', 'e3', 'ein']
    # hydration must survive the uuid round-trip, not just the traversal
    assert {e.uuid: e.name for e in found}['e1'] == 'ES_IDENTIFICADO'


# ---------------------------------------------------------------------------
# Each predicate in the implementation, made load-bearing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_bfs_returns_only_entity_vertices(age_driver):
    """`'Entity' IN n.labels` must actually filter.

    Reached through an entity -> non-entity edge, which Graphiti does not write
    today (`MENTIONS` and `HAS_MEMBER` both point AT entities). The predicate is
    therefore defence-in-depth — against exactly the case the review flagged: if
    `build_communities` ever writes `:Community` vertices through the generic
    fallback, an unfiltered BFS would feed them to `EntityNode` hydration. This
    test constructs that reachability explicitly so the predicate is not free.
    """
    from graphiti_core.search.search_utils import node_bfs_search

    await _chain(age_driver)
    await _structural_edge(age_driver, 'hm2', 'n3', 'com', 'HAS_MEMBER')

    # the non-entity vertex IS now reachable outgoing from the centre
    probe, _, _ = await age_driver.execute_query(
        "MATCH (o)-[*1..4]->(n) WHERE o.uuid = 'c' AND n.uuid = 'com' RETURN count(n) AS c",
        columns=['c'],
    )
    assert probe[0]['c'] == 1

    found = await node_bfs_search(age_driver, ['c'], SearchFilters(), 4, [GROUP], 20)
    assert 'com' not in {n.uuid for n in found}
    assert sorted(n.uuid for n in found) == ['n1', 'n2', 'n3']


@pytest.mark.asyncio
async def test_node_bfs_is_directed_like_the_generic_leg(age_driver):
    """The generic node leg is `(origin)-[*1..N]->(n)`. Undirected here would
    pull in the incoming neighbour and both structural vertices, and the two
    flavours would stop agreeing."""
    from graphiti_core.search.search_utils import node_bfs_search

    await _chain(age_driver)

    # all three are 1 hop from the centre — but only on an UNDIRECTED traversal
    probe, _, _ = await age_driver.execute_query(
        "MATCH (o)-[*1..1]-(n) WHERE o.uuid = 'c' RETURN count(n) AS c", columns=['c']
    )
    assert probe[0]['c'] == 4  # n1 (out) + src, ep, com (in)

    found = await node_bfs_search(age_driver, ['c'], SearchFilters(), 1, [GROUP], 20)
    assert [n.uuid for n in found] == ['n1']


@pytest.mark.asyncio
async def test_edge_bfs_is_undirected_like_the_generic_leg(age_driver):
    """The generic edge leg is `(origin)-[*1..N]-(:Entity)` — an edge is near the
    centre whichever way it points. Directed here would silently drop every
    inbound fact."""
    from graphiti_core.search.search_utils import edge_bfs_search

    await _chain(age_driver)

    found = await edge_bfs_search(age_driver, ['c'], 1, SearchFilters(), [GROUP], 20)
    assert 'ein' in {e.uuid for e in found}
    assert sorted(e.uuid for e in found) == ['e1', 'ein']


@pytest.mark.asyncio
async def test_edge_bfs_excludes_structural_edges(age_driver):
    """`MENTIONS` and `HAS_MEMBER` are not entity-to-entity facts. Both touch the
    centre here, so dropping either exclusion changes the result."""
    from graphiti_core.search.search_utils import edge_bfs_search

    await _chain(age_driver)

    probe, _, _ = await age_driver.execute_query(
        "MATCH (o)-[r]-(n) WHERE o.uuid = 'c' AND r.uuid IN ['me', 'hm'] RETURN count(r) AS c",
        columns=['c'],
    )
    assert probe[0]['c'] == 2  # both structural edges really are adjacent

    found = await edge_bfs_search(age_driver, ['c'], 2, SearchFilters(), [GROUP], 20)
    uuids = {e.uuid for e in found}
    assert {'me', 'hm'} & uuids == set()
    # positive half: the real facts around the same centre ARE returned, so a leg
    # that returns [] unconditionally cannot pass this test either
    assert {'e1', 'e2', 'ein'} <= uuids


@pytest.mark.asyncio
async def test_reranker_excludes_structural_edges(age_driver):
    """Adjacency via a structural edge is not adjacency: `ep` and `com` touch the
    centre only through `MENTIONS` / `HAS_MEMBER`, so they must not outrank a
    real neighbour."""
    from graphiti_core.search.search_utils import node_distance_reranker

    await _chain(age_driver)

    # control: both structural vertices really are one hop from the centre, so
    # scoring them 0.0 is a decision the predicate makes, not an accident of the
    # fixture not containing them
    probe, _, _ = await age_driver.execute_query(
        "MATCH (o)-[r]-(n) WHERE o.uuid = 'c' AND n.uuid IN ['ep', 'com'] RETURN count(n) AS c",
        columns=['c'],
    )
    assert probe[0]['c'] == 2

    uuids, scores = await node_distance_reranker(age_driver, ['ep', 'com', 'n1'], 'c')
    by_uuid = dict(zip(uuids, scores, strict=True))
    assert by_uuid['n1'] == 1.0
    assert by_uuid['ep'] == 0.0
    assert by_uuid['com'] == 0.0
    assert uuids[0] == 'n1'


# ---------------------------------------------------------------------------
# Scoping and edge cases — each paired with its positive half so the assertion
# cannot pass on an implementation that returns [] for everything
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bfs_is_scoped_to_the_requested_group(age_driver):
    from graphiti_core.search.search_utils import edge_bfs_search, node_bfs_search

    await _chain(age_driver)

    # positive half: the same call in the RIGHT group must find the chain, so a
    # leg that returns [] unconditionally cannot pass this test
    assert await node_bfs_search(age_driver, ['c'], SearchFilters(), 3, [GROUP], 20) != []
    assert await edge_bfs_search(age_driver, ['c'], 3, SearchFilters(), [GROUP], 20) != []

    assert await node_bfs_search(age_driver, ['c'], SearchFilters(), 3, ['other_group'], 20) == []
    assert await edge_bfs_search(age_driver, ['c'], 3, SearchFilters(), ['other_group'], 20) == []


@pytest.mark.asyncio
async def test_edge_bfs_honours_the_edge_types_filter(age_driver):
    """`explore_node(edge_types=[...])` must not silently return every type."""
    from graphiti_core.search.search_utils import edge_bfs_search

    await _chain(age_driver)

    unfiltered = await edge_bfs_search(age_driver, ['c'], 3, SearchFilters(), [GROUP], 20)
    assert len(unfiltered) > 1

    found = await edge_bfs_search(
        age_driver, ['c'], 3, SearchFilters(edge_types=['EN_PARTE']), [GROUP], 20
    )
    assert [e.uuid for e in found] == ['e2']


@pytest.mark.asyncio
async def test_bfs_respects_depth(age_driver):
    from graphiti_core.search.search_utils import node_bfs_search

    await _chain(age_driver)

    deep = await node_bfs_search(age_driver, ['c'], SearchFilters(), 3, [GROUP], 20)
    shallow = await node_bfs_search(age_driver, ['c'], SearchFilters(), 1, [GROUP], 20)
    assert sorted(n.uuid for n in deep) == ['n1', 'n2', 'n3']
    assert sorted(n.uuid for n in shallow) == ['n1']


@pytest.mark.asyncio
async def test_bfs_with_no_origins_returns_empty(age_driver):
    from graphiti_core.search.search_utils import edge_bfs_search, node_bfs_search

    await _chain(age_driver)

    # positive half first: with an origin the same call is non-empty
    assert await node_bfs_search(age_driver, ['c'], SearchFilters(), 2, [GROUP], 20) != []
    assert await edge_bfs_search(age_driver, ['c'], 2, SearchFilters(), [GROUP], 20) != []

    assert await node_bfs_search(age_driver, [], SearchFilters(), 2, [GROUP], 20) == []
    assert await edge_bfs_search(age_driver, None, 2, SearchFilters(), [GROUP], 20) == []


# ---------------------------------------------------------------------------
# Reranker contract + the end-to-end seam
# ---------------------------------------------------------------------------


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

    # depth 2 from the centre: n1, n2 outgoing; the edges on those paths plus the
    # inbound CONOCE (edge BFS is undirected), and no structural edge.
    assert sorted(n.uuid for n in results.nodes) == ['n1', 'n2']
    assert sorted(e.uuid for e in results.edges) == ['e1', 'e2', 'ein']
    assert results.nodes[0].uuid == 'n1'  # node_distance rerank: adjacent first
