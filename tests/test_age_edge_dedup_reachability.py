"""BUG-104: on AGE, edge dedup and invalidation must reach a typed vertex.

The bug
-------
``search_utils.get_relevant_edges`` and ``get_edge_invalidation_candidates``
anchor their MATCH on the NODE label — ``(n:Entity {uuid: …})`` /
``(n:Entity)-[e]->(m:Entity)``. AGE stores exactly one label per vertex and
``age_graph_operations._node_label()`` makes it the LEAF ontology class
(``Persona``, ``TipoHecho``, …); ``:Entity`` is only the fallback for a node
with no ontology class at all. So on AGE both functions match a small, arbitrary
minority of the graph.

BUG-87 fixed the RELATIONSHIP half of these same two queries (the pinned
``:RELATES_TO``). It does not help this arm: the failure here is the NODE label.

Measured on the live AGE bed (``policia_partes_real``, 2026-08-18, read-only):

    vertices carrying the `labels` property        554
    vertices with 'Entity' IN n.labels             554
    vertices matching (n:Entity)                    42     <- the pattern's reach
    vertices matching (n:Episodic)                  46
    total vertices                                 600

    dedup candidates for one real edge
      MATCH (n:Entity {uuid: SRC})-[e {group_id: G}]-(m:Entity {uuid: TGT})     0
      MATCH (n {uuid: SRC})-[e {group_id: G}]-(m {uuid: TGT})  + labels guard    1

    invalidation candidates for the same edge
      MATCH (n:Entity)-[e {group_id: G}]->(m:Entity) WHERE n.uuid IN [...] ...   0
      same, label-free + labels guard                                           19

    whole graph
      MATCH (n:Entity)-[e]-(m:Entity)                                            8
      MATCH (n)-[e]-(m) + labels guard                                        1548   (= 774 edges x 2)

Why the fix is an AGE OVERRIDE, not a label-aware shared pattern
---------------------------------------------------------------
The shared query cannot run on AGE even with the node label corrected. Its score
projection is ``get_vector_cosine_func_query(..., AGE)``, which falls through to
``vector.similarity.cosine(a, b)`` — a Neo4j function whose dots AGE parses as
property indirection:

    MATCH (n)-[e]-(m) WHERE e.fact_embedding IS NOT NULL
    WITH e, vector.similarity.cosine(e.fact_embedding, e.fact_embedding) AS score
    RETURN count(score) AS c
    -> ERROR: PostgresSyntaxError: invalid indirection syntax

So widening the node label in shared code would trade a silent zero-row answer
for a hard parse error, AND would push AGE's single-label knowledge into a code
path neo4j/kuzu/neptune/falkordb share. AGE instead owns both functions through
``search_interface`` — the seam it already uses for its reranker and BFS
overrides — and answers from the uuid-keyed pgvector shadow tables, which carry
no labels at all and so cannot have this bug in any form.

What this does NOT fix — read before quoting it
-----------------------------------------------
Neither function has a caller inside ``graphiti_core`` today. Upstream commit
3efe085 ("OpenSearch updates", #906, 2025-09-14) took them out of
``resolve_extracted_edges`` and replaced them with ``EntityEdge.get_between_nodes``
plus two hybrid ``search`` calls. They remain exported public API — and they
carried the defect — but repairing them does not, on its own, change what an AGE
ingest does. The LIVE AGE dedup path is:

  * ``EntityEdge.get_between_nodes`` -> ``AGEGraphOperations.edge_get_between_nodes``,
    which is already label-free (``MATCH (a)-[r]->(b) WHERE a.uuid = … AND
    b.uuid = …``) and does NOT have this bug; and
  * ``search(..., EDGE_HYBRID_SEARCH_RRF, SearchFilters(edge_uuids=[…]))`` —
    where ``edge_uuids`` is one of the SearchFilters the AGE legs drop, so the
    "edges between these two nodes" restriction is not applied on that arm at
    all. That is a separate defect in a different layer and is NOT fixed here.

How these tests prove it
------------------------
* the SEAM classes drive the real ``search_utils`` functions against a fake
  driver and assert the interface is consulted, that a ``NotImplementedError``
  falls back, and that the providers which never had this bug emit exactly the
  Cypher they emitted before;
* the AGE classes drive the real ``AGESearch`` methods and evaluate the SQL
  predicates the implementation embeds — imported as constants, not scraped —
  against a model of rows, so the assertions are about what the query WOULD
  SELECT rather than about the presence of a string.
"""

import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    DEFAULT_MIN_SCORE,
    get_edge_invalidation_candidates,
    get_relevant_edges,
)

# The live-bed reach of the two node patterns, from the module docstring. Kept as
# named constants so the numbers in a failure message are the measured ones.
LIVE_ENTITY_LABEL_VERTICES = 42
LIVE_ENTITY_VERTICES = 554


def _edge(
    uuid: str = 'edge-1',
    source: str = 'src',
    target: str = 'tgt',
    group_id: str = 'g',
    embedding: list[float] | None = None,
) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        source_node_uuid=source,
        target_node_uuid=target,
        group_id=group_id,
        name='TIPIFICADO_COMO',
        fact='X tipificado como Y',
        fact_embedding=[0.1, 0.2, 0.3, 0.4] if embedding is None else embedding,
        created_at=datetime.now(timezone.utc),
    )


# ------------------------------------------------------------------- the seam


class _FakeDriver:
    """Captures the Cypher a core function emits, and answers nothing useful."""

    def __init__(self, provider: GraphProvider, search_interface=None):
        self.provider = provider
        self.graph_operations_interface = None
        self.search_interface = search_interface
        self.queries: list[str] = []
        self.fulltext_syntax = ''

    async def execute_query(self, cypher: str, **kwargs):
        self.queries.append(cypher)
        return [], None, None


class _RecordingInterface(SearchInterface):
    """A SearchInterface that implements the two methods and records the call."""

    calls: list = []
    answer: list = []

    async def get_relevant_edges(self, driver, edges, search_filter, min_score=0.6, limit=10):
        self.calls.append(('get_relevant_edges', edges, search_filter, min_score, limit))
        return self.answer

    async def get_edge_invalidation_candidates(
        self, driver, edges, search_filter, min_score=0.6, limit=10
    ):
        self.calls.append(
            ('get_edge_invalidation_candidates', edges, search_filter, min_score, limit)
        )
        return self.answer


class _AbstainingInterface(SearchInterface):
    """A SearchInterface that declines both — the documented fall-through."""


class TestTheSeamIsConsulted:
    """A driver that owns these queries must be asked before the generic Cypher."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'fn,name',
        [
            (get_relevant_edges, 'get_relevant_edges'),
            (get_edge_invalidation_candidates, 'get_edge_invalidation_candidates'),
        ],
    )
    async def test_the_interface_answers_instead_of_the_generic_cypher(self, fn, name):
        interface = _RecordingInterface()
        interface.answer = [[_edge(uuid='existing-1')]]
        driver = _FakeDriver(GraphProvider.AGE, search_interface=interface)

        result = await fn(driver, [_edge()], SearchFilters(), min_score=0.5, limit=7)

        assert [c[0] for c in interface.calls] == [name]
        assert driver.queries == [], 'the generic Cypher was emitted anyway'
        assert [[e.uuid for e in lst] for lst in result] == [['existing-1']]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'fn,name',
        [
            (get_relevant_edges, 'get_relevant_edges'),
            (get_edge_invalidation_candidates, 'get_edge_invalidation_candidates'),
        ],
    )
    async def test_the_caller_arguments_reach_the_interface(self, fn, name):
        interface = _RecordingInterface()
        driver = _FakeDriver(GraphProvider.AGE, search_interface=interface)
        edges = [_edge()]
        search_filter = SearchFilters()

        await fn(driver, edges, search_filter, min_score=0.42, limit=3)

        _, got_edges, got_filter, got_min_score, got_limit = interface.calls[0]
        assert got_edges is edges
        assert got_filter is search_filter
        assert got_min_score == 0.42
        assert got_limit == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'fn', [get_relevant_edges, get_edge_invalidation_candidates]
    )
    async def test_an_abstaining_interface_falls_through_to_the_generic_cypher(self, fn):
        """`NotImplementedError` is the seam's documented "not mine" answer."""
        driver = _FakeDriver(GraphProvider.NEO4J, search_interface=_AbstainingInterface())
        await fn(driver, [_edge()], SearchFilters())
        assert len(driver.queries) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'fn', [get_relevant_edges, get_edge_invalidation_candidates]
    )
    async def test_an_empty_edge_list_never_touches_the_driver(self, fn):
        interface = _RecordingInterface()
        driver = _FakeDriver(GraphProvider.AGE, search_interface=interface)
        assert await fn(driver, [], SearchFilters()) == []
        assert interface.calls == []
        assert driver.queries == []


class TestTheProvidersThatNeverHadThisBugAreUntouched:
    """Neo4j/Kuzu/Neptune/FalkorDB set no `search_interface` — nothing intercepts.

    These pin the generic Cypher byte-for-byte at the two places BUG-104 could
    plausibly have been "fixed" in shared code: the endpoint node patterns.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'provider',
        [
            GraphProvider.NEO4J,
            GraphProvider.KUZU,
            GraphProvider.NEPTUNE,
            GraphProvider.FALKORDB,
        ],
    )
    async def test_dedup_keeps_the_entity_anchored_endpoints(self, provider):
        driver = _FakeDriver(provider)
        await get_relevant_edges(driver, [_edge()], SearchFilters())
        cypher = driver.queries[0]
        assert '(n:Entity {uuid: edge.source_node_uuid})' in cypher
        assert '(m:Entity {uuid: edge.target_node_uuid})' in cypher

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'provider',
        [
            GraphProvider.NEO4J,
            GraphProvider.KUZU,
            GraphProvider.NEPTUNE,
            GraphProvider.FALKORDB,
        ],
    )
    async def test_invalidation_keeps_the_entity_anchored_endpoints(self, provider):
        driver = _FakeDriver(provider)
        await get_edge_invalidation_candidates(driver, [_edge()], SearchFilters())
        cypher = driver.queries[0]
        assert '(n:Entity)' in cypher
        assert '(m:Entity)' in cypher

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'fn', [get_relevant_edges, get_edge_invalidation_candidates]
    )
    async def test_a_driver_without_a_search_interface_issues_exactly_one_query(self, fn):
        driver = _FakeDriver(GraphProvider.FALKORDB)
        await fn(driver, [_edge()], SearchFilters())
        assert len(driver.queries) == 1

    @pytest.mark.asyncio
    async def test_kuzu_keeps_its_reified_edge_node(self):
        driver = _FakeDriver(GraphProvider.KUZU)
        await get_relevant_edges(driver, [_edge()], SearchFilters())
        assert 'RelatesToNode_' in driver.queries[0]

    @pytest.mark.asyncio
    async def test_neo4j_keeps_its_typed_relationship(self):
        driver = _FakeDriver(GraphProvider.NEO4J)
        await get_relevant_edges(driver, [_edge()], SearchFilters())
        assert 'e:RELATES_TO {group_id: edge.group_id}' in driver.queries[0]


# ------------------------------------------------------- the AGE implementation


def _row(**kwargs):
    return SimpleNamespace(**kwargs)


def _predicate_selects(predicate: str, *, e_src: str, e_tgt: str, q_src: str, q_tgt: str) -> bool:
    """Would `predicate` admit an edge `e_src -> e_tgt` for a probe `q_src/q_tgt`?

    The SQL fragment is translated to the equivalent Python expression — `AND`,
    `OR`, `=` and `IN (a, b)` are all that appear — and evaluated. So the
    assertions below are about what the predicate WOULD SELECT, not about the
    presence of a substring.
    """
    python = (
        predicate.replace(' AND ', ' and ')
        .replace(' OR ', ' or ')
        .replace(' = ', ' == ')
        .replace(' IN ', ' in ')
    )
    namespace = {
        'e': _row(source_node_uuid=e_src, target_node_uuid=e_tgt),
        'q': _row(src=q_src, tgt=q_tgt),
    }
    return bool(eval(python, {'__builtins__': {}}, namespace))  # noqa: S307 - our own SQL


_ORDER_BY_RE = re.compile(r'ORDER BY\s+(?P<terms>[^\n]+?)\s*$', re.IGNORECASE | re.MULTILINE)


def _apply_order_by(sql: str, rows: list[dict]) -> list[dict]:
    """Sort `rows` the way the emitted SQL's own ORDER BY would.

    A model of the database, not a stub that invents an order: the clause is read
    out of the query under test and applied, so the ranking contract is covered by
    BEHAVIOUR. A query with NO ORDER BY leaves the rows exactly as the "table"
    hands them over — which is precisely what Postgres is entitled to do, and why
    deleting the clause is a real defect rather than a cosmetic one.

    NULLs sort last in both directions here (Postgres puts them first under DESC).
    The difference cannot matter: a null score is dropped by the min_score gate
    before it can reach a result list.
    """
    match = _ORDER_BY_RE.search(sql)
    if match is None:
        return list(rows)

    terms = []
    for term in match.group('terms').split(','):
        parts = term.strip().split()
        terms.append((parts[0].split('.')[-1], parts[-1].upper() == 'DESC'))

    ordered = list(rows)
    for column, descending in reversed(terms):  # stable: least significant first
        ordered.sort(
            key=lambda row, c=column: float('-inf') if row[c] is None else row[c],
            reverse=descending,
        )
    return ordered


class _FakeAGEDriver:
    """Records the SQL, answers canned rows, and hydrates through a fake ops layer."""

    _node_tbl = '"g__node_search"'
    _edge_tbl = '"g__edge_search"'
    text_search_config = 'simple'
    provider = GraphProvider.AGE

    def __init__(self, rows: list[dict] | None = None):
        self.rows = rows or []
        self.sql: list[str] = []
        self.args: list[tuple] = []
        self.graph_operations_interface = _FakeAGEOps()

    async def execute_sql(self, sql: str, *args):
        self.sql.append(sql)
        self.args.append(args)
        return _apply_order_by(sql, self.rows)

    async def execute_query(self, *args, **kwargs):  # pragma: no cover - must not be used
        raise AssertionError('the AGE dedup legs must not touch the labelled graph')


class _FakeAGEOps:
    """`edge_get_by_uuids` over an in-memory map, in arbitrary (unsorted) order."""

    def __init__(self):
        self.requested: list[list[str]] = []

    async def edge_get_by_uuids(self, cls, driver, uuids):
        self.requested.append(list(uuids))
        # Returned reversed on purpose: the hydration helper is what restores the
        # ranking order, and a test that got them pre-sorted would not see it.
        return [_edge(uuid=u, source=f'{u}-s', target=f'{u}-t') for u in reversed(uuids)]


@pytest.fixture
def age_search():
    from graphiti_core.driver.search_interface.age_search import AGESearch

    return AGESearch()


class TestTheAGEDedupLegAsksTheShadowTable:
    """No vertex label can appear, because no vertex is matched at all."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_the_sql_never_names_a_vertex_label(self, age_search, method):
        driver = _FakeAGEDriver()
        await getattr(age_search, method)(driver, [_edge()], SearchFilters())
        sql = driver.sql[0]
        assert ':Entity' not in sql
        assert not re.search(r'\bMATCH\b', sql), 'this must be SQL over the shadow table'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_the_sql_never_calls_the_neo4j_cosine_function(self, age_search, method):
        """`vector.similarity.cosine` is a hard parse error on AGE (measured)."""
        driver = _FakeAGEDriver()
        await getattr(age_search, method)(driver, [_edge()], SearchFilters())
        assert 'vector.similarity.cosine' not in driver.sql[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_the_sql_targets_the_edge_shadow_table(self, age_search, method):
        driver = _FakeAGEDriver()
        await getattr(age_search, method)(driver, [_edge()], SearchFilters())
        assert driver._edge_tbl in driver.sql[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_an_empty_edge_list_issues_no_sql(self, age_search, method):
        driver = _FakeAGEDriver()
        assert await getattr(age_search, method)(driver, [], SearchFilters()) == []
        assert driver.sql == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_the_score_is_the_normalized_cosine_every_provider_agrees_on(
        self, age_search, method
    ):
        """Same scale as `get_vector_cosine_func_query`'s FalkorDB branch (BUG-98)."""
        driver = _FakeAGEDriver()
        await getattr(age_search, method)(driver, [_edge()], SearchFilters())
        sql = driver.sql[0]
        assert '(2 - ' in sql and '<=>' in sql, sql


class TestTheAGECandidateSets:
    """The two predicates, evaluated rather than string-matched."""

    def test_dedup_admits_the_same_pair_in_either_direction(self):
        from graphiti_core.driver.search_interface.age_search import _SAME_PAIR_PREDICATE

        assert _predicate_selects(
            _SAME_PAIR_PREDICATE, e_src='A', e_tgt='B', q_src='A', q_tgt='B'
        )
        assert _predicate_selects(
            _SAME_PAIR_PREDICATE, e_src='B', e_tgt='A', q_src='A', q_tgt='B'
        ), 'the generic pattern is undirected — `-[e]-` — so this one must be too'

    def test_dedup_rejects_an_edge_that_only_shares_one_endpoint(self):
        from graphiti_core.driver.search_interface.age_search import _SAME_PAIR_PREDICATE

        assert not _predicate_selects(
            _SAME_PAIR_PREDICATE, e_src='A', e_tgt='C', q_src='A', q_tgt='B'
        )
        assert not _predicate_selects(
            _SAME_PAIR_PREDICATE, e_src='C', e_tgt='D', q_src='A', q_tgt='B'
        )

    def test_invalidation_admits_any_edge_incident_to_either_endpoint(self):
        from graphiti_core.driver.search_interface.age_search import _INCIDENT_PREDICATE

        for e_src, e_tgt in [('A', 'C'), ('C', 'A'), ('B', 'D'), ('D', 'B'), ('A', 'B')]:
            assert _predicate_selects(
                _INCIDENT_PREDICATE, e_src=e_src, e_tgt=e_tgt, q_src='A', q_tgt='B'
            ), f'{e_src}->{e_tgt} touches an endpoint and must be a candidate'

    def test_invalidation_rejects_an_edge_touching_neither_endpoint(self):
        from graphiti_core.driver.search_interface.age_search import _INCIDENT_PREDICATE

        assert not _predicate_selects(
            _INCIDENT_PREDICATE, e_src='C', e_tgt='D', q_src='A', q_tgt='B'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method,constant',
        [
            ('get_relevant_edges', '_SAME_PAIR_PREDICATE'),
            ('get_edge_invalidation_candidates', '_INCIDENT_PREDICATE'),
        ],
    )
    async def test_the_predicate_the_tests_evaluate_is_the_one_the_sql_carries(
        self, age_search, method, constant
    ):
        import graphiti_core.driver.search_interface.age_search as mod

        driver = _FakeAGEDriver()
        await getattr(age_search, method)(driver, [_edge()], SearchFilters())
        assert getattr(mod, constant) in driver.sql[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_the_candidate_set_is_scoped_to_the_edges_group(self, age_search, method):
        driver = _FakeAGEDriver()
        await getattr(age_search, method)(driver, [_edge(group_id='g7')], SearchFilters())
        assert 'group_id' in driver.sql[0]
        assert ['g7'] in [list(a) for a in driver.args[0] if isinstance(a, list)]


class TestTheAGEResultsAreGroupedGatedAndRanked:
    """The Python half: one list per input edge, in input order."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_one_result_list_per_input_edge_in_input_order(self, age_search, method):
        driver = _FakeAGEDriver(
            rows=[
                {'idx': 1, 'uuid': 'b1', 'score': 0.99},
                {'idx': 0, 'uuid': 'a1', 'score': 0.95},
            ]
        )
        result = await getattr(age_search, method)(
            driver, [_edge(uuid='in-0'), _edge(uuid='in-1'), _edge(uuid='in-2')], SearchFilters()
        )
        assert [[e.uuid for e in lst] for lst in result] == [['a1'], ['b1'], []]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_the_sql_orders_by_input_index_then_score_descending(
        self, age_search, method
    ):
        """The ORDER BY is the WHOLE ranking contract — nothing else sorts.

        Gating and truncation happen in Python over the rows AS DELIVERED, so if
        this clause is wrong or absent there is no second line of defence: the
        `limit` then keeps an arbitrary slice instead of the best candidates.
        `q.idx` first is what makes the per-input-edge grouping contiguous.
        """
        driver = _FakeAGEDriver()
        await getattr(age_search, method)(driver, [_edge()], SearchFilters())
        assert 'ORDER BY q.idx, score DESC' in driver.sql[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_candidates_come_back_ranked_from_an_unordered_table(
        self, age_search, method
    ):
        """Rows are handed over SHUFFLED — the query's own ORDER BY must rank them.

        Feeding pre-sorted rows would only prove that Python preserves what it
        receives. The fake applies the emitted clause instead (`_apply_order_by`),
        so an ASC flip or a deleted ORDER BY changes what this test observes.
        """
        driver = _FakeAGEDriver(
            rows=[
                {'idx': 0, 'uuid': 'low', 'score': 0.70},
                {'idx': 0, 'uuid': 'best', 'score': 0.99},
                {'idx': 0, 'uuid': 'mid', 'score': 0.80},
            ]
        )
        result = await getattr(age_search, method)(driver, [_edge()], SearchFilters())
        assert [e.uuid for e in result[0]] == ['best', 'mid', 'low']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_a_limit_of_one_keeps_the_BEST_candidate_not_an_arbitrary_one(
        self, age_search, method
    ):
        """The consequence of an unpinned ORDER BY, made observable.

        With the clause flipped to ASC the worst candidate still above the floor
        (0.61) is the one that survives `limit=1`, and the true 0.99 duplicate is
        never offered to the resolver — a silently wrong merge decision.
        """
        driver = _FakeAGEDriver(
            rows=[
                {'idx': 0, 'uuid': 'worst-above-the-floor', 'score': 0.61},
                {'idx': 0, 'uuid': 'the-real-duplicate', 'score': 0.99},
                {'idx': 0, 'uuid': 'middling', 'score': 0.80},
            ]
        )
        result = await getattr(age_search, method)(
            driver, [_edge()], SearchFilters(), limit=1
        )
        assert [e.uuid for e in result[0]] == ['the-real-duplicate']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_the_shuffled_rows_are_also_regrouped_by_input_edge(
        self, age_search, method
    ):
        """`q.idx` leading the ORDER BY is what keeps each input edge's rows together."""
        driver = _FakeAGEDriver(
            rows=[
                {'idx': 1, 'uuid': 'b-low', 'score': 0.70},
                {'idx': 0, 'uuid': 'a-best', 'score': 0.99},
                {'idx': 1, 'uuid': 'b-best', 'score': 0.95},
                {'idx': 0, 'uuid': 'a-low', 'score': 0.65},
            ]
        )
        result = await getattr(age_search, method)(
            driver, [_edge(uuid='in-0'), _edge(uuid='in-1')], SearchFilters()
        )
        assert [[e.uuid for e in lst] for lst in result] == [
            ['a-best', 'a-low'],
            ['b-best', 'b-low'],
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_the_min_score_gate_is_strict(self, age_search, method):
        """Every other provider writes `WHERE score > $min_score` (BUG-98)."""
        driver = _FakeAGEDriver(
            rows=[
                {'idx': 0, 'uuid': 'above', 'score': DEFAULT_MIN_SCORE + 0.01},
                {'idx': 0, 'uuid': 'exactly-at', 'score': DEFAULT_MIN_SCORE},
                {'idx': 0, 'uuid': 'below', 'score': DEFAULT_MIN_SCORE - 0.01},
            ]
        )
        result = await getattr(age_search, method)(driver, [_edge()], SearchFilters())
        assert [e.uuid for e in result[0]] == ['above']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_the_limit_truncates_per_input_edge_after_ranking(self, age_search, method):
        driver = _FakeAGEDriver(
            rows=[
                {'idx': 0, 'uuid': 'a', 'score': 0.99},
                {'idx': 0, 'uuid': 'b', 'score': 0.98},
                {'idx': 0, 'uuid': 'c', 'score': 0.97},
                {'idx': 1, 'uuid': 'd', 'score': 0.96},
                {'idx': 1, 'uuid': 'e', 'score': 0.95},
            ]
        )
        result = await getattr(age_search, method)(
            driver, [_edge(uuid='in-0'), _edge(uuid='in-1')], SearchFilters(), limit=2
        )
        assert [[e.uuid for e in lst] for lst in result] == [['a', 'b'], ['d', 'e']]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_a_null_score_row_is_dropped_not_crashed_on(self, age_search, method):
        driver = _FakeAGEDriver(
            rows=[
                {'idx': 0, 'uuid': 'ok', 'score': 0.9},
                {'idx': 0, 'uuid': 'null', 'score': None},
            ]
        )
        result = await getattr(age_search, method)(driver, [_edge()], SearchFilters())
        assert [e.uuid for e in result[0]] == ['ok']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_an_edge_without_an_embedding_yields_no_candidates_but_keeps_its_slot(
        self, age_search, method
    ):
        """A missing embedding is one edge's problem, not the whole batch's."""
        driver = _FakeAGEDriver(rows=[{'idx': 0, 'uuid': 'a', 'score': 0.9}])
        result = await getattr(age_search, method)(
            driver,
            [_edge(uuid='with'), _edge(uuid='without', embedding=[])],
            SearchFilters(),
        )
        assert len(result) == 2
        assert [e.uuid for e in result[0]] == ['a']
        assert result[1] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method', ['get_relevant_edges', 'get_edge_invalidation_candidates']
    )
    async def test_no_edge_has_an_embedding_so_no_sql_runs(self, age_search, method):
        driver = _FakeAGEDriver()
        result = await getattr(age_search, method)(
            driver, [_edge(embedding=[]), _edge(embedding=[])], SearchFilters()
        )
        assert result == [[], []]
        assert driver.sql == []
