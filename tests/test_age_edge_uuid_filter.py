"""BUG-107: on AGE, `SearchFilters.edge_uuids` must NARROW the edge legs.

The bug
-------
`AGESearch` documented `edge_uuids` as DROPPED ("Dropped filters BROADEN
results; they never narrow them wrongly. Callers that need real filtering must
post-filter"). That reasoning holds for a SEARCH whose caller ranks what it is
offered. It does not hold for the caller that actually passes this filter.

The live ingest dedup path passes it and never post-filters:

    graphiti_core/utils/maintenance/edge_operations.py:565
    graphiti_core/graphiti.py:1807

        valid_edges = await EntityEdge.get_between_nodes(driver, src, tgt)
        related_edges = (await search(
            clients, edge.fact,
            group_ids=[edge.group_id],
            config=EDGE_HYBRID_SEARCH_RRF,
            search_filter=SearchFilters(edge_uuids=[e.uuid for e in valid_edges]),
        )).edges

`EDGE_HYBRID_SEARCH_RRF` fans out bm25 + cosine over edges, and
`graphiti_core/search/search.py::edge_search` hands `search_filter` straight to
`edge_fulltext_search` / `edge_similarity_search` / `edge_bfs_search`. Nothing
between those legs and the dedup prompt filters by uuid. So on AGE the question
"which stored edge is this extracted fact a duplicate of?" was asked of the
graph-wide top-K, spanning OTHER node pairs entirely — the BUG-94 D1/D2
wrong-merge mechanism class.

Two distinct failures, both covered below:

  (a) an edge belonging to a DIFFERENT node pair is offered as a duplicate
      candidate — a wrong merge is one LLM judgement away; and
  (b) the real same-pair candidate is LOST, because `LIMIT` truncates the
      graph-wide ranking before the caller ever sees it. This is why a
      post-filter is not an equivalent fix and why the filter has to be in the
      query: filtering after the limit cannot recover a row the limit dropped.

The EMPTY list is the common case, not an edge case: `get_between_nodes`
returns `[]` for a pair that has no stored edge yet, i.e. on every first ingest
of a pair. Every other provider renders that as `e.uuid in []` (see
`edge_search_filter_query_constructor`, which branches on `is not None`) and
therefore offers ZERO dedup candidates. AGE offered the whole top-K.

What this does NOT change
-------------------------
`node_labels` and the temporal filters stay DROPPED on this backend for the
reasons the module docstring already gives (`n:A|B` is Neo4j syntax AGE cannot
parse; the shadow tables carry no temporal columns). `TestTheScopeDidNotWiden`
pins that.

How these tests prove it
------------------------
The fakes EVALUATE the emitted query rather than string-matching it: the WHERE
clause is parsed out of the SQL under test and applied conjunct by conjunct to
modelled rows, and the AGE Cypher's `WITH rel WHERE …` is applied the same way
to modelled relationships. A conjunct the model does not recognise raises
instead of being ignored, so a filter cannot pass this suite by being invisible,
and `LIMIT` is applied where the database would apply it — after the filter.
"""

import ast
import re
from typing import Any

import pytest

from graphiti_core.driver.search_interface.age_search import AGESearch
from graphiti_core.search.search_filters import SearchFilters

# ---------------------------------------------------------------- the SQL model

_WHERE_RE = re.compile(r'\bWHERE\s+(?P<where>.*?)\s+ORDER BY\b', re.DOTALL | re.IGNORECASE)
_LIMIT_RE = re.compile(r'\bLIMIT\s+(?P<limit>\d+)', re.IGNORECASE)

_IS_NOT_NULL_RE = re.compile(r'^(?P<col>[a-z_]+) IS NOT NULL$')
_EQ_ANY_RE = re.compile(r'^(?P<col>[a-z_]+) = ANY\(\$(?P<n>\d+)(?:::[a-z\[\]]+)?\)$')
_EQ_PARAM_RE = re.compile(r'^(?P<col>[a-z_]+) = \$(?P<n>\d+)$')
_TSMATCH_RE = re.compile(r'^tsv @@ .+$', re.DOTALL)


def _lexemes(text: str) -> set[str]:
    """The `simple` text-search configuration: word tokens, case folded."""
    return {t.lower() for t in re.findall(r'[0-9A-Za-zÀ-ÿ]+', text)}


def _row_matches(conjunct: str, row: dict, args: list) -> bool:
    """Evaluate ONE emitted WHERE conjunct against one modelled row.

    Every shape the AGE edge legs can emit is enumerated here. Anything else
    raises: a model that silently ignored an unknown conjunct would report a
    filter as working while the query did nothing.
    """
    match = _IS_NOT_NULL_RE.match(conjunct)
    if match:
        return row.get(match.group('col')) is not None

    match = _EQ_ANY_RE.match(conjunct)
    if match:
        return row.get(match.group('col')) in args[int(match.group('n')) - 1]

    match = _EQ_PARAM_RE.match(conjunct)
    if match:
        return row.get(match.group('col')) == args[int(match.group('n')) - 1]

    if _TSMATCH_RE.match(conjunct):
        # OR semantics over the query's lexemes (BUG-98); `$1` is the query text.
        return bool(_lexemes(str(args[0])) & _lexemes(row.get('text', '')))

    raise AssertionError(f'the SQL model does not understand this conjunct: {conjunct!r}')


class _FakeEdgeShadowTable:
    """The uuid-keyed edge shadow table, as a model that RUNS the emitted SQL.

    `execute_sql` parses the query's own WHERE clause and LIMIT and applies
    them, in that order, to `rows`. Ordering is by descending `score` (the
    similarity leg orders by ascending `fact_embedding <=> $1`, a monotone
    inverse of the projected score) or descending `rank` for the fulltext leg,
    matching the emitted `ORDER BY` in both cases.
    """

    _node_tbl = '"g__node_search"'
    _edge_tbl = '"g__edge_search"'
    text_search_config = 'simple'

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.sql: list[str] = []
        self.args: list[list] = []

    async def execute_sql(self, sql: str, *args) -> list[dict]:
        self.sql.append(sql)
        self.args.append(list(args))

        where = _WHERE_RE.search(sql)
        assert where, f'no WHERE … ORDER BY in the emitted SQL:\n{sql}'
        conjuncts = [c.strip() for c in where.group('where').split(' AND ') if c.strip()]

        kept = [
            row for row in self.rows if all(_row_matches(c, row, list(args)) for c in conjuncts)
        ]

        if 'AS rank' in sql:
            out = [
                {
                    'uuid': r['uuid'],
                    'rank': float(len(_lexemes(str(args[0])) & _lexemes(r['text']))),
                }
                for r in kept
            ]
            out.sort(key=lambda r: -r['rank'])
        else:
            out = [{'uuid': r['uuid'], 'score': r['score']} for r in kept]
            out.sort(key=lambda r: -r['score'])

        limit = _LIMIT_RE.search(sql)
        assert limit, f'no LIMIT in the emitted SQL:\n{sql}'
        return out[: int(limit.group('limit'))]

    async def execute_query(self, *args, **kwargs):  # pragma: no cover - must not be used
        raise AssertionError('the shadow-table legs must not touch the labelled graph')


# -------------------------------------------------------------- the Cypher model

_REL_WHERE_RE = re.compile(r'WITH rel WHERE (?P<conds>.*?) RETURN DISTINCT', re.DOTALL)
_ORIGIN_RE = re.compile(r'origin\.uuid IN (?P<list>\[.*?\])')

_NOT_TYPE_IN_RE = re.compile(r'^NOT type\(rel\) IN (?P<list>\[.*\])$', re.DOTALL)
_REL_PROP_IN_RE = re.compile(r'^rel\.(?P<prop>[a-z_]+) IN (?P<list>\[.*\])$', re.DOTALL)


def _rel_matches(conjunct: str, rel: dict) -> bool:
    """Evaluate ONE emitted `WITH rel WHERE` conjunct against one relationship.

    Same contract as `_row_matches`: an unrecognised conjunct raises. Cypher
    list literals are read with `ast.literal_eval` — `_cy`'s escaping produces
    valid Python literals — so no `eval` of query text happens here.
    """
    match = _NOT_TYPE_IN_RE.match(conjunct)
    if match:
        return rel['type'] not in ast.literal_eval(match.group('list'))

    match = _REL_PROP_IN_RE.match(conjunct)
    if match:
        return rel.get(match.group('prop')) in ast.literal_eval(match.group('list'))

    raise AssertionError(f'the Cypher model does not understand this conjunct: {conjunct!r}')


class _FakeAGEGraph:
    """The labelled AGE graph, as a model that RUNS the emitted BFS Cypher.

    Relationships declare which origins they are reachable from; the traversal
    is not what is under test, the `WITH rel WHERE …` filter is. The filter is
    applied BEFORE `LIMIT`, which is where AGE applies it — a post-filter in
    Python would run after, and could not recover a truncated row.
    """

    _node_tbl = '"g__node_search"'
    _edge_tbl = '"g__edge_search"'
    text_search_config = 'simple'

    def __init__(self, rels: list[dict]):
        self.rels = rels
        self.cypher: list[str] = []

    async def execute_query(self, cypher: str, **kwargs):
        self.cypher.append(cypher)

        origins = _ORIGIN_RE.search(cypher)
        assert origins, f'the BFS Cypher lost its origin anchor:\n{cypher}'
        reachable_from = set(ast.literal_eval(origins.group('list')))

        conds = _REL_WHERE_RE.search(cypher)
        assert conds, f'no `WITH rel WHERE … RETURN DISTINCT` in:\n{cypher}'
        conjuncts = [c.strip() for c in conds.group('conds').split(' AND ') if c.strip()]

        kept = [
            rel
            for rel in self.rels
            if (set(rel['from']) & reachable_from) and all(_rel_matches(c, rel) for c in conjuncts)
        ]

        limit = _LIMIT_RE.search(cypher)
        assert limit, f'no LIMIT in the emitted Cypher:\n{cypher}'
        kept = kept[: int(limit.group('limit'))]
        return [{'uuid': rel['uuid']} for rel in kept], None, None

    async def execute_sql(self, *args, **kwargs):  # pragma: no cover - must not be used
        raise AssertionError('the BFS leg is graph traversal, not a shadow-table read')


# ------------------------------------------------------------------- fixtures


async def _capture_uuids(self, driver, uuids):  # noqa: ARG001 - hydration stub
    return list(uuids)


@pytest.fixture
def search() -> AGESearch:
    return AGESearch()


@pytest.fixture(autouse=True)
def _no_hydration(monkeypatch):
    monkeypatch.setattr(AGESearch, '_hydrate_edges_in_order', _capture_uuids, raising=True)
    monkeypatch.setattr(AGESearch, '_hydrate_nodes_in_order', _capture_uuids, raising=True)


# The shape the defect was reasoned from: the extracted fact concerns the pair
# (A, B), which has exactly one stored edge — and that edge is NOT the top of
# the graph-wide ranking. Two edges of an unrelated pair (C, D) outrank it.
PAIR_EDGE = {
    'uuid': 'same-pair',
    'group_id': 'g',
    'source_node_uuid': 'A',
    'target_node_uuid': 'B',
    'fact_embedding': '[0.1]',
    'text': 'A tipificado como B',
    'score': 0.70,
}
OTHER_PAIR_EDGES = [
    {
        'uuid': 'other-1',
        'group_id': 'g',
        'source_node_uuid': 'C',
        'target_node_uuid': 'D',
        'fact_embedding': '[0.1]',
        'text': 'C tipificado como D',
        'score': 0.99,
    },
    {
        'uuid': 'other-2',
        'group_id': 'g',
        'source_node_uuid': 'C',
        'target_node_uuid': 'E',
        'fact_embedding': '[0.1]',
        'text': 'C tipificado como E',
        'score': 0.95,
    },
]
ALL_EDGES = [*OTHER_PAIR_EDGES, PAIR_EDGE]

QUERY = 'tipificado como'
VECTOR = [0.1, 0.2, 0.3, 0.4]


async def _similarity(search, driver, search_filter, limit=10):
    return await search.edge_similarity_search(
        driver, VECTOR, None, None, search_filter, ['g'], limit, 0.5
    )


async def _fulltext(search, driver, search_filter, limit=10):
    return await search.edge_fulltext_search(driver, QUERY, search_filter, ['g'], limit)


# --------------------------------------------------------- the shadow-table legs


@pytest.mark.parametrize('leg', [_similarity, _fulltext])
class TestTheShadowTableLegsHonourEdgeUuids:
    """`edge_similarity_search` and `edge_fulltext_search` — the two legs
    `EDGE_HYBRID_SEARCH_RRF` actually fans out on the ingest dedup path."""

    @pytest.mark.asyncio
    async def test_an_edge_of_another_node_pair_is_never_offered(self, search, leg):
        driver = _FakeEdgeShadowTable(ALL_EDGES)

        result = await leg(search, driver, SearchFilters(edge_uuids=['same-pair']))

        assert result == ['same-pair'], (
            'edges spanning OTHER node pairs reached the dedup candidate set — '
            'the wrong-merge mechanism BUG-107 records'
        )

    @pytest.mark.asyncio
    async def test_the_real_candidate_survives_a_limit_the_others_would_fill(self, search, leg):
        """Why the filter has to be IN the query, not applied to its output.

        With `limit=2` the two unrelated edges fill the whole result and the one
        edge that IS between this node pair never leaves the database. No
        post-filter can recover it, so the dedup prompt would be asked about a
        pair it was never given the stored fact for.
        """
        driver = _FakeEdgeShadowTable(ALL_EDGES)

        result = await leg(search, driver, SearchFilters(edge_uuids=['same-pair']), limit=2)

        assert result == ['same-pair']

    @pytest.mark.asyncio
    async def test_an_empty_filter_offers_no_candidate_at_all(self, search, leg):
        """`get_between_nodes` returns `[]` on every first ingest of a pair.

        Every other provider renders that as `e.uuid in []` and offers nothing.
        Treating `[]` as "no filter" is the widest form of this bug: the whole
        graph-wide top-K becomes duplicate candidates for a brand-new pair.
        """
        driver = _FakeEdgeShadowTable(ALL_EDGES)

        assert await leg(search, driver, SearchFilters(edge_uuids=[])) == []
        # Answered BY THE QUERY (`= ANY('{}')` is defined Postgres), not by a
        # Python short-circuit that a later refactor could drop.
        assert 'uuid = ANY(' in driver.sql[0]
        assert [] in driver.args[0]

    @pytest.mark.asyncio
    async def test_no_filter_still_searches_the_whole_table(self, search, leg):
        driver = _FakeEdgeShadowTable(ALL_EDGES)

        result = await leg(search, driver, SearchFilters())

        assert sorted(result) == ['other-1', 'other-2', 'same-pair']

    @pytest.mark.asyncio
    async def test_a_missing_search_filter_object_is_tolerated(self, search, leg):
        """Callers in this repo pass `search_filter=None` (see the offline
        combined-search guards); reading the attribute must not raise."""
        driver = _FakeEdgeShadowTable(ALL_EDGES)

        assert sorted(await leg(search, driver, None)) == ['other-1', 'other-2', 'same-pair']

    @pytest.mark.asyncio
    async def test_the_uuids_are_bound_never_inlined(self, search, leg):
        hostile = "x') OR true --"
        driver = _FakeEdgeShadowTable([])

        await leg(search, driver, SearchFilters(edge_uuids=[hostile]))

        assert hostile not in driver.sql[0], driver.sql[0]
        assert [hostile] in driver.args[0], driver.args[0]


class TestTheSimilarityLegParameterSlots:
    """The uuid parameter must not collide with the slots already in use."""

    @pytest.mark.asyncio
    async def test_every_placeholder_resolves_to_the_value_it_names(self, search):
        driver = _FakeEdgeShadowTable([])

        await search.edge_similarity_search(
            driver, VECTOR, 'A', 'B', SearchFilters(edge_uuids=['u1']), ['g'], 10, 0.5
        )

        sql, args = driver.sql[0], driver.args[0]
        slots = {
            'group_id': _EQ_ANY_RE,
            'source_node_uuid': _EQ_PARAM_RE,
            'target_node_uuid': _EQ_PARAM_RE,
            'uuid': _EQ_ANY_RE,
        }
        expected = {
            'group_id': ['g'],
            'source_node_uuid': 'A',
            'target_node_uuid': 'B',
            'uuid': ['u1'],
        }
        where = _WHERE_RE.search(sql)
        assert where
        seen: dict[str, Any] = {}
        for conjunct in (c.strip() for c in where.group('where').split(' AND ')):
            for col, pattern in slots.items():
                match = pattern.match(conjunct)
                if match and match.group('col') == col:
                    seen[col] = args[int(match.group('n')) - 1]
        assert seen == expected, f'placeholder/value mismatch\nSQL: {sql}\nargs: {args}'


# ------------------------------------------------------------------- the BFS leg

REL_SAME_PAIR = {
    'uuid': 'same-pair',
    'type': 'TIPIFICADO_COMO',
    'name': 'TIPIFICADO_COMO',
    'group_id': 'g',
    'from': ['A'],
}
REL_OTHER_PAIR = [
    {
        'uuid': 'other-1',
        'type': 'TIPIFICADO_COMO',
        'name': 'TIPIFICADO_COMO',
        'group_id': 'g',
        'from': ['A'],
    },
    {
        'uuid': 'other-2',
        'type': 'TIPIFICADO_COMO',
        'name': 'TIPIFICADO_COMO',
        'group_id': 'g',
        'from': ['A'],
    },
]
ALL_RELS = [*REL_OTHER_PAIR, REL_SAME_PAIR]


class TestTheBfsLegHonoursEdgeUuids:
    """`edge_bfs_search` is the third leg an `EdgeSearchMethod.bfs` recipe fans
    out with the same `SearchFilters`."""

    @pytest.mark.asyncio
    async def test_a_relationship_outside_the_filter_is_never_returned(self, search):
        driver = _FakeAGEGraph(ALL_RELS)

        result = await search.edge_bfs_search(
            driver, ['A'], 2, SearchFilters(edge_uuids=['same-pair']), ['g'], 10
        )

        assert result == ['same-pair']

    @pytest.mark.asyncio
    async def test_the_filter_is_applied_before_the_limit(self, search):
        """In-query, not post-hoc: with `limit=2` the two unfiltered relationships
        would fill the result and the wanted one would never be returned."""
        driver = _FakeAGEGraph(ALL_RELS)

        result = await search.edge_bfs_search(
            driver, ['A'], 2, SearchFilters(edge_uuids=['same-pair']), ['g'], 2
        )

        assert result == ['same-pair']

    @pytest.mark.asyncio
    async def test_an_empty_filter_returns_nothing(self, search):
        """Answered without a round trip, and so without emitting `IN []`.

        The SQL legs hand an empty array to `= ANY($n)`, which is defined
        Postgres; AGE's Cypher parser is a different engine and no offline test
        can prove it accepts an empty list literal. Same result either way.
        """
        driver = _FakeAGEGraph(ALL_RELS)

        assert (
            await search.edge_bfs_search(driver, ['A'], 2, SearchFilters(edge_uuids=[]), ['g'], 10)
            == []
        )
        assert driver.cypher == [], 'an empty allow-list needs no query at all'

    @pytest.mark.asyncio
    async def test_no_filter_still_traverses_everything(self, search):
        driver = _FakeAGEGraph(ALL_RELS)

        result = await search.edge_bfs_search(driver, ['A'], 2, SearchFilters(), ['g'], 10)

        assert sorted(result) == ['other-1', 'other-2', 'same-pair']

    @pytest.mark.asyncio
    async def test_it_still_composes_with_the_edge_types_filter(self, search):
        """`edge_types` was already HONOURED here; the new conjunct must AND with
        it rather than replace it."""
        driver = _FakeAGEGraph(ALL_RELS)

        result = await search.edge_bfs_search(
            driver,
            ['A'],
            2,
            SearchFilters(edge_types=['NO_SUCH_TYPE'], edge_uuids=['same-pair']),
            ['g'],
            10,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_the_uuids_are_escaped_by_the_shared_serializer(self, search):
        """Inlined into Cypher — but through `_cy`, the write path's serializer,
        so a quote cannot terminate the literal."""
        driver = _FakeAGEGraph([])

        await search.edge_bfs_search(
            driver, ['A'], 2, SearchFilters(edge_uuids=["x' OR true --"]), ['g'], 10
        )

        assert "\\'" in driver.cypher[0], driver.cypher[0]
        assert "'x' OR true --'" not in driver.cypher[0]


# ------------------------------------------------------------------ scope guards


class TestTheScopeDidNotWiden:
    """`edge_uuids` moved to HONOURED. Nothing else did."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', [_similarity, _fulltext])
    async def test_node_labels_are_still_dropped_on_the_shadow_table_legs(self, search, leg):
        """The generic constructor emits `n:A|B`, Neo4j syntax AGE cannot parse,
        and the shadow tables carry no label column to read instead."""
        driver = _FakeEdgeShadowTable(ALL_EDGES)

        result = await leg(search, driver, SearchFilters(node_labels=['Persona']))

        assert sorted(result) == ['other-1', 'other-2', 'same-pair']
        assert 'label' not in driver.sql[0].lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', [_similarity, _fulltext])
    async def test_temporal_filters_are_still_dropped(self, search, leg):
        """The edge shadow table has no `valid_at` / `invalid_at` columns."""
        from graphiti_core.search.search_filters import ComparisonOperator, DateFilter

        never = SearchFilters(
            valid_at=[[DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)]]
        )
        driver = _FakeEdgeShadowTable(ALL_EDGES)

        result = await leg(search, driver, never)

        assert sorted(result) == ['other-1', 'other-2', 'same-pair']
        assert 'valid_at' not in driver.sql[0]

    def test_the_module_docstring_records_edge_uuids_as_honoured(self):
        """The contract this module publishes is read by the next person who has
        to decide whether a caller may rely on the filter. It has to be true."""
        import graphiti_core.driver.search_interface.age_search as mod

        doc = mod.__doc__ or ''
        assert '`edge_uuids`   — DROPPED' not in doc, doc
        assert 'edge_uuids' in doc and 'HONOURED' in doc
        # unchanged, and stated so
        assert '`node_labels`  — DROPPED' in doc

    def test_no_not_implemented_swallowing_was_introduced(self):
        """BUG-108 guard: the delegation idiom that catches `NotImplementedError`
        and re-issues the generic query lives in `search_utils`, and this fix
        must not add a sixth site of it."""
        import inspect

        import graphiti_core.driver.search_interface.age_search as mod

        assert 'except NotImplementedError' not in inspect.getsource(mod)


class TestTheCallerThatMakesThisLoadBearing:
    """If these callers change, re-read BUG-107 before relaxing anything here."""

    @pytest.mark.parametrize(
        'module',
        [
            'graphiti_core.utils.maintenance.edge_operations',
            'graphiti_core.graphiti',
        ],
    )
    def test_the_dedup_path_still_narrows_by_edge_uuid(self, module):
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(module))
        assert 'SearchFilters(edge_uuids=[edge.uuid for edge in valid_edges])' in source

    def test_nothing_post_filters_the_edge_results_by_uuid(self):
        """The reason the query must do it: `edge_search` hands the filter to the
        legs and returns whatever they give back."""
        import inspect

        from graphiti_core.search import search as search_mod

        source = inspect.getsource(search_mod.edge_search)
        assert 'edge_uuids' not in source
