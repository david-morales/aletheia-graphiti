"""BUG-98 residue (a): the AGE fulltext legs must OR their terms, not AND them.

The bug
-------
Both AGE fulltext legs matched on ``plainto_tsquery('simple', $1)``. That is an
AND over the query's terms — ``plainto_tsquery('simple', 'tipo de hecho')`` is
``'tipo' & 'de' & 'hecho'`` — so a document had to carry EVERY word of the
question, verbatim, unstemmed. The FalkorDB arm does the opposite:
``FalkorDriver.build_fulltext_query`` (``falkordb_driver.py:482-499``) drops
stopwords and joins what is left with ``' | '``, and RediSearch stems each term.
The two flavours therefore answered different questions from the same corpus,
and the AGE keyword leg contributed almost nothing to the hybrid RRF for any
phrasing that was not already a literal corpus string.

``'simple'`` is also the identity dictionary: it folds case and stems nothing.
A real configuration (``spanish``, ``english``, …) buys stemming and stopwords,
but it cannot be hardcoded — Graphiti is language-agnostic — so it is a driver
parameter with ``'simple'`` as the default.

What these tests prove, and how
-------------------------------
Two levels, both driven through the SQL the driver ACTUALLY emits:

* *shape* — the emitted SQL is asserted directly: the disjunction rewrite is
  present on both legs, the configuration is inlined as an IMMUTABLE literal
  (a bound ``::regconfig`` cast is STABLE and costs a per-row re-evaluation of
  the whole rewrite — measured 5.33 ms vs 1.62 ms), the USER TEXT is the half
  that stays a bind parameter, and an unvalidated configuration cannot reach
  the SQL text at all.
* *behaviour* — ``_FakePostgres`` extracts the tsquery expression out of that
  same SQL and evaluates it against a tiny corpus with a faithful model of the
  Postgres pieces involved: ``plainto_tsquery`` (lexize + quote + ``' & '``
  join), the textual ``replace`` exactly as the SQL writes it, then ``@@``.
  Revert the rewrite and the OR tests go red, because the model reads the
  operator out of the SQL rather than assuming it.

The model covers the ``simple`` configuration only. Stemming is a claim about
PostgreSQL's dictionaries, not about this code; what this code owes is that the
configured name reaches the query and the index, and THAT is asserted at the
shape level and in ``TestTheIndexAndTheQueryAgree``.

These tests live at ``tests/`` root deliberately, for the reason recorded in
``test_age_similarity_score_scale.py``: the unit-test CI job runs
``pytest tests/ -m "not integration"`` with ``--ignore=tests/driver/``.
"""

import re

import pytest

from graphiti_core.driver.age_driver import (
    DEFAULT_TEXT_SEARCH_CONFIG,
    AGEDriver,
    stored_tsv_config,
    text_search_configs_agree,
    validate_text_search_config,
)
from graphiti_core.driver.search_interface.age_search import AGESearch

# ---------------------------------------------------------------- the PG model

_PLAINTO_RE = re.compile(r"plainto_tsquery\(\s*'(?P<cfg>[^']*)'\s*,\s*\$(?P<txt>\d+)\s*\)")
# `replace(<x>, 'a', 'b')` — the two literal arguments, in order of application.
_REPLACE_ARGS_RE = re.compile(r",\s*'(?P<old>[^']*)'\s*,\s*'(?P<new>[^']*)'\s*\)")
_RANK_RE = re.compile(r'ts_rank_cd\(tsv,\s*(?P<expr>.+?)\)\s+AS rank', re.DOTALL)
_MATCH_RE = re.compile(r'WHERE tsv @@ (?P<expr>.+?)(?:\s+AND group_id|\s*\n)', re.DOTALL)


def _lexize(text: str) -> list[str]:
    """The `simple` configuration: split into word tokens, fold case, keep all."""
    return [t.lower() for t in re.findall(r'[0-9A-Za-zÀ-ÿ]+', text)]


def _tsquery_from_sql(expr: str, args: list) -> tuple[str, list[list[str]]]:
    """Evaluate the driver's tsquery expression into (operator, alternatives).

    Reads `plainto_tsquery`'s operands out of the SQL, models its output text,
    applies every `replace(..., 'x', 'y')` the expression performs, and parses
    what comes out. The operator is whatever the SQL produced — nothing here
    assumes OR.
    """
    call = _PLAINTO_RE.search(expr)
    assert call, f'no plainto_tsquery call with an inlined configuration in: {expr}'
    text = args[int(call.group('txt')) - 1]
    lexemes = _lexize(text)
    rendered = ' & '.join(f"'{lexeme}'" for lexeme in lexemes)
    for replacement in _REPLACE_ARGS_RE.finditer(expr):
        rendered = rendered.replace(replacement.group('old'), replacement.group('new'))
    if not rendered:
        return '&', []
    operator = '|' if ' | ' in rendered else '&'
    parts = [p.strip().strip("'") for p in re.split(r'\s[&|]\s', rendered)]
    return operator, [[p] for p in parts if p]


class _FakePostgres:
    """Runs the emitted SQL against an in-memory corpus.

    `rank` models `ts_rank_cd` as the number of distinct query lexemes the
    document carries — enough to assert that a document matching more of the
    question outranks one matching less, which is the ordering the AND-gate
    destroyed by not emitting the row at all. It is NOT a reimplementation of
    cover density, and no test here depends on more than that ordering.
    """

    _node_tbl = '"g__node_search"'
    _edge_tbl = '"g__edge_search"'

    def __init__(self, corpus: dict[str, str], text_search_config: str = 'simple'):
        self.corpus = corpus
        self.text_search_config = text_search_config
        self.last_sql: str | None = None
        self.last_args: list | None = None

    async def execute_sql(self, sql: str, *args):
        self.last_sql, self.last_args = sql, list(args)
        rank_expr = _RANK_RE.search(sql)
        match_expr = _MATCH_RE.search(sql)
        assert rank_expr, f'no `ts_rank_cd(tsv, …) AS rank` projection in:\n{sql}'
        assert match_expr, f'no `WHERE tsv @@ …` predicate in:\n{sql}'
        # The two must be the same query, or ranking and matching disagree.
        assert rank_expr.group('expr').strip() == match_expr.group('expr').strip(), (
            'the ranked tsquery differs from the matched one:\n'
            f'  rank:  {rank_expr.group("expr").strip()}\n'
            f'  match: {match_expr.group("expr").strip()}'
        )
        operator, alternatives = _tsquery_from_sql(rank_expr.group('expr'), list(args))
        wanted = [alt[0] for alt in alternatives]

        rows = []
        for uuid, content in self.corpus.items():
            present = {lexeme for lexeme in _lexize(content)}
            hits = [term for term in wanted if term in present]
            if not wanted:
                continue
            matched = bool(hits) if operator == '|' else len(hits) == len(wanted)
            if matched:
                rows.append({'uuid': uuid, 'rank': float(len(hits))})
        rows.sort(key=lambda r: -r['rank'])
        return rows


async def _capture_uuids(self, driver, uuids):  # noqa: ARG001 - hydration stub
    return list(uuids)


@pytest.fixture
def search() -> AGESearch:
    return AGESearch()


@pytest.fixture(autouse=True)
def _no_hydration(monkeypatch):
    monkeypatch.setattr(AGESearch, '_hydrate_nodes_in_order', _capture_uuids, raising=True)
    monkeypatch.setattr(AGESearch, '_hydrate_edges_in_order', _capture_uuids, raising=True)


# A corpus in the shape the bug was measured on: one document carrying the whole
# phrase, one carrying part of it, one carrying none.
CORPUS = {
    'all': 'TIPO DE HECHO registrado en el parte',
    'partial': 'OTROS HECHOS DE INTERES POLICIAL',
    'none': 'ATESTADO POR ROBO',
}


# ------------------------------------------------------------------ behaviour


class TestOrSemanticsOnTheNodeLeg:
    @pytest.mark.asyncio
    async def test_a_query_whose_terms_are_split_across_documents_matches_both(self, search):
        """The bug, exactly: 'tipo de hecho' returned NOTHING but the literal doc."""
        driver = _FakePostgres(CORPUS)
        uuids = await search.node_fulltext_search(driver, 'tipo de hecho', None, limit=10)
        assert uuids == ['all', 'partial'], (
            'the AND-gate is back: a document carrying only some of the query terms '
            'was dropped instead of ranked'
        )

    @pytest.mark.asyncio
    async def test_a_term_absent_from_every_document_does_not_veto_the_match(self, search):
        """One unknown word used to zero the whole result set."""
        driver = _FakePostgres(CORPUS)
        uuids = await search.node_fulltext_search(driver, 'hecho ornitorrinco', None, limit=10)
        assert uuids == ['all']

    @pytest.mark.asyncio
    async def test_under_simple_a_plural_still_does_not_reach_its_singular(self, search):
        """The limitation this change does NOT lift, pinned so it stays honest.

        OR-semantics is the half that lands on an existing `simple` bed. Matching
        'hechos' from 'hecho' is stemming, which lives in the CONFIGURATION and
        therefore in the shadow tables — see TestTheIndexAndTheQueryAgree.
        """
        driver = _FakePostgres(CORPUS)
        assert await search.node_fulltext_search(driver, 'hecho', None, limit=10) == ['all']

    @pytest.mark.asyncio
    async def test_more_matched_terms_rank_higher(self, search):
        driver = _FakePostgres(CORPUS)
        uuids = await search.node_fulltext_search(driver, 'tipo de hecho', None, limit=10)
        assert uuids[0] == 'all', 'the document matching every term must rank first'

    @pytest.mark.asyncio
    async def test_a_single_term_query_is_unchanged(self, search):
        """OR over one term is that term — the literal-string path must not move."""
        driver = _FakePostgres(CORPUS)
        assert await search.node_fulltext_search(driver, 'hechos', None, limit=10) == ['partial']

    @pytest.mark.asyncio
    async def test_a_query_matching_nothing_still_matches_nothing(self, search):
        driver = _FakePostgres(CORPUS)
        assert await search.node_fulltext_search(driver, 'ornitorrinco', None, limit=10) == []

    @pytest.mark.parametrize('empty', ['', '   ', '\t\n'])
    @pytest.mark.asyncio
    async def test_an_empty_query_never_reaches_the_database(self, search, empty):
        driver = _FakePostgres(CORPUS)
        assert await search.node_fulltext_search(driver, empty, None, limit=10) == []
        assert driver.last_sql is None

    @pytest.mark.asyncio
    async def test_a_punctuation_only_query_yields_the_empty_tsquery(self, search):
        """Non-empty but lexeme-free: it reaches SQL and matches nothing, no error."""
        driver = _FakePostgres(CORPUS)
        assert await search.node_fulltext_search(driver, '¿?¡! ---', None, limit=10) == []
        assert driver.last_sql is not None


class TestOrSemanticsOnTheEdgeLeg:
    """The edge leg is a separate copy of the same SQL and regressed separately."""

    @pytest.mark.asyncio
    async def test_a_query_whose_terms_are_split_across_edges_matches_both(self, search):
        driver = _FakePostgres(CORPUS)
        uuids = await search.edge_fulltext_search(driver, 'tipo de hecho', None, limit=10)
        assert uuids == ['all', 'partial']

    @pytest.mark.asyncio
    async def test_a_single_term_query_is_unchanged(self, search):
        driver = _FakePostgres(CORPUS)
        assert await search.edge_fulltext_search(driver, 'hechos', None, limit=10) == ['partial']

    @pytest.mark.parametrize('empty', ['', '   '])
    @pytest.mark.asyncio
    async def test_an_empty_query_never_reaches_the_database(self, search, empty):
        driver = _FakePostgres(CORPUS)
        assert await search.edge_fulltext_search(driver, empty, None, limit=10) == []
        assert driver.last_sql is None


# ----------------------------------------------------------------- SQL shape


class TestTheEmittedSql:
    """Asserted against the driver's own SQL — the BUG-98 fix set the precedent."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', ['node_fulltext_search', 'edge_fulltext_search'])
    async def test_the_conjunction_is_rewritten_to_a_disjunction(self, search, leg):
        driver = _FakePostgres(CORPUS)
        await getattr(search, leg)(driver, 'tipo de hecho', None, limit=10)
        assert 'replace(' in driver.last_sql
        assert "' & ', ' | '" in driver.last_sql, (
            "the emitted SQL no longer turns plainto_tsquery's AND into an OR:\n" + driver.last_sql
        )
        assert '::tsquery' in driver.last_sql

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', ['node_fulltext_search', 'edge_fulltext_search'])
    async def test_the_configuration_is_inlined_as_an_immutable_literal(self, search, leg):
        """Inlined ON PURPOSE, and the reason is a measurement.

        Bound as `$2::text::regconfig` the cast runs `regconfigin`, which is
        STABLE — the expression stops being foldable and `ts_rank_cd`
        re-evaluates the whole rewrite PER ROW (review measured 5.33 ms vs
        1.62 ms on a 774-row table, same rows returned). A literal is a
        plan-time constant. Safe because the value is validated before it ever
        gets here — see `TestTheConfigurationIsValidatedBeforeItCanReachSql`.
        """
        driver = _FakePostgres(CORPUS, text_search_config='spanish')
        await getattr(search, leg)(driver, 'tipo de hecho', None, limit=10)
        assert "plainto_tsquery('spanish', $1)" in driver.last_sql
        assert '::regconfig' not in driver.last_sql, (
            'the STABLE regconfig cast is back — this is the per-row re-evaluation '
            'the inlining exists to avoid:\n' + driver.last_sql
        )
        assert driver.last_args == ['tipo de hecho'], 'only the USER TEXT may be bound'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', ['node_fulltext_search', 'edge_fulltext_search'])
    async def test_the_user_text_is_never_inlined(self, search, leg):
        """The half that must stay a parameter, whatever the configuration does."""
        hostile = "x'); DROP TABLE nodes; --"
        driver = _FakePostgres(CORPUS)
        await getattr(search, leg)(driver, hostile, None, limit=10)
        assert 'DROP TABLE' not in driver.last_sql
        assert driver.last_args == [hostile]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', ['node_fulltext_search', 'edge_fulltext_search'])
    async def test_the_default_configuration_is_simple(self, search, leg):
        driver = _FakePostgres(CORPUS)
        del driver.text_search_config  # a driver predating the parameter
        await getattr(search, leg)(driver, 'hecho', None, limit=10)
        assert f"plainto_tsquery('{DEFAULT_TEXT_SEARCH_CONFIG}', $1)" in driver.last_sql
        assert DEFAULT_TEXT_SEARCH_CONFIG == 'simple'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('leg', 'table'),
        [
            ('node_fulltext_search', '"g__node_search"'),
            ('edge_fulltext_search', '"g__edge_search"'),
        ],
    )
    async def test_group_ids_take_the_second_parameter(self, search, leg, table):
        """Nothing but the query text and the group ids is bound."""
        driver = _FakePostgres(CORPUS)
        await getattr(search, leg)(driver, 'hecho', None, group_ids=['g1'], limit=10)
        assert 'AND group_id = ANY($2::text[])' in driver.last_sql
        assert driver.last_args == ['hecho', ['g1']]
        assert table in driver.last_sql

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', ['node_fulltext_search', 'edge_fulltext_search'])
    async def test_without_group_ids_only_the_query_text_is_bound(self, search, leg):
        driver = _FakePostgres(CORPUS)
        await getattr(search, leg)(driver, 'hecho', None, limit=10)
        assert driver.last_args == ['hecho']
        assert '$2' not in driver.last_sql

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', ['node_fulltext_search', 'edge_fulltext_search'])
    async def test_no_bare_plainto_tsquery_survives_anywhere(self, search, leg):
        """The AND gate is gone from BOTH the ranking and the matching halves.

        `plainto_tsquery(…)` still appears — it is the thing being rewritten —
        so the assertion is that EVERY occurrence is wrapped in the rewrite,
        which is what an occurrence-counting check can state and a substring
        check cannot.
        """
        driver = _FakePostgres(CORPUS)
        await getattr(search, leg)(driver, 'hecho', None, limit=10)
        calls = driver.last_sql.count('plainto_tsquery(')
        wrapped = driver.last_sql.count('replace(plainto_tsquery(')
        assert calls == 2, driver.last_sql  # one to rank with, one to match on
        assert wrapped == calls, (
            'a bare plainto_tsquery survived the rewrite — that half still ANDs:\n'
            + driver.last_sql
        )


# ------------------------------------------------- configuration & validation


class TestTheConfigurationIsValidatedBeforeItCanReachSql:
    @pytest.mark.parametrize('name', ['simple', 'spanish', 'english', 'pg_catalog.spanish', '_x9'])
    def test_valid_names_are_accepted(self, name):
        assert validate_text_search_config(name) == name

    @pytest.mark.parametrize(
        'name',
        [
            "simple'); DROP TABLE users; --",
            "simple' || (SELECT 1) || '",
            'simple simple',
            'simple;',
            '',
            '1simple',
            'a.b.c',
            'sim"ple',
        ],
    )
    def test_injection_shapes_are_rejected(self, name):
        with pytest.raises(ValueError, match='Invalid text_search_config'):
            validate_text_search_config(name)

    def test_the_driver_rejects_a_hostile_value_at_construction(self):
        with pytest.raises(ValueError, match='Invalid text_search_config'):
            AGEDriver(
                dsn='postgresql://unused/unused',
                graph_name='g',
                text_search_config="simple'); DROP TABLE x; --",
            )

    def test_the_driver_defaults_to_simple(self):
        driver = AGEDriver(dsn='postgresql://unused/unused', graph_name='g')
        assert driver.text_search_config == 'simple'

    def test_the_driver_carries_the_configured_value(self):
        driver = AGEDriver(
            dsn='postgresql://unused/unused', graph_name='g', text_search_config='spanish'
        )
        assert driver.text_search_config == 'spanish'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', ['node_fulltext_search', 'edge_fulltext_search'])
    @pytest.mark.parametrize(
        'hostile', ["simple'); DROP TABLE x; --", "simple' || (SELECT 1) || '", 'a b', 'x;']
    )
    async def test_the_search_leg_refuses_to_inline_an_unvalidated_value(
        self, search, leg, hostile
    ):
        """The ctor is the first gate; this is the one that makes inlining LOCAL.

        `AGEDriver.__init__` validates, and every production path goes through
        it — but the search leg now writes the value into SQL text, and
        "inlined because someone upstream checked" is how injections happen.
        Handed an object that never saw the ctor, the leg must still refuse
        rather than emit the string.
        """
        driver = _FakePostgres(CORPUS, text_search_config=hostile)
        with pytest.raises(ValueError, match='Invalid text_search_config'):
            await getattr(search, leg)(driver, 'hecho', None, limit=10)
        assert driver.last_sql is None, 'nothing may reach the database on this path'


class TestTheIndexAndTheQueryAgree:
    """The stored tsvector and the tsquery must name the SAME configuration."""

    def test_the_generated_column_uses_the_configured_value(self):
        driver = AGEDriver(
            dsn='postgresql://unused/unused', graph_name='g', text_search_config='spanish'
        )
        assert driver._tsv_generated_expr == "to_tsvector('spanish', coalesce(content, ''))"

    def test_the_generated_column_defaults_to_simple(self):
        driver = AGEDriver(dsn='postgresql://unused/unused', graph_name='g')
        assert driver._tsv_generated_expr == "to_tsvector('simple', coalesce(content, ''))"

    @pytest.mark.parametrize(
        ('expr', 'expected'),
        [
            ("to_tsvector('simple'::regconfig, COALESCE(content, ''::text))", 'simple'),
            ("to_tsvector('spanish'::regconfig, COALESCE(content, ''::text))", 'spanish'),
            ("to_tsvector('simple', coalesce(content, ''))", 'simple'),
            ('lower(content)', None),
            (None, None),
        ],
    )
    def test_the_stored_configuration_is_read_back_out_of_the_catalog_expression(
        self, expr, expected
    ):
        assert stored_tsv_config(expr) == expected

    @pytest.mark.parametrize(
        ('configured', 'stored', 'agree'),
        [
            ('simple', 'simple', True),
            ('spanish', 'simple', False),
            ('simple', 'spanish', False),
            ('pg_catalog.spanish', 'spanish', True),
            ('spanish', 'pg_catalog.spanish', True),
            ('SPANISH', 'spanish', True),
            ('spanish', None, True),  # unreadable: never a false alarm
        ],
    )
    def test_drift_detection(self, configured, stored, agree):
        assert text_search_configs_agree(configured, stored) is agree


class TestTheDdlCarriesTheConfiguration:
    """Drives the real `build_indices_and_constraints` against a fake pool."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('config', ['simple', 'spanish'])
    async def test_both_shadow_tables_are_generated_with_it(self, monkeypatch, config):
        driver = AGEDriver(
            dsn='postgresql://unused/unused', graph_name='g', text_search_config=config
        )
        conn = _FakeConn(graph_exists=True, stored_expr=f"to_tsvector('{config}'::regconfig, x)")
        monkeypatch.setattr(AGEDriver, '_get_pool', _pool_returning(conn))
        await driver.build_indices_and_constraints()

        creates = [s for s in conn.executed if 'CREATE TABLE IF NOT EXISTS' in s]
        assert len(creates) == 2, conn.executed
        for statement in creates:
            assert f"to_tsvector('{config}', coalesce(content, ''))" in statement

    @pytest.mark.asyncio
    async def test_a_table_lexized_differently_is_reported(self, monkeypatch, caplog):
        driver = AGEDriver(
            dsn='postgresql://unused/unused', graph_name='g', text_search_config='spanish'
        )
        conn = _FakeConn(
            graph_exists=True, stored_expr="to_tsvector('simple'::regconfig, COALESCE(content, ''))"
        )
        monkeypatch.setattr(AGEDriver, '_get_pool', _pool_returning(conn))
        with caplog.at_level('WARNING'):
            await driver.build_indices_and_constraints()
        warnings = [r.message for r in caplog.records if r.levelname == 'WARNING']
        assert len(warnings) == 2, warnings  # one per shadow table
        assert "'simple'" in warnings[0] and "'spanish'" in warnings[0]

    @pytest.mark.asyncio
    async def test_a_matching_table_is_silent(self, monkeypatch, caplog):
        driver = AGEDriver(
            dsn='postgresql://unused/unused', graph_name='g', text_search_config='spanish'
        )
        conn = _FakeConn(
            graph_exists=True,
            stored_expr="to_tsvector('spanish'::regconfig, COALESCE(content, ''))",
        )
        monkeypatch.setattr(AGEDriver, '_get_pool', _pool_returning(conn))
        with caplog.at_level('WARNING'):
            await driver.build_indices_and_constraints()
        assert [r.message for r in caplog.records if r.levelname == 'WARNING'] == []

    @pytest.mark.asyncio
    async def test_an_unreadable_expression_is_not_reported_as_drift(self, monkeypatch, caplog):
        driver = AGEDriver(
            dsn='postgresql://unused/unused', graph_name='g', text_search_config='spanish'
        )
        conn = _FakeConn(graph_exists=True, stored_expr=None)
        monkeypatch.setattr(AGEDriver, '_get_pool', _pool_returning(conn))
        with caplog.at_level('WARNING'):
            await driver.build_indices_and_constraints()
        assert [r.message for r in caplog.records if r.levelname == 'WARNING'] == []


# ---------------------------------------------------------------- fake asyncpg


class _FakeConn:
    """Just enough asyncpg connection for `build_indices_and_constraints`."""

    def __init__(self, graph_exists: bool, stored_expr: str | None):
        self.graph_exists = graph_exists
        self.stored_expr = stored_expr
        self.executed: list[str] = []

    async def execute(self, sql: str, *args):
        self.executed.append(sql)

    async def fetchval(self, sql: str, *args):
        if 'ag_catalog.ag_graph' in sql:
            return 1 if self.graph_exists else 0
        if 'pg_get_expr' in sql:
            return self.stored_expr
        return None


def _pool_returning(conn):
    class _Acquire:
        async def __aenter__(self):
            return conn

        async def __aexit__(self, *exc):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _get_pool(self):
        return _Pool()

    return _get_pool
