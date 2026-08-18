"""BUG-98 regression: the AGE similarity legs must score on the NORMALIZED scale.

Bug (wire-measured 2026-08-17, root-caused 2026-08-18): the AGE flavour's
``search`` was paraphrase-blind. Nine phrasings against the same corpus returned
5 nodes each on FalkorDB; on AGE only the near-literal corpus strings answered
(``'robo'`` 5, ``'ROBO CON FUERZA EN LAS COSAS'`` 5) while every abstract
paraphrase returned ZERO (``'tipo de hecho'``, ``'clasificación del hecho en el
parte'``, ...).

Root cause — a SCALE mismatch against a shared threshold, not a broken query:

  * ``DEFAULT_MIN_SCORE = 0.6`` (``graphiti_core/search/search_utils.py``) is
    calibrated on the NORMALIZED [0, 1] cosine scale. Every other provider
    produces that scale: Neo4j's ``vector.similarity.cosine`` is normalized by
    definition, and FalkorDB rescales explicitly in
    ``graph_queries.get_vector_cosine_func_query`` —
    ``(2 - vec.cosineDistance(a, b)) / 2``, i.e. ``(1 + cos) / 2``.
  * ``AGESearch`` projected ``1 - (name_embedding <=> $1::vector)``. pgvector's
    ``<=>`` is cosine DISTANCE (``1 - cos``), so that expression is the RAW
    cosine, range [-1, 1] — and it was then gated on the same 0.6.

A raw 0.6 is a normalized 0.8, so AGE ran an ~0.8 floor where every other
provider ran 0.6. Measured on the live bench corpus (1361 nodes / 1336 on the
falkor twin), query ``'tipo de hecho'``:

    AGE raw    0.4130  OTROS HECHOS DE INTERES POLICIAL   -> rejected at 0.6
    falkor     0.7061  OTROS HECHOS DE INTERES POLICIAL   -> kept
    nodes passing 0.6:  falkor 642 / 1336      AGE 0 / 1361

Same nodes, same order, scores related by exactly ``falkor = (1 + age) / 2``.
The legs agreed on the ranking and disagreed only on the scale the floor was
applied to — so the semantically correct hit was found and then thrown away.

These tests are offline — no live backend, no fixtures from
``tests/driver/conftest.py`` — and they live at ``tests/`` root DELIBERATELY.
The unit-test CI job runs ``pytest tests/ -m "not integration"`` with
``--ignore=tests/driver/`` (``.github/workflows/unit_tests.yml``), so a
regression guard placed beside the other AGE tests would never have run in CI
at all. Here it gates every push.

The behavioural tests drive the REAL SQL the driver emits through a fake driver
that evaluates the projected score expression against a known cosine distance —
the exact condition that triggered the bug: a node at raw cosine 0.4130 must
survive ``min_score=0.6``.
"""

import re

import pytest

from graphiti_core.driver.search_interface.age_search import AGESearch

# The live top hit for 'tipo de hecho' that the bug discarded.
MEASURED_RAW_COSINE = 0.4130
DEFAULT_MIN_SCORE = 0.6

_SCORE_RE = re.compile(r'SELECT uuid,\s*(?P<expr>.+?)\s+AS score', re.IGNORECASE | re.DOTALL)


class _FakeAGEDriver:
    """Evaluates the driver's OWN projected score expression at a given cosine.

    Not a stub that invents a score: it extracts the expression the search code
    actually wrote into the SQL and computes it, so a regression in that
    expression changes what these tests observe.
    """

    def __init__(self, cosine: float, *, uuid: str = 'n1'):
        self._distance = 1.0 - cosine  # pgvector `<=>` is cosine distance
        self._uuid = uuid
        self.last_sql: str | None = None

    _node_tbl = '"g__node_search"'
    _edge_tbl = '"g__edge_search"'

    async def execute_sql(self, sql: str, *args):
        self.last_sql = sql
        match = _SCORE_RE.search(sql)
        assert match, f'no `... AS score` projection found in:\n{sql}'
        expr = match.group('expr')
        # Substitute the pgvector distance operand with the numeric distance.
        numeric = re.sub(r'\w+\s*<=>\s*\$1::vector', repr(self._distance), expr)
        assert '<=>' not in numeric, f'distance operand not substituted in: {expr}'
        return [{'uuid': self._uuid, 'score': eval(numeric)}]  # noqa: S307 - our own SQL


@pytest.fixture
def search() -> AGESearch:
    return AGESearch()


class TestTheProjectedScoreIsNormalized:
    """The SQL must emit the [0, 1] scale `min_score` is calibrated for."""

    @pytest.mark.asyncio
    async def test_node_similarity_projects_the_normalized_cosine(self, search, monkeypatch):
        driver = _FakeAGEDriver(MEASURED_RAW_COSINE)
        monkeypatch.setattr(AGESearch, '_hydrate_nodes_in_order', _capture_uuids, raising=True)
        await search.node_similarity_search(
            driver, [0.1] * 4, None, group_ids=['g'], limit=10, min_score=DEFAULT_MIN_SCORE
        )
        assert driver.last_sql is not None
        expr = _SCORE_RE.search(driver.last_sql).group('expr')
        # `1 - (... <=> ...)` is the RAW cosine and is the bug.
        assert '(2 -' in expr, f'expected the normalized (2 - distance)/2 form, got: {expr!r}'

    @pytest.mark.asyncio
    async def test_edge_similarity_projects_the_normalized_cosine(self, search, monkeypatch):
        driver = _FakeAGEDriver(MEASURED_RAW_COSINE)
        monkeypatch.setattr(AGESearch, '_hydrate_edges_in_order', _capture_uuids, raising=True)
        await search.edge_similarity_search(
            driver,
            [0.1] * 4,
            None,
            None,
            None,
            group_ids=['g'],
            limit=10,
            min_score=DEFAULT_MIN_SCORE,
        )
        expr = _SCORE_RE.search(driver.last_sql).group('expr')
        assert '(2 -' in expr, f'expected the normalized (2 - distance)/2 form, got: {expr!r}'


class TestTheMinScoreGateKeepsTheMeasuredHit:
    """The condition that triggered BUG-98, reproduced exactly."""

    @pytest.mark.asyncio
    async def test_node_at_the_measured_cosine_survives_the_default_floor(
        self, search, monkeypatch
    ):
        driver = _FakeAGEDriver(MEASURED_RAW_COSINE)
        monkeypatch.setattr(AGESearch, '_hydrate_nodes_in_order', _capture_uuids, raising=True)
        kept = await search.node_similarity_search(
            driver, [0.1] * 4, None, group_ids=['g'], limit=10, min_score=DEFAULT_MIN_SCORE
        )
        assert kept == ['n1'], (
            'a node at raw cosine 0.4130 (normalized 0.7065) must clear a 0.6 '
            'floor — this is the live hit OTROS HECHOS DE INTERES POLICIAL that '
            'the raw-scale comparison discarded'
        )

    @pytest.mark.asyncio
    async def test_edge_at_the_measured_cosine_survives_the_default_floor(
        self, search, monkeypatch
    ):
        # Live AGE edge scores for the same dead query were 0.5141/0.5140/0.5138 raw.
        driver = _FakeAGEDriver(0.5141)
        monkeypatch.setattr(AGESearch, '_hydrate_edges_in_order', _capture_uuids, raising=True)
        kept = await search.edge_similarity_search(
            driver,
            [0.1] * 4,
            None,
            None,
            None,
            group_ids=['g'],
            limit=10,
            min_score=DEFAULT_MIN_SCORE,
        )
        assert kept == ['n1'], (
            'edges=0 for a paraphrase while edges=10 for a literal was the same '
            'raw-vs-normalized scale bug on the edge leg'
        )

    @pytest.mark.parametrize('leg', ['node', 'edge'])
    @pytest.mark.asyncio
    async def test_a_score_exactly_at_the_floor_is_excluded(self, search, monkeypatch, leg):
        """The gate is STRICT, as it is for every other provider.

        Reachable in practice, not theoretical: an ORTHOGONAL vector (raw cosine
        0) scores exactly 0.5 on the normalized scale, so a `>=` gate at
        min_score=0.5 would admit a node sharing nothing with the query. Under
        the old raw scale that node scored 0.0 and was excluded by arithmetic,
        which is why the loose comparison never showed.
        """
        driver = _FakeAGEDriver(0.0)  # orthogonal -> normalized exactly 0.5
        kept = await _run_leg(search, monkeypatch, driver, leg, min_score=0.5)
        assert kept == [], (
            f'{leg} leg admitted a score exactly equal to min_score; every other '
            f'provider gates `WHERE score > $min_score`'
        )

    @pytest.mark.asyncio
    async def test_a_genuinely_unrelated_node_is_still_rejected(self, search, monkeypatch):
        """The fix must not degenerate into "keep everything".

        Normalized 0.6 == raw 0.2, so a node below raw 0.2 must still be cut —
        otherwise the floor stops discriminating at all.
        """
        driver = _FakeAGEDriver(0.10)
        monkeypatch.setattr(AGESearch, '_hydrate_nodes_in_order', _capture_uuids, raising=True)
        kept = await search.node_similarity_search(
            driver, [0.1] * 4, None, group_ids=['g'], limit=10, min_score=DEFAULT_MIN_SCORE
        )
        assert kept == []


class TestTheScaleMatchesTheOtherProviders:
    """Parity with `get_vector_cosine_func_query`'s FalkorDB normalization."""

    @pytest.mark.parametrize('leg', ['node', 'edge'])
    @pytest.mark.parametrize('cosine', [-1.0, -0.25, 0.0, 0.3926, 0.4130, 0.6241, 1.0])
    @pytest.mark.asyncio
    async def test_age_score_equals_the_falkordb_normalized_score(
        self, search, monkeypatch, cosine, leg
    ):
        driver = _FakeAGEDriver(cosine)
        # min_score=-1 keeps every row so the raw score reaches the assertion.
        await _run_leg(search, monkeypatch, driver, leg, min_score=-1.0)
        expr = _SCORE_RE.search(driver.last_sql).group('expr')
        numeric = re.sub(r'\w+\s*<=>\s*\$1::vector', repr(1.0 - cosine), expr)
        age_score = eval(numeric)  # noqa: S307 - our own SQL
        falkor_score = (2 - (1 - cosine)) / 2  # graph_queries.py, FALKORDB branch
        assert age_score == pytest.approx(falkor_score), (
            f'AGE scores {age_score} where every other provider scores {falkor_score}'
        )
        assert 0.0 <= age_score <= 1.0


async def _capture_uuids(self, driver, uuids):
    """Stand in for hydration — returns the uuids that passed the min_score gate."""
    return list(uuids)


_HYDRATORS = {'node': '_hydrate_nodes_in_order', 'edge': '_hydrate_edges_in_order'}


async def _run_leg(search, monkeypatch, driver, leg: str, *, min_score: float) -> list[str]:
    """Drive one similarity leg with hydration stubbed, returning the kept uuids.

    The two legs take different signatures (edges carry source/target filters),
    so parametrizing over them needs this seam rather than a single call.
    """
    monkeypatch.setattr(AGESearch, _HYDRATORS[leg], _capture_uuids, raising=True)
    if leg == 'node':
        return await search.node_similarity_search(
            driver, [0.1] * 4, None, group_ids=['g'], limit=10, min_score=min_score
        )
    return await search.edge_similarity_search(
        driver, [0.1] * 4, None, None, None, group_ids=['g'], limit=10, min_score=min_score
    )
