"""BUG-98 residue (b): Kuzu carried the same raw-cosine-vs-0.6-gate defect.

The measurement that opened this
--------------------------------
BUG-98 was a SCALE mismatch, not a broken query: `DEFAULT_MIN_SCORE = 0.6`
(`search_utils`) is calibrated on the NORMALIZED `(1 + cos) / 2` scale — the one
Neo4j's `vector.similarity.cosine` produces by definition and FalkorDB builds
explicitly with `(2 - vec.cosineDistance(a, b)) / 2` — while the AGE leg gated
the RAW cosine on the same 0.6. Raw 0.6 is normalized 0.8, so AGE ran a floor a
third higher than every other arm and answered only near-literal corpus strings.

The ledger recorded Kuzu as a suspected sibling, uninvestigated. It is one.
Measured on kuzu 0.11.3, in-process, no fixtures:

    array_cosine_similarity([1,0], [1,0])    =  1.000000   identical
    array_cosine_similarity([1,0], [0,1])    =  0.000000   ORTHOGONAL
    array_cosine_similarity([1,0], [-1,0])   = -1.000000   opposite

Range [-1, 1] — the raw cosine, exactly like pgvector's `1 - (a <=> b)`. Every
Kuzu caller then writes `WHERE score > $min_score` against it
(`kuzu/operations/search_ops.py:151/390/614`, `search_utils:554/913/979`), so a
Kuzu graph ran the same effective ~0.8 floor.

The live top hit BUG-98 was measured on lands in exactly the same place:

    raw 0.4130  ->  rejected at 0.6      (the bug)
    (1 + 0.4130) / 2 = 0.7065  ->  kept  (the cure; falkor scored it 0.7061)

The cure is the same algebra as the FalkorDB branch, applied in the one place
every provider's expression is built: `graph_queries.get_vector_cosine_func_query`.
It changes the score, never the order — `(1 + c) / 2` is monotone in `c`, and no
caller does anything with the value but gate it and `ORDER BY` it.
"""

import re

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graph_queries import get_vector_cosine_func_query
from graphiti_core.search.search_utils import DEFAULT_MIN_SCORE

# The live top hit for 'tipo de hecho' that BUG-98's AGE arm discarded.
MEASURED_RAW_COSINE = 0.4130


def _replace_call(expr: str, func: str, value: float) -> str:
    """Substitute `func(...)` — parentheses balanced — with a numeric literal."""
    start = expr.index(func + '(')
    depth, i = 0, start + len(func)
    while i < len(expr):
        if expr[i] == '(':
            depth += 1
        elif expr[i] == ')':
            depth -= 1
            if depth == 0:
                break
        i += 1
    return expr[:start] + repr(value) + expr[i + 1 :]


def _score(provider: GraphProvider, cosine: float) -> float:
    """Evaluate the driver's OWN emitted expression at a known cosine.

    Not a stub that invents a score: the expression the query builder wrote is
    what gets evaluated, so a regression in it changes what these tests observe.
    """
    expr = get_vector_cosine_func_query('a', 'b', provider)
    if provider == GraphProvider.KUZU:
        numeric = _replace_call(expr, 'array_cosine_similarity', cosine)
    elif provider == GraphProvider.FALKORDB:
        # `vec.cosineDistance` is `1 - cos`.
        numeric = _replace_call(expr, 'vec.cosineDistance', 1.0 - cosine)
    else:
        raise AssertionError(f'no numeric model for {provider}')
    assert not re.search(r'[a-z_]+\.?[a-z_]*\(', numeric), f'operand not substituted: {numeric}'
    return eval(numeric)  # noqa: S307 - our own generated expression


class TestTheKuzuExpressionIsNormalized:
    @pytest.mark.parametrize(
        ('cosine', 'expected'),
        [(1.0, 1.0), (0.0, 0.5), (-1.0, 0.0), (0.5, 0.75), (MEASURED_RAW_COSINE, 0.7065)],
    )
    def test_the_raw_cosine_is_mapped_onto_zero_to_one(self, cosine, expected):
        assert _score(GraphProvider.KUZU, cosine) == pytest.approx(expected, abs=1e-4)

    def test_the_bug_condition_the_measured_hit_now_clears_the_floor(self):
        """Raw 0.4130 was rejected by a 0.6 gate; normalized it is 0.7065."""
        assert MEASURED_RAW_COSINE < DEFAULT_MIN_SCORE  # what the gate saw before
        assert _score(GraphProvider.KUZU, MEASURED_RAW_COSINE) > DEFAULT_MIN_SCORE

    def test_an_orthogonal_vector_lands_exactly_on_the_boundary(self):
        """0.5, so the strict `>` every caller writes is what excludes it."""
        assert _score(GraphProvider.KUZU, 0.0) == 0.5

    def test_the_expression_still_calls_the_kuzu_function(self):
        expr = get_vector_cosine_func_query('n.name_embedding', '$v', GraphProvider.KUZU)
        assert 'array_cosine_similarity(n.name_embedding, $v)' in expr


class TestKuzuAndFalkordbNowAgree:
    """The two arms must be comparable by construction, not by coincidence."""

    @pytest.mark.parametrize('cosine', [-1.0, -0.25, 0.0, MEASURED_RAW_COSINE, 0.5, 1.0])
    def test_the_same_cosine_scores_the_same_on_both(self, cosine):
        assert _score(GraphProvider.KUZU, cosine) == pytest.approx(
            _score(GraphProvider.FALKORDB, cosine), abs=1e-9
        )

    def test_neo4j_is_left_alone(self):
        """`vector.similarity.cosine` is normalized by definition — do not rescale."""
        expr = get_vector_cosine_func_query('a', 'b', GraphProvider.NEO4J)
        assert expr == 'vector.similarity.cosine(a, b)'


class TestTheMeasurementItself:
    """The claim `array_cosine_similarity` is RAW, checked against real Kuzu.

    Skipped where the optional `kuzu` extra is absent; CI installs
    `--all-extras`, so this runs there.
    """

    @pytest.fixture(scope='class')
    def conn(self, tmp_path_factory):
        kuzu = pytest.importorskip('kuzu')
        db = kuzu.Database(str(tmp_path_factory.mktemp('kuzu') / 'db'))
        return kuzu.Connection(db)

    def _eval(self, conn, cypher: str) -> float:
        return conn.execute(cypher).get_next()[0]

    @pytest.mark.parametrize(
        ('a', 'b', 'raw'),
        [
            ('[1.0,0.0]', '[1.0,0.0]', 1.0),
            ('[1.0,0.0]', '[0.0,1.0]', 0.0),
            ('[1.0,0.0]', '[-1.0,0.0]', -1.0),
        ],
    )
    def test_the_native_function_returns_the_raw_cosine(self, conn, a, b, raw):
        value = self._eval(
            conn,
            f'RETURN array_cosine_similarity(CAST({a} AS FLOAT[2]), CAST({b} AS FLOAT[2])) AS s',
        )
        assert value == pytest.approx(raw, abs=1e-6)

    @pytest.mark.parametrize(
        ('a', 'b', 'normalized'),
        [
            ('[1.0,0.0]', '[1.0,0.0]', 1.0),
            ('[1.0,0.0]', '[0.0,1.0]', 0.5),
            ('[1.0,0.0]', '[-1.0,0.0]', 0.0),
        ],
    )
    def test_the_emitted_expression_returns_the_normalized_cosine(self, conn, a, b, normalized):
        expr = get_vector_cosine_func_query(
            f'CAST({a} AS FLOAT[2])', f'CAST({b} AS FLOAT[2])', GraphProvider.KUZU
        )
        assert self._eval(conn, f'RETURN {expr} AS s') == pytest.approx(normalized, abs=1e-6)

    def test_an_orthogonal_pair_no_longer_clears_a_zero_point_six_floor_by_accident(self, conn):
        """The whole point: the floor now means the same thing it means elsewhere."""
        expr = get_vector_cosine_func_query(
            'CAST([1.0,0.0] AS FLOAT[2])', 'CAST([0.0,1.0] AS FLOAT[2])', GraphProvider.KUZU
        )
        assert self._eval(conn, f'RETURN {expr} AS s') < DEFAULT_MIN_SCORE
