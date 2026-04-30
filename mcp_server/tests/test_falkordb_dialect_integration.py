"""Integration tests that exercise validate_and_sanitize + live FalkorDB.

Marked `@pytest.mark.integration` — requires a running FalkorDB on
localhost:6379 with a populated `policia_partes_v3` graph.

Run with:
    pytest tests/test_falkordb_dialect_integration.py -v -m integration
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from falkordb import FalkorDB

src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.cypher import (
    SanitizedQuery,
    CypherError,
    classify_execution_error,
    validate_and_sanitize,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def graph():
    client = FalkorDB(host='localhost', port=6379)
    return client.select_graph('policia_partes_v3')


class TestRegressionIncidentQueries:
    """The two queries that failed in the 2026-04-14 workbench incident."""

    def test_bare_variable_query_now_succeeds_after_autofix(self, graph):
        # Simplified form of the incident query — bare variable.
        query = (
            'MATCH (p:Persona) '
            'OPTIONAL MATCH parte-[:TIPIFICADO_COMO]->(tipo:TipoDelito) '
            'RETURN p.name LIMIT 1'
        )
        result = validate_and_sanitize(query)
        assert isinstance(result, SanitizedQuery), f'Pipeline rejected: {result}'
        # Bare-variable auto-fix was applied
        assert any('bare variable' in f.lower() for f in result.auto_fixes)
        # Query runs against FalkorDB
        res = graph.ro_query(result.query)
        assert res.result_set is not None

    def test_unwind_where_query_is_rejected_with_guidance(self):
        query = (
            'MATCH (p:Persona) WITH p LIMIT 1 '
            'UNWIND [1,2,3] AS x '
            'WHERE x > 1 '
            'RETURN p, x'
        )
        result = validate_and_sanitize(query)
        assert isinstance(result, CypherError)
        assert result.reason == 'unwind_where_missing_with'


class TestRemovedRejectsNowPass:
    """Queries that used to be wrongly rejected should now succeed."""

    def test_pattern_comprehension_executes(self, graph):
        query = 'MATCH (n:Persona) RETURN n.name, [(n)-[r]->(m) | type(r)] AS rels LIMIT 1'
        result = validate_and_sanitize(query)
        assert isinstance(result, SanitizedQuery)
        res = graph.ro_query(result.query)
        assert res.result_set is not None

    def test_call_subquery_executes(self, graph):
        query = (
            'MATCH (n:Persona) '
            'CALL { WITH n MATCH (n)-[r]->(m) RETURN count(r) AS c } '
            'RETURN n.name, c LIMIT 1'
        )
        result = validate_and_sanitize(query)
        assert isinstance(result, SanitizedQuery)
        res = graph.ro_query(result.query)
        assert res.result_set is not None

    def test_map_projection_executes(self, graph):
        query = 'MATCH (n:Persona) RETURN n { .name, .uuid } LIMIT 1'
        result = validate_and_sanitize(query)
        assert isinstance(result, SanitizedQuery)
        res = graph.ro_query(result.query)
        assert res.result_set is not None


class TestExecutionErrorClassification:
    """Errors from FalkorDB are translated into actionable suggestions."""

    def test_bare_variable_parser_error_is_classified(self):
        # Use a verbatim FalkorDB error string (one we already observed).
        raw_msg = "errMsg: Invalid input '-': expected '=' line: 1, column: 40"
        err = classify_execution_error(raw_msg)
        assert err.reason == 'bare_variable_in_pattern'

    def test_unwind_where_parser_error_is_classified(self):
        raw_msg = (
            "errMsg: Invalid input 'H': expected WITH "
            "line: 10, column: 2 errCtx: WHERE evento.numero"
        )
        err = classify_execution_error(raw_msg)
        assert err.reason == 'where_needs_with'

    def test_return_where_parser_error_is_classified(self):
        # Observed in the field: LLM emitted `RETURN <projections> WHERE
        # alias IS NOT NULL ORDER BY ts LIMIT 201`. FalkorDB's parser error
        # is the same shape as the UNWIND-WHERE case, so the classifier must
        # give a suggestion that works for both query shapes.
        raw_msg = (
            "errMsg: Invalid input 'H': expected WITH line: 33, column: 2, offset: 1376 "
            "errCtx: WHERE alias_name IS NOT NULL errCtxOffset: 1"
        )
        err = classify_execution_error(raw_msg)
        assert err.reason == 'where_needs_with'
        assert 'RETURN' in err.suggestion

    def test_round_arity_mismatch_end_to_end(self, graph):
        # LLM generates Neo4j's `round(x, N)` signature (Round 2 Q2 incident).
        # FalkorDB rejects at execution time; classifier converts the
        # cryptic message into an actionable envelope.
        query = 'MATCH (n:Persona) RETURN round(avg(size(n.name)), 2) AS avg_len LIMIT 1'
        result = validate_and_sanitize(query)
        assert isinstance(result, SanitizedQuery), f'Pipeline unexpectedly rejected: {result}'
        try:
            graph.ro_query(result.query)
            pytest.fail('Expected FalkorDB to reject round(x, 2) as arity mismatch')
        except Exception as e:
            err = classify_execution_error(str(e))
            assert err.reason == 'function_arity_mismatch'
