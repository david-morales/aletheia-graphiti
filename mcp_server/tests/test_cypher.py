"""Tests for Cypher validation pipeline and result formatting."""
from __future__ import annotations
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cypher import CypherError, SanitizedQuery


class TestDataModels:
    def test_sanitized_query_holds_query_and_fixes(self):
        sq = SanitizedQuery(query='MATCH (n) RETURN n', auto_fixes=[])
        assert sq.query == 'MATCH (n) RETURN n'
        assert sq.auto_fixes == []

    def test_sanitized_query_with_fixes(self):
        sq = SanitizedQuery(
            query="MATCH (n) WHERE n.date > '2024-01-01' RETURN n",
            auto_fixes=["date('2024-01-01') -> '2024-01-01' (FalkorDB uses string dates)"],
        )
        assert len(sq.auto_fixes) == 1

    def test_cypher_error_holds_structured_fields(self):
        err = CypherError(
            stage='falkordb_dialect',
            reason='apoc_unsupported',
            found='apoc.path.expand(...)',
            explanation='APOC procedures are not available in FalkorDB.',
            suggestion='Use variable-length path: MATCH path = (o)-[*1..3]->(end) RETURN path',
            doc_hint='FalkorDB supports openCypher variable-length paths with [*min..max] syntax',
        )
        assert err.stage == 'falkordb_dialect'
        assert err.reason == 'apoc_unsupported'
        assert 'variable-length' in err.suggestion
