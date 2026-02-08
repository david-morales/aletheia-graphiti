"""Tests for Cypher validation pipeline and result formatting."""
from __future__ import annotations
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cypher import CypherError, SanitizedQuery, _fix_llm_syntax


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


class TestStage1LLMFixups:
    def test_smart_quotes_replaced(self):
        query = 'MATCH (n) WHERE n.name = \u201cBoeing\u201d RETURN n'
        fixed, fixes = _fix_llm_syntax(query)
        assert '\u201c' not in fixed and '\u201d' not in fixed
        assert '"Boeing"' in fixed
        assert any('smart quote' in f.lower() for f in fixes)

    def test_single_smart_quotes_replaced(self):
        query = "MATCH (n) WHERE n.name = \u2018Boeing\u2019 RETURN n"
        fixed, fixes = _fix_llm_syntax(query)
        assert '\u2018' not in fixed and '\u2019' not in fixed
        assert "'Boeing'" in fixed

    def test_code_block_extraction(self):
        query = "```cypher\nMATCH (n) RETURN n\n```"
        fixed, fixes = _fix_llm_syntax(query)
        assert fixed.strip() == 'MATCH (n) RETURN n'
        assert any('code block' in f.lower() for f in fixes)

    def test_code_block_without_language(self):
        query = "```\nMATCH (n) RETURN n\n```"
        fixed, fixes = _fix_llm_syntax(query)
        assert fixed.strip() == 'MATCH (n) RETURN n'

    def test_no_code_block_passthrough(self):
        query = 'MATCH (n) RETURN n'
        fixed, fixes = _fix_llm_syntax(query)
        assert fixed == query
        assert fixes == []

    def test_multi_word_node_label_quoted(self):
        query = 'MATCH (n:Data Science) RETURN n'
        fixed, _ = _fix_llm_syntax(query)
        assert ':`Data Science`' in fixed

    def test_already_quoted_label_untouched(self):
        query = 'MATCH (n:`Data Science`) RETURN n'
        fixed, fixes = _fix_llm_syntax(query)
        assert fixed == query

    def test_return_injection_simple_match(self):
        query = 'MATCH (o:Occurrence)-[:OPERATED_BY]->(op:Operator)'
        fixed, fixes = _fix_llm_syntax(query)
        assert 'RETURN' in fixed
        assert any('RETURN' in f for f in fixes)

    def test_return_injection_with_scope(self):
        query = 'MATCH (o:Occurrence)-[:OPERATED_BY]->(op:Operator) WITH op, count(o) AS cnt'
        fixed, fixes = _fix_llm_syntax(query)
        assert 'RETURN' in fixed
        return_part = fixed.split('RETURN')[-1]
        assert 'op' in return_part
        assert 'cnt' in return_part

    def test_return_not_injected_when_present(self):
        query = 'MATCH (n) RETURN n'
        fixed, fixes = _fix_llm_syntax(query)
        assert fixed.count('RETURN') == 1
        assert not any('RETURN' in f for f in fixes)

    def test_return_not_injected_for_call(self):
        query = 'CALL db.labels()'
        fixed, fixes = _fix_llm_syntax(query)
        assert 'RETURN' not in fixed

    def test_html_entity_decoding(self):
        query = 'MATCH (n) WHERE n.name = &quot;Boeing&quot; RETURN n'
        fixed, _ = _fix_llm_syntax(query)
        assert '&quot;' not in fixed
        assert '"Boeing"' in fixed

    def test_multiple_fixes_accumulated(self):
        query = '```cypher\nMATCH (n) WHERE n.name = \u201cBoeing\u201d\n```'
        fixed, fixes = _fix_llm_syntax(query)
        assert len(fixes) >= 2  # code block + smart quotes + RETURN injection
