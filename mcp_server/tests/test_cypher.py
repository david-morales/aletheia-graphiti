"""Tests for Cypher validation pipeline and result formatting."""
from __future__ import annotations
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cypher import (
    CypherError,
    SanitizedQuery,
    _check_falkordb_dialect,
    _check_whitelist,
    _fix_falkordb_dialect,
    _fix_llm_syntax,
)


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


class TestStage2Reject:
    def test_reject_apoc(self):
        query = "MATCH (n) CALL apoc.path.expand(n, 'KNOWS>', '', 1, 3) YIELD path RETURN path"
        err = _check_falkordb_dialect(query)
        assert err is not None
        assert err.reason == 'apoc_unsupported'
        assert 'variable-length' in err.suggestion.lower()
        assert err.doc_hint != ''

    def test_reject_pattern_comprehension(self):
        query = 'MATCH (n:Occurrence) RETURN n.name, [(n)-[:INVOLVED_AIRCRAFT]->(a) | a.name] AS aircraft'
        err = _check_falkordb_dialect(query)
        assert err is not None
        assert err.reason == 'pattern_comprehension_unsupported'
        assert 'collect' in err.suggestion.lower()

    def test_reject_exists_subquery(self):
        query = 'MATCH (n:Aircraft) WHERE EXISTS { MATCH (n)<-[:INVOLVED_AIRCRAFT]-(o) } RETURN n'
        err = _check_falkordb_dialect(query)
        assert err is not None
        assert err.reason == 'exists_subquery_unsupported'

    def test_reject_call_subquery(self):
        query = 'MATCH (n:Occurrence) CALL { WITH n MATCH (n)-[:OPERATED_BY]->(op) RETURN op } RETURN n, op'
        err = _check_falkordb_dialect(query)
        assert err is not None
        assert err.reason == 'call_subquery_unsupported'
        assert 'OPTIONAL MATCH' in err.suggestion

    def test_reject_map_projection(self):
        query = 'MATCH (n:Occurrence) RETURN n {.name, .date_value, .description}'
        err = _check_falkordb_dialect(query)
        assert err is not None
        assert err.reason == 'map_projection_unsupported'
        assert 'individually' in err.suggestion.lower()

    def test_pass_clean_query(self):
        query = 'MATCH (n:Occurrence) RETURN n.name, n.date_value'
        assert _check_falkordb_dialect(query) is None

    def test_pass_variable_length_path(self):
        query = 'MATCH path = (a)-[*1..3]->(b) RETURN path'
        assert _check_falkordb_dialect(query) is None

    def test_pass_exists_pattern(self):
        query = 'MATCH (n:Aircraft) WHERE EXISTS((n)<-[:INVOLVED_AIRCRAFT]-()) RETURN n'
        assert _check_falkordb_dialect(query) is None


class TestStage2AutoFix:
    def test_strip_date_function(self):
        query = "MATCH (o:Occurrence) WHERE o.date_value > date('2024-06-01') RETURN o"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert "date(" not in fixed
        assert "'2024-06-01'" in fixed
        assert any('date' in f.lower() for f in fixes)

    def test_strip_datetime_function(self):
        query = "MATCH (o) WHERE o.ts > datetime('2024-06-01T10:00:00') RETURN o"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert "datetime(" not in fixed
        assert "'2024-06-01T10:00:00'" in fixed

    def test_fix_lower_to_toLower(self):
        query = "MATCH (n) WHERE lower(n.name) = 'boeing' RETURN n"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'lower(' not in fixed
        assert 'toLower(' in fixed

    def test_fix_LOWER_to_toLower(self):
        query = "MATCH (n) WHERE LOWER(n.name) = 'boeing' RETURN n"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'LOWER(' not in fixed
        assert 'toLower(' in fixed

    def test_fix_upper_to_toUpper(self):
        query = "MATCH (n) WHERE upper(n.name) = 'BOEING' RETURN n"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'upper(' not in fixed
        assert 'toUpper(' in fixed

    def test_strip_profile_prefix(self):
        query = 'PROFILE MATCH (n:Occurrence) RETURN count(n)'
        fixed, fixes = _fix_falkordb_dialect(query)
        assert not fixed.strip().startswith('PROFILE')
        assert 'MATCH' in fixed

    def test_strip_explain_prefix(self):
        query = 'EXPLAIN MATCH (n:Occurrence) RETURN count(n)'
        fixed, fixes = _fix_falkordb_dialect(query)
        assert not fixed.strip().startswith('EXPLAIN')

    def test_no_fix_needed(self):
        query = 'MATCH (n:Occurrence) RETURN n.name, count(n)'
        fixed, fixes = _fix_falkordb_dialect(query)
        assert fixed == query
        assert fixes == []


class TestStage2Ordering:
    def test_apoc_with_date_rejects_on_apoc(self):
        query = "MATCH (n) WHERE n.date > date('2024-01-01') CALL apoc.path.expand(n, 'KNOWS>') YIELD path RETURN path"
        err = _check_falkordb_dialect(query)
        assert err is not None
        assert err.reason == 'apoc_unsupported'


class TestStage3SecurityWhitelist:
    def test_allow_simple_match_return(self):
        assert _check_whitelist('MATCH (n) RETURN n') is None

    def test_allow_optional_match(self):
        assert _check_whitelist('MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, m') is None

    def test_allow_where_with_order_skip_limit(self):
        query = 'MATCH (n) WHERE n.name = "X" WITH n ORDER BY n.name SKIP 10 LIMIT 20 RETURN n'
        assert _check_whitelist(query) is None

    def test_allow_unwind(self):
        assert _check_whitelist("UNWIND ['a','b'] AS x MATCH (n) WHERE n.name = x RETURN n") is None

    def test_allow_union(self):
        query = 'MATCH (n:Aircraft) RETURN n.name UNION MATCH (n:Operator) RETURN n.name'
        assert _check_whitelist(query) is None

    def test_allow_call_db_labels(self):
        assert _check_whitelist('CALL db.labels()') is None

    def test_allow_call_db_relationshipTypes(self):
        assert _check_whitelist('CALL db.relationshipTypes()') is None

    def test_reject_create(self):
        err = _check_whitelist('CREATE (n:Test {name: "test"})')
        assert err is not None
        assert err.stage == 'security'
        assert 'read-only' in err.explanation.lower()

    def test_reject_delete(self):
        err = _check_whitelist('MATCH (n) DELETE n')
        assert err is not None
        assert err.reason == 'write_operation'

    def test_reject_set(self):
        assert _check_whitelist('MATCH (n) SET n.name = "new"') is not None

    def test_reject_merge(self):
        assert _check_whitelist('MERGE (n:Test {name: "test"})') is not None

    def test_reject_remove(self):
        assert _check_whitelist('MATCH (n) REMOVE n.name') is not None

    def test_reject_drop(self):
        assert _check_whitelist('DROP INDEX ON :Person(name)') is not None

    def test_reject_detach_delete(self):
        assert _check_whitelist('MATCH (n) DETACH DELETE n') is not None

    def test_reject_call_dbms(self):
        assert _check_whitelist('CALL dbms.security.changePassword("new")') is not None

    def test_keywords_case_insensitive(self):
        assert _check_whitelist('match (n) return n') is None
        assert _check_whitelist('create (n:Test)') is not None

    def test_reject_foreach(self):
        assert _check_whitelist('FOREACH (n IN nodes(path) | SET n.visited = true)') is not None

    def test_keyword_inside_string_not_matched(self):
        # "DELETE" inside a string literal should NOT trigger rejection
        assert _check_whitelist('MATCH (n) WHERE n.name = "DELETE ME" RETURN n') is None

    def test_keyword_in_property_not_matched(self):
        # n.description should not trigger on any keyword
        assert _check_whitelist('MATCH (n) RETURN n.description') is None
