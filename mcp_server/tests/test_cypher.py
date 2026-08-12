"""Tests for Cypher validation pipeline and result formatting."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cypher import (
    DEFAULT_LIMIT,
    CypherError,
    SanitizedQuery,
    _check_whitelist,
    _fix_llm_syntax,
    _inject_safety,
    format_error,
    format_result,
    validate_and_sanitize,
)
from flavours.falkordb import (
    FalkorDbFlavour,
    check_falkordb_dialect as _check_falkordb_dialect,
    classify_falkordb_execution_error as classify_execution_error,
    fix_falkordb_dialect as _fix_falkordb_dialect,
)

# FalkorDB flavour for validate_and_sanitize(query, flavour, ..., _FLAVOUR) calls (these tests
# exercise FalkorDB dialect behavior through the now-flavour-driven pipeline).
_FLAVOUR = FalkorDbFlavour()


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

    def test_pass_pattern_comprehension(self):
        # Pattern comprehensions ARE supported in FalkorDB.
        query = 'MATCH (n:Occurrence) RETURN n.name, [(n)-[:INVOLVED_AIRCRAFT]->(a) | a.name] AS aircraft'
        assert _check_falkordb_dialect(query) is None

    def test_reject_exists_subquery(self):
        query = 'MATCH (n:Aircraft) WHERE EXISTS { MATCH (n)<-[:INVOLVED_AIRCRAFT]-(o) } RETURN n'
        err = _check_falkordb_dialect(query)
        assert err is not None
        assert err.reason == 'exists_subquery_unsupported'

    def test_pass_call_subquery(self):
        # CALL {} subqueries ARE supported in FalkorDB.
        query = 'MATCH (n:Occurrence) CALL { WITH n MATCH (n)-[:OPERATED_BY]->(op) RETURN op } RETURN n, op'
        assert _check_falkordb_dialect(query) is None

    def test_pass_map_projection(self):
        # Map projections ARE supported in FalkorDB.
        query = 'MATCH (n:Occurrence) RETURN n {.name, .date_value, .description}'
        assert _check_falkordb_dialect(query) is None

    def test_pass_clean_query(self):
        query = 'MATCH (n:Occurrence) RETURN n.name, n.date_value'
        assert _check_falkordb_dialect(query) is None

    def test_pass_variable_length_path(self):
        query = 'MATCH path = (a)-[*1..3]->(b) RETURN path'
        assert _check_falkordb_dialect(query) is None

    def test_pass_exists_pattern(self):
        query = 'MATCH (n:Aircraft) WHERE EXISTS((n)<-[:INVOLVED_AIRCRAFT]-()) RETURN n'
        assert _check_falkordb_dialect(query) is None


class TestStage2RejectUnwindWhere:
    def test_reject_unwind_where_no_with(self):
        query = (
            'MATCH (p:Persona) WITH p '
            'UNWIND [1,2,3] AS x '
            'WHERE x > 1 '
            'RETURN p, x'
        )
        err = _check_falkordb_dialect(query)
        assert err is not None
        assert err.reason == 'unwind_where_missing_with'
        assert 'WITH' in err.suggestion

    def test_reject_unwind_where_multiline(self):
        query = (
            'MATCH (p) UNWIND p.list AS item\n'
            'WHERE item IS NOT NULL\n'
            'RETURN item'
        )
        err = _check_falkordb_dialect(query)
        assert err is not None
        assert err.reason == 'unwind_where_missing_with'

    def test_pass_unwind_with_where(self):
        # UNWIND ... AS x WITH x WHERE ... is the valid form.
        query = 'UNWIND [1,2,3] AS x WITH x WHERE x > 1 RETURN x'
        assert _check_falkordb_dialect(query) is None

    def test_pass_unwind_alone(self):
        query = 'UNWIND [1,2,3] AS x RETURN x'
        assert _check_falkordb_dialect(query) is None

    def test_pass_unwind_followed_by_match(self):
        query = "UNWIND ['a','b'] AS name MATCH (n {name: name}) RETURN n"
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


class TestStage2AutoFixBareVariable:
    def test_fix_bare_variable_in_optional_match(self):
        query = 'MATCH (p:Persona) OPTIONAL MATCH parte-[:TIPIFICADO_COMO]->(tipo:TipoDelito) RETURN p, tipo'
        fixed, fixes = _fix_falkordb_dialect(query)
        assert '(parte)-[:TIPIFICADO_COMO]->' in fixed
        assert 'parte-[:TIPIFICADO_COMO]->' not in fixed
        assert any('bare variable' in f.lower() for f in fixes)

    def test_fix_bare_variable_in_match(self):
        query = 'MATCH a-[:KNOWS]->(b) RETURN a, b'
        fixed, fixes = _fix_falkordb_dialect(query)
        assert '(a)-[:KNOWS]->' in fixed
        assert any('bare variable' in f.lower() for f in fixes)

    def test_fix_bare_variable_idempotent(self):
        # Applying twice produces the same result.
        query = 'OPTIONAL MATCH x-[:R]->(y) RETURN x, y'
        fixed_once, _ = _fix_falkordb_dialect(query)
        fixed_twice, fixes_twice = _fix_falkordb_dialect(fixed_once)
        assert fixed_once == fixed_twice
        assert fixes_twice == []  # second pass finds nothing to fix

    def test_no_fix_when_var_already_parenthesized(self):
        query = 'MATCH (a)-[:KNOWS]->(b) RETURN a, b'
        fixed, fixes = _fix_falkordb_dialect(query)
        assert fixed == query
        assert not any('bare variable' in f.lower() for f in fixes)

    def test_no_fix_when_no_pattern_match(self):
        # MATCH with parens and a WHERE — must not be mangled.
        query = 'MATCH (a:Persona) WHERE a.name = "X" RETURN a'
        fixed, fixes = _fix_falkordb_dialect(query)
        assert fixed == query

    def test_comma_separated_patterns_second_var_unchanged(self):
        # Known limitation: regex only matches a bare variable immediately
        # after MATCH.  Second-position bare variable in comma-separated
        # patterns is left alone.  Documents the limitation explicitly so a
        # future regex change cannot silently introduce an unsafe rewrite
        # for this case without updating the test.
        query = 'MATCH (a)-[:R1]->(b), c-[:R2]->(d) RETURN a, b, c, d'
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'c-[:R2]->' in fixed
        assert '(c)-[:R2]->' not in fixed
        assert fixes == []


class TestStage2AutoFixNonAscii:
    def test_fix_non_ascii_alias(self):
        query = "MATCH (p:Persona) WITH p, toInteger(substring(p.fecha, 6, 4)) AS año_nacimiento RETURN año_nacimiento"
        fixed, fixes = _fix_falkordb_dialect(query)
        # ñ → n (1:1 transliteration via str.maketrans)
        assert 'ano_nacimiento' in fixed
        assert 'año_nacimiento' not in fixed
        assert any('non-ascii' in f.lower() or 'transliterat' in f.lower() for f in fixes)

    def test_non_ascii_in_string_literal_preserved(self):
        query = "MATCH (n) WHERE n.name CONTAINS 'INTIMIDACIÓN' RETURN n AS año"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'INTIMIDACIÓN' in fixed  # string literal preserved
        assert 'AS ano' in fixed  # alias transliterated (ñ → n)
        assert 'año' not in fixed.replace("'INTIMIDACIÓN'", '')  # outside string is clean

    def test_no_fix_when_all_ascii(self):
        query = 'MATCH (n:Persona) RETURN n.name AS nombre'
        fixed, fixes = _fix_falkordb_dialect(query)
        assert fixed == query
        assert not any('non-ascii' in f.lower() or 'transliterat' in f.lower() for f in fixes)

    def test_fix_idempotent(self):
        query = "WITH n AS señal RETURN señal"
        fixed_once, _ = _fix_falkordb_dialect(query)
        fixed_twice, fixes2 = _fix_falkordb_dialect(fixed_once)
        assert fixed_once == fixed_twice
        assert not any('non-ascii' in f.lower() or 'transliterat' in f.lower() for f in fixes2)

    def test_multiple_non_ascii_aliases(self):
        query = "WITH 1 AS año, 2 AS señal, 3 AS código RETURN año, señal, código"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'ano' in fixed  # ñ → n
        assert 'senal' in fixed  # ñ → n
        assert 'codigo' in fixed  # ó → o
        assert 'año' not in fixed
        assert 'señal' not in fixed
        assert 'código' not in fixed


class TestStage2AutoFixNotInList:
    def test_fix_property_not_in_list(self):
        # Most common case: `prop.x NOT IN [a, b, c]` rewritten to
        # `NOT (prop.x IN [a, b, c])`. Semantically equivalent, parses cleanly.
        query = "MATCH (n) WHERE n.kind NOT IN ['a', 'b', 'c'] RETURN n LIMIT 1"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert "NOT (n.kind IN ['a', 'b', 'c'])" in fixed
        assert "NOT IN" not in fixed
        assert any('NOT IN' in f and 'NOT (' in f for f in fixes)

    def test_fix_simple_var_not_in_list(self):
        query = "WITH 1 AS x WHERE x NOT IN [1, 2, 3] RETURN x"
        fixed, _ = _fix_falkordb_dialect(query)
        assert "NOT (x IN [1, 2, 3])" in fixed

    def test_fix_inside_case(self):
        # The actual field-observed shape: `count(CASE WHEN x NOT IN [...] THEN 1 END)`.
        query = (
            "MATCH (o) RETURN count(CASE WHEN o.kind NOT IN ['a', 'b'] "
            "THEN 1 ELSE 0 END) AS c"
        )
        fixed, _ = _fix_falkordb_dialect(query)
        assert "NOT (o.kind IN ['a', 'b'])" in fixed
        assert "NOT IN" not in fixed

    def test_already_correct_form_unchanged(self):
        # Don't double-wrap an already-correct `NOT (x IN [...])` form.
        query = "MATCH (n) WHERE NOT (n.kind IN ['a', 'b']) RETURN n LIMIT 1"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert fixed == query
        assert not any('NOT IN' in f for f in fixes)

    def test_not_in_inside_string_literal_preserved(self):
        # Don't rewrite text inside string literals.
        query = "MATCH (n) WHERE n.note = 'x NOT IN [a, b]' RETURN n LIMIT 1"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert "'x NOT IN [a, b]'" in fixed
        assert not any('NOT IN' in f for f in fixes)

    def test_complex_expression_falls_through(self):
        # Function-call or compound expressions on the LHS are out of scope
        # for this regex auto-fix (would need balanced-paren matching). The
        # query must be left untouched so the Layer-3 classifier handles it
        # with the actionable error message.
        query = "MATCH (n) WHERE coalesce(n.x, n.y) NOT IN ['a'] RETURN n LIMIT 1"
        fixed, _ = _fix_falkordb_dialect(query)
        assert fixed == query

    def test_fix_idempotent(self):
        query = "MATCH (n) WHERE n.x NOT IN [1, 2] RETURN n LIMIT 1"
        fixed_once, _ = _fix_falkordb_dialect(query)
        fixed_twice, fixes2 = _fix_falkordb_dialect(fixed_once)
        assert fixed_once == fixed_twice
        assert not any('NOT IN' in f for f in fixes2)


class TestStage2AutoFixNotEquals:
    def test_fix_simple_not_equals(self):
        # FalkorDB rejects `!=` outright; only `<>` is accepted. The fixer
        # rewrites bare `!=` to `<>` so the query reaches FalkorDB cleanly.
        query = "MATCH (n) WITH n LIMIT 1 RETURN count(CASE WHEN n.x != 'None' THEN 1 END) AS c"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert "<>" in fixed
        assert "!=" not in fixed
        assert any('!=' in f and '<>' in f for f in fixes)

    def test_fix_not_equals_in_where(self):
        query = "MATCH (n) WHERE n.x != 'foo' RETURN n LIMIT 1"
        fixed, _ = _fix_falkordb_dialect(query)
        assert "n.x <> 'foo'" in fixed
        assert "!=" not in fixed

    def test_not_equals_inside_string_literal_preserved(self):
        # The literal `!=` must NOT be replaced inside a string.
        query = "MATCH (n) WHERE n.label = 'a != b' RETURN n LIMIT 1"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert "'a != b'" in fixed
        # No fix entry — the operator never appeared outside a string.
        assert not any('!=' in f for f in fixes)

    def test_fix_idempotent(self):
        query = "MATCH (n) WHERE n.x != 1 RETURN n LIMIT 1"
        fixed_once, _ = _fix_falkordb_dialect(query)
        fixed_twice, fixes2 = _fix_falkordb_dialect(fixed_once)
        assert fixed_once == fixed_twice
        assert not any('!=' in f for f in fixes2)


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

    def test_allow_call_subquery_block(self):
        # CALL { ... } subquery — no procedure name after CALL, so the
        # security whitelist should not treat it as a forbidden procedure.
        query = 'MATCH (n) CALL { WITH n MATCH (n)-[r]->(m) RETURN m } RETURN n, m'
        assert _check_whitelist(query) is None


class TestStage4SafetyInjection:
    def test_inject_limit_when_missing(self):
        query = 'MATCH (n) RETURN n'
        fixed, fixes, effective_limit = _inject_safety(query)
        assert f'LIMIT {DEFAULT_LIMIT + 1}' in fixed
        assert any('LIMIT' in f for f in fixes)
        assert effective_limit == DEFAULT_LIMIT

    def test_no_inject_when_limit_present(self):
        query = 'MATCH (n) RETURN n LIMIT 50'
        fixed, fixes, effective_limit = _inject_safety(query)
        assert fixed == query
        assert fixes == []
        assert effective_limit == 50

    def test_no_inject_when_limit_present_lowercase(self):
        query = 'MATCH (n) RETURN n limit 50'
        fixed, fixes, effective_limit = _inject_safety(query)
        assert fixed == query
        assert effective_limit == 50

    def test_limit_appended_after_order_by(self):
        query = 'MATCH (n) RETURN n ORDER BY n.name'
        fixed, fixes, effective_limit = _inject_safety(query)
        assert fixed.endswith(f'LIMIT {DEFAULT_LIMIT + 1}')
        assert 'ORDER BY' in fixed
        assert effective_limit == DEFAULT_LIMIT

    def test_limit_with_skip(self):
        query = 'MATCH (n) RETURN n SKIP 10'
        fixed, fixes, effective_limit = _inject_safety(query)
        assert f'LIMIT {DEFAULT_LIMIT + 1}' in fixed
        assert effective_limit == DEFAULT_LIMIT

    def test_call_query_no_limit(self):
        query = 'CALL db.labels()'
        fixed, fixes, effective_limit = _inject_safety(query)
        assert 'LIMIT' not in fixed
        assert effective_limit == DEFAULT_LIMIT


class TestPipelineOrchestration:
    def test_clean_query_passes(self):
        result = validate_and_sanitize('MATCH (n:Occurrence) RETURN n.name LIMIT 10', _FLAVOUR)
        assert isinstance(result, SanitizedQuery)
        assert result.auto_fixes == []
        assert result.effective_limit == 10

    def test_fixable_query_returns_fixes(self):
        result = validate_and_sanitize(
            "MATCH (o:Occurrence) WHERE o.date_value > date('2024-06-01') RETURN o", _FLAVOUR)
        assert isinstance(result, SanitizedQuery)
        assert any('date' in f.lower() for f in result.auto_fixes)
        assert result.effective_limit == DEFAULT_LIMIT

    def test_write_query_rejected(self):
        result = validate_and_sanitize('CREATE (n:Test {name: "test"})', _FLAVOUR)
        assert isinstance(result, CypherError)
        assert result.stage == 'security'

    def test_apoc_rejected_before_date_fix(self):
        result = validate_and_sanitize(
            "MATCH (n) WHERE n.date > date('2024-01-01') CALL apoc.path.expand(n, 'KNOWS>') YIELD path RETURN path", _FLAVOUR)
        assert isinstance(result, CypherError)
        assert result.reason == 'apoc_unsupported'

    def test_smart_quotes_fixed_then_dialect_fixed(self):
        result = validate_and_sanitize(
            'MATCH (o) WHERE o.name = \u201cBoeing\u201d AND o.date > date(\u20182024-01-01\u2019) RETURN o', _FLAVOUR)
        assert isinstance(result, SanitizedQuery)
        assert len(result.auto_fixes) >= 2

    def test_limit_injected_on_clean_query(self):
        result = validate_and_sanitize('MATCH (n) RETURN n', _FLAVOUR)
        assert isinstance(result, SanitizedQuery)
        assert 'LIMIT' in result.query
        assert any('LIMIT' in f for f in result.auto_fixes)

    def test_existing_limit_preserved(self):
        result = validate_and_sanitize('MATCH (n) RETURN n LIMIT 50', _FLAVOUR)
        assert isinstance(result, SanitizedQuery)
        assert 'LIMIT 50' in result.query
        assert not any('LIMIT' in f for f in result.auto_fixes)

    def test_code_block_plus_missing_return_plus_limit(self):
        result = validate_and_sanitize('```cypher\nMATCH (n:Occurrence)\n```', _FLAVOUR)
        assert isinstance(result, SanitizedQuery)
        assert 'RETURN' in result.query
        assert 'LIMIT' in result.query
        assert len(result.auto_fixes) >= 2


class TestResultFormatter:
    def test_scalar_single_value(self):
        records = [{'count': 47}]
        header = ['count']
        result = format_result(records, header, 'MATCH (n) RETURN count(n)', [], 5.0, 200)
        assert result['type'] == 'scalar'
        assert result['result'] == 47
        assert result['row_count'] == 1
        assert result['truncated'] is False

    def test_scalar_null_result(self):
        records = []
        header = []
        result = format_result(records, header, 'MATCH (n) RETURN count(n)', [], 1.0, 200)
        assert result['type'] == 'scalar'
        assert result['result'] is None
        assert result['row_count'] == 0

    def test_tabular_multiple_rows(self):
        records = [
            {'name': 'Boeing 737', 'count': 12},
            {'name': 'Airbus A320', 'count': 8},
        ]
        header = ['name', 'count']
        result = format_result(records, header, 'MATCH ...', [], 3.0, 200)
        assert result['type'] == 'tabular'
        assert result['columns'] == ['name', 'count']
        assert result['rows'] == [['Boeing 737', 12], ['Airbus A320', 8]]
        assert result['row_count'] == 2
        assert result['truncated'] is False

    def test_tabular_token_efficient_no_repeated_keys(self):
        records = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        header = ['a', 'b']
        result = format_result(records, header, 'MATCH ...', [], 1.0, 200)
        assert isinstance(result['rows'][0], list)
        assert isinstance(result['rows'][1], list)

    def test_truncation_n_plus_1(self):
        records = [{'n': i} for i in range(201)]
        header = ['n']
        result = format_result(records, header, 'MATCH ...', [], 10.0, 200)
        assert result['truncated'] is True
        assert result['row_count'] == 200
        assert len(result['rows']) == 200

    def test_no_truncation_exact_limit(self):
        records = [{'n': i} for i in range(200)]
        header = ['n']
        result = format_result(records, header, 'MATCH ...', [], 10.0, 200)
        assert result['truncated'] is False
        assert result['row_count'] == 200

    def test_no_truncation_under_limit(self):
        records = [{'n': i} for i in range(50)]
        header = ['n']
        result = format_result(records, header, 'MATCH ...', [], 10.0, 200)
        assert result['truncated'] is False
        assert result['row_count'] == 50

    def test_metadata_envelope_present(self):
        records = [{'count': 5}]
        header = ['count']
        fixes = ['Injected LIMIT 200']
        result = format_result(records, header, 'MATCH (n) RETURN count(n) LIMIT 201', fixes, 23.0, 200)
        assert result['query'] == 'MATCH (n) RETURN count(n) LIMIT 201'
        assert result['auto_fixes'] == fixes
        assert result['execution_ms'] == 23.0
        assert result['limit_applied'] == 200

    def test_single_row_multiple_columns_is_tabular(self):
        records = [{'name': 'Boeing', 'count': 5}]
        header = ['name', 'count']
        result = format_result(records, header, 'MATCH ...', [], 1.0, 200)
        assert result['type'] == 'tabular'

    def test_error_envelope_structure(self):
        err = CypherError(
            stage='security',
            reason='write_operation',
            found='CREATE',
            explanation='Write not allowed.',
            suggestion='Use MATCH instead.',
        )
        result = format_error('CREATE (n:Test)', err)
        assert result['type'] == 'error'
        assert result['query'] == 'CREATE (n:Test)'
        # ADR-015 R4: top-level `error` is a STRING; structured detail is additive.
        assert isinstance(result['error'], str)
        assert result['error'] == 'Write not allowed.'
        assert result['hint'] == 'Use MATCH instead.'
        assert result['error_detail']['stage'] == 'security'
        assert result['execution_ms'] == 0


# ---------------------------------------------------------------------------
# Tool description & server instruction tests
# ---------------------------------------------------------------------------


def _make_test_profile():
    entity_types = {
        'Occurrence': SimpleNamespace(
            label='Occurrence', count=10, description='Aviation incident',
            sample_names=['2024-0975-EU'], hierarchy=False,
        ),
        'Aircraft': SimpleNamespace(
            label='Aircraft', count=15, description='Aircraft entity',
            sample_names=['Boeing 737'], hierarchy=False,
        ),
        'Operator': SimpleNamespace(
            label='Operator', count=8, description='Airline operator',
            sample_names=['KLM'], hierarchy=False,
        ),
    }
    edge_types = {
        'INVOLVED_AIRCRAFT': SimpleNamespace(
            name='INVOLVED_AIRCRAFT', count=12, description='',
            source_target_pattern='Occurrence -> Aircraft',
        ),
        'OPERATED_BY': SimpleNamespace(
            name='OPERATED_BY', count=10, description='',
            source_target_pattern='Occurrence -> Operator',
        ),
    }

    class FakeProfile:
        def __init__(self):
            self.group_id = 'aviation_safety_quality'
            self.entity_types = entity_types
            self.edge_types = edge_types
            self.time_range = ('2024-01-01', '2024-12-31')

        def entity_type_names(self):
            return sorted(self.entity_types.keys())

        def storage_entity_type_names(self):
            return sorted(
                label for label, info in self.entity_types.items() if not info.hierarchy
            )

        def edge_type_names(self):
            return sorted(self.edge_types.keys())

    return FakeProfile()


class TestToolDescriptions:
    def test_get_schema_description_includes_graph_name(self):
        from tool_descriptions import build_get_schema_description

        desc = build_get_schema_description(_make_test_profile())
        assert 'aviation_safety_quality' in desc

    def test_get_schema_description_lists_entity_types(self):
        from tool_descriptions import build_get_schema_description

        desc = build_get_schema_description(_make_test_profile())
        assert 'Occurrence' in desc
        assert 'Aircraft' in desc

    def test_run_cypher_description_includes_dialect_cheatsheet(self):
        # ADR-019 R6: the dialect cheatsheet is now flavour-driven (sourced from the flavour's
        # dialect_summary), not hardcoded. Passing the FalkorDB flavour surfaces its notes.
        from tool_descriptions import build_graph_query_description
        from flavours.falkordb import FalkorDbFlavour

        desc = build_graph_query_description(_make_test_profile(), FalkorDbFlavour())
        assert 'FalkorDB' in desc
        assert 'APOC' in desc

    def test_run_cypher_description_no_flavour_omits_dialect_cheatsheet(self):
        # Without a flavour there is no per-backend dialect block (it is no longer hardcoded).
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(_make_test_profile())
        assert 'Dialect notes:' not in desc

    def test_run_cypher_description_includes_examples(self):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(_make_test_profile())
        assert 'Example' in desc or 'example' in desc
        assert 'MATCH' in desc

    def test_run_cypher_description_includes_chained_workflow(self):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(_make_test_profile())
        assert 'search' in desc.lower()

    def test_run_cypher_description_includes_where_in_example(self):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(_make_test_profile())
        assert 'IN [' in desc or 'found via search' in desc.lower()


class TestServerInstructions:
    def test_instructions_mention_get_schema(self):
        from tool_descriptions import build_instructions

        assert 'get_schema' in build_instructions(_make_test_profile())

    def test_instructions_mention_run_cypher(self):
        from tool_descriptions import build_instructions

        assert 'graph_query' in build_instructions(_make_test_profile())

    def test_instructions_mention_chained_workflow(self):
        from tool_descriptions import build_instructions

        instructions = build_instructions(_make_test_profile())
        assert 'search' in instructions.lower()
        assert 'cypher' in instructions.lower() or 'graph_query' in instructions.lower()

    def test_instructions_mention_semantic_vs_analytical(self):
        from tool_descriptions import build_instructions

        instructions = build_instructions(_make_test_profile())
        assert 'count' in instructions.lower() or 'aggregat' in instructions.lower()


# ---------------------------------------------------------------------------
# graph_query tool tests
# ---------------------------------------------------------------------------


class TestRunCypher:
    """graph_query tool: end-to-end Cypher execution."""

    @pytest.mark.asyncio
    async def test_simple_count_query(self):
        from graphiti_mcp_server import graph_query

        # Mock the FalkorDB graph's ro_query
        mock_graph = AsyncMock()
        mock_query_result = MagicMock()
        mock_query_result.header = [('string', 'cnt')]
        mock_query_result.result_set = [[47]]
        mock_graph.ro_query = AsyncMock(return_value=mock_query_result)

        mock_driver = MagicMock()
        mock_driver._get_graph = MagicMock(return_value=mock_graph)
        mock_driver._database = 'test_db'

        mock_client = MagicMock()
        mock_client.driver = mock_driver

        mock_svc = AsyncMock()
        mock_svc.flavour = _FLAVOUR
        mock_svc.get_client = AsyncMock(return_value=mock_client)
        mock_svc.config = MagicMock()
        mock_svc.config.graphiti.group_id = 'test_graph'

        with patch('graphiti_mcp_server.graphiti_service', mock_svc):
            result = await graph_query(query='MATCH (n) RETURN count(n) AS cnt')

        assert result['type'] == 'scalar'
        assert result['result'] == 47

    @pytest.mark.asyncio
    async def test_write_query_rejected(self):
        from graphiti_mcp_server import graph_query

        mock_svc = AsyncMock()
        mock_svc.flavour = _FLAVOUR
        mock_svc.config = MagicMock()
        mock_svc.config.graphiti.group_id = 'test_graph'

        with patch('graphiti_mcp_server.graphiti_service', mock_svc):
            result = await graph_query(query='CREATE (n:Test {name: "test"})')

        assert result['type'] == 'error'
        assert result['error_detail']['stage'] == 'security'

    @pytest.mark.asyncio
    async def test_auto_fixes_in_response(self):
        from graphiti_mcp_server import graph_query

        mock_graph = AsyncMock()
        mock_query_result = MagicMock()
        mock_query_result.header = [('string', 'cnt')]
        mock_query_result.result_set = [[10]]
        mock_graph.ro_query = AsyncMock(return_value=mock_query_result)

        mock_driver = MagicMock()
        mock_driver._get_graph = MagicMock(return_value=mock_graph)
        mock_driver._database = 'test_db'

        mock_client = MagicMock()
        mock_client.driver = mock_driver

        mock_svc = AsyncMock()
        mock_svc.flavour = _FLAVOUR
        mock_svc.get_client = AsyncMock(return_value=mock_client)
        mock_svc.config = MagicMock()
        mock_svc.config.graphiti.group_id = 'test_graph'

        with patch('graphiti_mcp_server.graphiti_service', mock_svc):
            result = await graph_query(
                query="MATCH (o) WHERE o.date > date('2024-01-01') RETURN count(o) AS cnt"
            )

        assert 'auto_fixes' in result
        assert len(result['auto_fixes']) >= 1  # At least date fix

    @pytest.mark.asyncio
    async def test_execution_error_returns_error_type(self):
        from graphiti_mcp_server import graph_query

        mock_graph = AsyncMock()
        mock_graph.ro_query = AsyncMock(side_effect=Exception("Unknown function 'foo'"))

        mock_driver = MagicMock()
        mock_driver._get_graph = MagicMock(return_value=mock_graph)
        mock_driver._database = 'test_db'

        mock_client = MagicMock()
        mock_client.driver = mock_driver

        mock_svc = AsyncMock()
        mock_svc.flavour = _FLAVOUR
        mock_svc.get_client = AsyncMock(return_value=mock_client)
        mock_svc.config = MagicMock()
        mock_svc.config.graphiti.group_id = 'test_graph'

        with patch('graphiti_mcp_server.graphiti_service', mock_svc):
            result = await graph_query(query='MATCH (n) RETURN foo(n)')

        assert result['type'] == 'error'
        assert result['error_detail']['stage'] == 'execution'
        assert 'auto_fixes' in result

    @pytest.mark.asyncio
    async def test_service_not_ready_returns_error(self):
        from graphiti_mcp_server import graph_query

        with patch('graphiti_mcp_server.graphiti_service', None):
            result = await graph_query(query='MATCH (n) RETURN n')

        assert result['type'] == 'error'
        assert result['error_detail']['stage'] == 'initialization'


# ---------------------------------------------------------------------------
# get_schema tool tests
# ---------------------------------------------------------------------------


def make_mock_driver():
    """Create a mock FalkorDB driver for schema queries."""
    driver = AsyncMock()

    async def execute_query(query, **kwargs):
        if 'labels(n) AS lbls' in query:
            return [
                {'lbls': ['Entity', 'Occurrence'], 'cnt': 10},
                {'lbls': ['Entity', 'Aircraft'], 'cnt': 15},
                {'lbls': ['Episodic'], 'cnt': 100},  # Should be filtered
            ], ['lbls', 'cnt'], None
        elif 'type(r) AS rel_type' in query:
            return [
                {'rel_type': 'INVOLVED_AIRCRAFT', 'cnt': 12},
                {'rel_type': 'RELATES_TO', 'cnt': 50},
            ], ['rel_type', 'cnt'], None
        elif 'keys(n)' in query:
            return [{'key': 'name'}, {'key': 'date_value'}, {'key': 'id_value'}], ['key'], None
        elif 'source_labels' in query:
            return [
                {'source_labels': ['Entity', 'Occurrence'], 'target_labels': ['Entity', 'Aircraft']},
            ], ['source_labels', 'target_labels'], None
        return [], [], None

    driver.execute_query = execute_query
    return driver


def _make_mock_schema_service():
    """Create a mock GraphitiService wired to the mock driver for schema tests."""
    mock_client = MagicMock()
    mock_client.driver = make_mock_driver()

    mock_svc = AsyncMock()
    mock_svc.flavour = _FLAVOUR
    mock_svc.get_client = AsyncMock(return_value=mock_client)
    mock_svc._schema_cache = None
    mock_svc._schema_dirty = True
    mock_svc.config = MagicMock()
    mock_svc.config.graphiti.group_id = 'test_graph'
    return mock_svc


class TestGetSchema:
    """get_schema tool: structural schema discovery with caching."""

    @pytest.mark.asyncio
    async def test_schema_filters_entity_label(self):
        """Schema response should NOT include the generic :Entity label."""
        from graphiti_mcp_server import get_schema

        mock_svc = _make_mock_schema_service()

        with patch('graphiti_mcp_server.graphiti_service', mock_svc):
            result = await get_schema()

        assert 'Entity' not in result.get('node_labels', {})
        assert 'Occurrence' in result.get('node_labels', {})
        assert 'Aircraft' in result.get('node_labels', {})

    @pytest.mark.asyncio
    async def test_schema_filters_entity_from_patterns(self):
        """Relationship patterns should NOT include :Entity in source/target labels."""
        from graphiti_mcp_server import get_schema

        mock_svc = _make_mock_schema_service()

        with patch('graphiti_mcp_server.graphiti_service', mock_svc):
            result = await get_schema()

        for rel_info in result.get('relationship_types', {}).values():
            for pattern in rel_info.get('patterns', []):
                assert 'Entity' not in pattern

    @pytest.mark.asyncio
    async def test_schema_cache_returns_without_querying(self):
        """Second call should return cached result without hitting the driver."""
        from graphiti_mcp_server import get_schema

        mock_svc = _make_mock_schema_service()

        with patch('graphiti_mcp_server.graphiti_service', mock_svc):
            first = await get_schema()

            # After the first call the service should have cached and cleared dirty
            assert mock_svc._schema_dirty is False
            assert mock_svc._schema_cache is not None

            # Reset the mock so we can verify no new calls are made
            mock_client = await mock_svc.get_client()
            mock_client.driver.execute_query = AsyncMock(side_effect=AssertionError(
                'Driver should not be called when cache is clean'
            ))

            second = await get_schema()

        assert first == second

    @pytest.mark.asyncio
    async def test_schema_includes_tool_capabilities(self):
        """get_schema response includes a tool_capabilities section."""
        from graphiti_mcp_server import get_schema

        mock_svc = _make_mock_schema_service()

        with patch('graphiti_mcp_server.graphiti_service', mock_svc):
            result = await get_schema()

        caps = result.get('tool_capabilities')
        assert caps is not None, "get_schema must include tool_capabilities"
        assert 'search' in caps
        assert 'graph_query' in caps
        assert 'explore_entity' in caps

        # search: must declare what it covers and doesn't cover
        search = caps['search']
        assert 'covers' in search
        assert 'does_not_cover' in search
        assert 'search_methods' in search
        assert 'name' in search['covers'].get('entity_fields', [])
        assert 'summary' in search['covers'].get('entity_fields', [])

        # graph_query: must declare full property access
        cypher = caps['graph_query']
        assert 'all_properties' in cypher['covers'].get('entity_fields', [])

        # explore_entity: must declare graph traversal
        explore = caps['explore_entity']
        assert explore['covers'].get('neighborhood') is True


# ---------------------------------------------------------------------------
# Integration smoke tests (require running FalkorDB)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_falkordb
class TestCypherIntegration:
    """Integration tests requiring a running FalkorDB instance."""

    @pytest.mark.asyncio
    async def test_get_schema_real_graph(self):
        """get_schema returns valid schema from a real graph."""
        pytest.skip('Integration test — run manually with FalkorDB')

    @pytest.mark.asyncio
    async def test_run_cypher_count(self):
        """graph_query executes a count query and returns scalar."""
        pytest.skip('Integration test — run manually with FalkorDB')

    @pytest.mark.asyncio
    async def test_run_cypher_ro_enforcement(self):
        """Write queries are rejected at both pipeline and database level."""
        pytest.skip('Integration test — run manually with FalkorDB')


class TestCypherQualityInEnvelope:
    """Verify cypher_quality field appears in format_result output."""

    def test_quality_field_present(self):
        records = [{'name': 'Alice'}]
        header = ['name']
        result = format_result(
            records, header,
            query="MATCH (n:Evento) RETURN n.name",
            auto_fixes=[],
            execution_ms=5.0,
            limit=200,
            schema={'node_labels': {'Evento': {'properties': ['name']}}, 'relationship_types': {}},
        )
        assert 'cypher_quality' in result
        assert result['cypher_quality']['verdict'] == 'success'

    def test_quality_schema_mismatch(self):
        records = []
        header = []
        result = format_result(
            records, header,
            query="MATCH (n:Fake) RETURN n.name",
            auto_fixes=[],
            execution_ms=5.0,
            limit=200,
            schema={'node_labels': {'Evento': {'properties': ['name']}}, 'relationship_types': {}},
        )
        assert result['cypher_quality']['verdict'] == 'schema_mismatch'

    def test_quality_without_schema(self):
        records = [{'x': 1}]
        header = ['x']
        result = format_result(
            records, header,
            query="MATCH (n) RETURN count(n) AS x",
            auto_fixes=[],
            execution_ms=5.0,
            limit=200,
            schema=None,
        )
        assert result['cypher_quality']['verdict'] == 'success'

    def test_quality_empty_legit(self):
        """Empty result with valid schema -> empty_legit verdict."""
        records = []
        header = []
        result = format_result(
            records, header,
            query="MATCH (n:Evento) RETURN n.name",
            auto_fixes=[],
            execution_ms=5.0,
            limit=200,
            schema={'node_labels': {'Evento': {'properties': ['name']}}, 'relationship_types': {}},
        )
        assert result['cypher_quality']['verdict'] == 'empty_legit'


class TestCypherQualityInErrorEnvelope:
    """Verify cypher_quality field appears in format_error output."""

    def test_rejected_query_has_quality(self):
        err = CypherError(
            stage='falkordb_dialect', reason='apoc_unsupported',
            found='apoc.path.expand()', explanation='Not supported',
            suggestion='Use paths', doc_hint='',
        )
        result = format_error("MATCH ... apoc.path.expand() ...", err)
        assert 'cypher_quality' in result
        assert result['cypher_quality']['outcome'] == 'rejected'
        assert result['cypher_quality']['verdict'] == 'rejected'

    def test_execution_error_has_quality(self):
        err = CypherError(
            stage='execution', reason='query_failed',
            found='SyntaxError', explanation='Bad syntax',
            suggestion='Fix it', doc_hint='',
        )
        result = format_error("BAD QUERY", err)
        assert result['cypher_quality']['outcome'] == 'error'
        assert result['cypher_quality']['verdict'] == 'error'

    def test_security_rejection_has_quality(self):
        err = CypherError(
            stage='security', reason='write_operation',
            found='DELETE', explanation='Not allowed',
            suggestion='Use MATCH', doc_hint='',
        )
        result = format_error("MATCH (n) DELETE n", err)
        assert result['cypher_quality']['outcome'] == 'rejected'


class TestClassifyExecutionError:
    def test_classify_where_needs_with_unwind_shape(self):
        # FalkorDB emits the same parser error for both UNWIND-WHERE and
        # RETURN-WHERE: `expected WITH ... errCtx: WHERE ...`. The classifier
        # can't see the query, so it emits a single reason covering both.
        msg = "errMsg: Invalid input 'H': expected WITH line: 10, column: 2 errCtx: WHERE evento.numero"
        err = classify_execution_error(msg)
        assert err.reason == 'where_needs_with'
        assert 'WITH' in err.suggestion
        assert err.doc_hint != ''

    def test_classify_where_needs_with_return_shape(self):
        # Observed in the field: LLM wrote `RETURN <projections> WHERE alias IS NOT NULL`.
        # FalkorDB emits the same `expected WITH ... errCtx: WHERE` message as for
        # the UNWIND case, and the classifier used to mislabel it as UNWIND-specific.
        # Suggestion must now mention RETURN explicitly so the LLM retry converges.
        msg = (
            "errMsg: Invalid input 'H': expected WITH line: 33, column: 2, offset: 1376 "
            "errCtx: WHERE alias_name IS NOT NULL errCtxOffset: 1"
        )
        err = classify_execution_error(msg)
        assert err.reason == 'where_needs_with'
        assert 'RETURN' in err.suggestion
        assert 'WITH' in err.suggestion

    def test_classify_bare_variable(self):
        msg = "errMsg: Invalid input '-': expected '=' line: 5, column: 21 errCtx: OPTIONAL MATCH parte-[:R]->(t)"
        err = classify_execution_error(msg)
        assert err.reason == 'bare_variable_in_pattern'
        assert 'paren' in err.suggestion.lower()

    def test_classify_unknown_error_returns_generic(self):
        msg = "errMsg: some completely unknown error that we haven't seen"
        err = classify_execution_error(msg)
        assert err.stage == 'execution'
        assert err.reason == 'query_failed'
        assert 'syntax' in err.suggestion.lower()

    def test_classify_non_ascii_identifier(self):
        msg = "errMsg: Invalid input '\ufffd': expected ',' errCtx: AS año_nacimiento"
        err = classify_execution_error(msg)
        assert err.reason == 'non_ascii_identifier'
        assert 'ASCII' in err.suggestion

    def test_classify_round_arity_mismatch(self):
        # Observed verbatim from FalkorDB on `round(avg(x), 2)` (Round 2 Q2).
        msg = "Received 2 arguments to function 'round', expected at most 1"
        err = classify_execution_error(msg)
        assert err.reason == 'function_arity_mismatch'
        assert 'round' in err.suggestion.lower() or '100.0' in err.suggestion

    def test_classify_arity_mismatch_other_functions(self):
        # Pattern is generic — the same error shape for any FalkorDB/Neo4j
        # arity delta should still classify correctly.
        msg = "Received 3 arguments to function 'size', expected at most 1"
        err = classify_execution_error(msg)
        assert err.reason == 'function_arity_mismatch'
        assert 'arit' in err.suggestion.lower() or 'signature' in err.suggestion.lower()

    def test_classify_variable_not_in_scope(self):
        # Observed in the field: LLM wrote a UNION ALL where the right
        # side referenced variables defined only on the left side. FalkorDB
        # responds with `'<var>' not defined`. The classifier used to fall
        # through to the generic `query_failed` envelope; now it should give
        # explicit guidance about UNION scope and WITH carry-through.
        msg = "'x' not defined"
        err = classify_execution_error(msg)
        assert err.reason == 'variable_not_in_scope'
        assert 'UNION' in err.suggestion
        assert 'WITH' in err.suggestion
        assert err.doc_hint != ''

    def test_classify_not_equals_operator(self):
        # FalkorDB rejects `!=` with `Invalid input '!'` and the expected-token
        # list including `<>`. Layer-2 auto-fix should normally catch this
        # before execution, but defense-in-depth at Layer 3 still gives a
        # clear message if the auto-fix ever misses an edge case.
        msg = (
            "errMsg: Invalid input '!': expected '.', '(', AND, OR, XOR, NOT, "
            "'=~', '=', '<>', '+', '-', '*', '/', '%', '^', IN, CONTAINS, "
            "STARTS WITH, ENDS WITH, '<=', '>=', '<', '>', IS NULL, IS NOT NULL, "
            "'[', '{', a label or THEN line: 1, column: 53"
        )
        err = classify_execution_error(msg)
        assert err.reason == 'not_equals_operator'
        assert '<>' in err.suggestion
        assert '!=' in err.suggestion

    def test_classify_sql_window_function(self):
        # FalkorDB's parser rejects `<agg>(...) OVER (PARTITION BY ...)` with
        # `Invalid input 'V'` (the V from OVER). openCypher does not have SQL
        # window functions; the equivalent is a chained WITH that aggregates
        # by group, then re-joins.
        msg = (
            "errMsg: Invalid input 'V': expected OR, ORDER BY or OPTIONAL MATCH "
            "line: 32, column: 31, offset: 2604 "
            "errCtx: sum(weighted_incidents) OVER (PARTITION BY operator) "
            "AS total_weighted errCtxOffset: 30"
        )
        err = classify_execution_error(msg)
        assert err.reason == 'sql_window_function'
        assert 'OVER' in err.suggestion
        assert 'WITH' in err.suggestion

    def test_classify_not_in_list_form(self):
        # FalkorDB does NOT support `x NOT IN [list]` even with parens around
        # the list — only `NOT (x IN [list])`. The parser fails on the comma
        # inside the list because it has already consumed the `[` as something
        # else. Tell the LLM to wrap with explicit parens and prefix NOT.
        msg = (
            "errMsg: Invalid input ',': expected '.', AND, OR, XOR, NOT, '=~', "
            "'=', '<>', '+', '-', '*', '/', '%', '^', IN, CONTAINS, STARTS WITH, "
            "ENDS WITH, '<=', '>=', '<', '>', IS NULL, IS NOT NULL, '[', '{', "
            "a label, ']' or '..' line: 1, column: 32, offset: 31 "
            "errCtx: MATCH (n) WHERE n.x NOT IN ['a', 'b'] RETURN n LIMIT 1 "
            "errCtxOffset: 31"
        )
        err = classify_execution_error(msg)
        assert err.reason == 'not_in_list_form'
        assert 'NOT (' in err.suggestion
        assert 'IN [' in err.suggestion

    def test_classify_variable_not_in_scope_full_envelope(self):
        # The matcher is anchored on the `'<var>' not defined` substring, so
        # it should still classify when wrapped in the full FalkorDB error
        # envelope (errMsg / line / column trim).
        msg = "errMsg: 'foo_bar' not defined line: 12, column: 8"
        err = classify_execution_error(msg)
        assert err.reason == 'variable_not_in_scope'

    def test_classify_first_match_wins(self):
        # Construct a message that genuinely matches BOTH patterns — bare-var
        # `Invalid input '-': expected '='` AND where-needs-with `expected WITH
        # ... errCtx: WHERE`.  The first pattern listed (bare_variable_in_pattern)
        # must win.  Guards against a future re-ordering of the table.
        msg = (
            "errMsg: Invalid input '-': expected '=' "
            "expected WITH line: 5, column: 2 errCtx: WHERE x"
        )
        err = classify_execution_error(msg)
        assert err.reason == 'bare_variable_in_pattern'


class TestNonCodeSpanPreservation:
    """Pipeline operations must only inspect / modify code spans.

    String literals and comments are user-authored content that must pass
    through unchanged.  These tests lock in the categorical behavior across
    Stage 1 (LLM fixups), Stage 2a (reject track), Stage 2b (auto-fix
    track), and Stage 3 (security whitelist).
    """

    # ---- Stage 2b auto-fix: each fix must not modify comments/strings ----

    def test_date_wrapper_in_line_comment_preserved(self):
        query = "// use date('2024-01-01') for comparisons\nMATCH (n) RETURN n"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert "date('2024-01-01')" in fixed
        assert fixes == []

    def test_date_wrapper_in_block_comment_preserved(self):
        query = "/* date('2024-01-01') is the syntax */\nMATCH (n) RETURN n"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert "date('2024-01-01')" in fixed
        assert fixes == []

    def test_lower_in_comment_preserved(self):
        query = "// don't use lower(), use toLower()\nMATCH (n) RETURN n"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'lower()' in fixed
        assert fixes == []

    def test_upper_in_comment_preserved(self):
        query = "// upper() is unsupported\nMATCH (n) RETURN n"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'upper()' in fixed
        assert fixes == []

    def test_bare_var_in_comment_preserved(self):
        query = "// Example: MATCH foo-[:R]->(bar) RETURN foo\nMATCH (p) RETURN p"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'MATCH foo-[:R]->(bar)' in fixed
        assert fixes == []

    def test_non_ascii_in_comment_preserved(self):
        query = "// año actual 2026\nMATCH (p) RETURN p.name AS nombre"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert 'año actual' in fixed
        assert fixes == []

    def test_non_ascii_in_string_preserved(self):
        query = "MATCH (n) WHERE n.name = 'AÑO 2026' RETURN n.name AS nombre"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert "'AÑO 2026'" in fixed
        assert fixes == []

    def test_non_ascii_identifier_still_fixed_with_comment(self):
        # Comment with accents + identifier with accents — identifier gets
        # fixed, comment preserved.
        query = "// año actual 2026\nWITH val AS año_nacimiento RETURN año_nacimiento"
        fixed, fixes = _fix_falkordb_dialect(query)
        # Comment preserved
        assert '// año actual 2026' in fixed
        # Identifier transliterated
        assert 'ano_nacimiento' in fixed
        # Only one fix reported
        assert any('non-ascii' in f.lower() for f in fixes)

    # ---- Stage 2a reject: mentions inside comments/strings don't trigger ----

    def test_apoc_in_comment_not_rejected(self):
        query = "// apoc.path.expand is not supported\nMATCH (n) RETURN n"
        assert _check_falkordb_dialect(query) is None

    def test_apoc_in_string_not_rejected(self):
        query = "MATCH (n) WHERE n.doc = 'apoc.path.expand(x)' RETURN n"
        assert _check_falkordb_dialect(query) is None

    def test_exists_brace_in_comment_not_rejected(self):
        query = "// EXISTS { ... } subqueries are unsupported\nMATCH (n) RETURN n"
        assert _check_falkordb_dialect(query) is None

    def test_unwind_where_in_comment_not_rejected(self):
        query = "// UNWIND list AS x WHERE x > 0 fails\nMATCH (n) RETURN n"
        assert _check_falkordb_dialect(query) is None

    # ---- Stage 3 security whitelist: keywords in non-code spans ignored ----

    def test_create_in_comment_not_rejected(self):
        assert _check_whitelist("// CREATE a diagnostic\nMATCH (p) RETURN count(p)") is None

    def test_delete_in_string_not_rejected(self):
        assert _check_whitelist("MATCH (p) WHERE p.note = 'DELETE this' RETURN p") is None

    def test_set_merge_in_block_comment_not_rejected(self):
        assert _check_whitelist("/* SET and MERGE docs */\nMATCH (p) RETURN p") is None

    def test_call_apoc_in_comment_not_rejected(self):
        assert _check_whitelist("// Don't use CALL apoc.foo()\nCALL db.labels()") is None

    def test_real_create_still_rejected(self):
        err = _check_whitelist('CREATE (n:Test)')
        assert err is not None
        assert err.reason == 'write_operation'

    def test_real_call_apoc_still_rejected(self):
        err = _check_whitelist("MATCH (n) CALL apoc.path.expand(n, 'R>') YIELD p RETURN p")
        assert err is not None

    # ---- End-to-end pipeline: Spanish comments must not pollute auto_fixes ----

    def test_spanish_comments_no_false_auto_fix(self):
        # Realistic LLM output from the 2026-04-16 Round-1 incident: Spanish
        # comments with accented characters should pass through the pipeline
        # with an empty auto_fixes list (besides LIMIT injection).
        query = (
            "// Análisis completo de KHADIJA DAOUD\n"
            "// Calcular edad aproximada (asumiendo año actual 2026)\n"
            "MATCH (p:Persona {name: 'KHADIJA DAOUD'}) RETURN p"
        )
        result = validate_and_sanitize(query, _FLAVOUR)
        assert isinstance(result, SanitizedQuery)
        # Only the LIMIT injection should have fired
        real_fixes = [f for f in result.auto_fixes if 'LIMIT' not in f]
        assert real_fixes == []
        # Content preserved
        assert 'Análisis' in result.query
        assert 'año actual' in result.query

    # ---- Edge case: `//` inside a string literal is NOT a comment ----

    def test_url_in_string_not_treated_as_comment(self):
        # String literal containing `//` (URL-like) — the string mask wins,
        # the comment regex must not treat the quoted `//` as a comment start.
        query = "MATCH (n) WHERE n.url = 'http://example.com/path' RETURN n"
        fixed, fixes = _fix_falkordb_dialect(query)
        assert fixed == query
        assert fixes == []


class TestRunCypherToolDescription:
    def test_docstring_is_backend_neutral_and_points_to_dialect_reference(self):
        # ADR-019 R6: the graph_query docstring must NOT hardcode a single backend's dialect —
        # per-backend dialect lives in the (flavour-driven) tool description + get_schema's
        # dialect_reference. The docstring points there instead.
        import re
        from pathlib import Path
        src_path = Path(__file__).parent.parent / 'src' / 'graphiti_mcp_server.py'
        source = src_path.read_text()
        m = re.search(
            r'async def graph_query\([^)]*\)[^:]*:\s*"""(.*?)"""',
            source,
            re.DOTALL,
        )
        assert m, 'graph_query docstring not found'
        doc = m.group(1)
        assert 'dialect' in doc.lower()
        assert 'dialect_reference' in doc            # points to the announced dialect
        # No hardcoded FalkorDB-only gotchas remain in the docstring.
        assert 'FalkorDB Cypher dialect' not in doc
