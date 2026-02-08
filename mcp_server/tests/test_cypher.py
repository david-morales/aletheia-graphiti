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
    _check_falkordb_dialect,
    _check_whitelist,
    _fix_falkordb_dialect,
    _fix_llm_syntax,
    _inject_safety,
    format_error,
    format_result,
    validate_and_sanitize,
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


class TestStage4SafetyInjection:
    def test_inject_limit_when_missing(self):
        query = 'MATCH (n) RETURN n'
        fixed, fixes = _inject_safety(query)
        assert f'LIMIT {DEFAULT_LIMIT + 1}' in fixed
        assert any('LIMIT' in f for f in fixes)

    def test_no_inject_when_limit_present(self):
        query = 'MATCH (n) RETURN n LIMIT 50'
        fixed, fixes = _inject_safety(query)
        assert fixed == query
        assert fixes == []

    def test_no_inject_when_limit_present_lowercase(self):
        query = 'MATCH (n) RETURN n limit 50'
        fixed, fixes = _inject_safety(query)
        assert fixed == query

    def test_limit_appended_after_order_by(self):
        query = 'MATCH (n) RETURN n ORDER BY n.name'
        fixed, fixes = _inject_safety(query)
        assert fixed.endswith(f'LIMIT {DEFAULT_LIMIT + 1}')
        assert 'ORDER BY' in fixed

    def test_limit_with_skip(self):
        query = 'MATCH (n) RETURN n SKIP 10'
        fixed, fixes = _inject_safety(query)
        assert f'LIMIT {DEFAULT_LIMIT + 1}' in fixed

    def test_call_query_no_limit(self):
        query = 'CALL db.labels()'
        fixed, fixes = _inject_safety(query)
        assert 'LIMIT' not in fixed


class TestPipelineOrchestration:
    def test_clean_query_passes(self):
        result = validate_and_sanitize('MATCH (n:Occurrence) RETURN n.name LIMIT 10')
        assert isinstance(result, SanitizedQuery)
        assert result.auto_fixes == []

    def test_fixable_query_returns_fixes(self):
        result = validate_and_sanitize(
            "MATCH (o:Occurrence) WHERE o.date_value > date('2024-06-01') RETURN o"
        )
        assert isinstance(result, SanitizedQuery)
        assert any('date' in f.lower() for f in result.auto_fixes)

    def test_write_query_rejected(self):
        result = validate_and_sanitize('CREATE (n:Test {name: "test"})')
        assert isinstance(result, CypherError)
        assert result.stage == 'security'

    def test_apoc_rejected_before_date_fix(self):
        result = validate_and_sanitize(
            "MATCH (n) WHERE n.date > date('2024-01-01') CALL apoc.path.expand(n, 'KNOWS>') YIELD path RETURN path"
        )
        assert isinstance(result, CypherError)
        assert result.reason == 'apoc_unsupported'

    def test_smart_quotes_fixed_then_dialect_fixed(self):
        result = validate_and_sanitize(
            'MATCH (o) WHERE o.name = \u201cBoeing\u201d AND o.date > date(\u20182024-01-01\u2019) RETURN o'
        )
        assert isinstance(result, SanitizedQuery)
        assert len(result.auto_fixes) >= 2

    def test_limit_injected_on_clean_query(self):
        result = validate_and_sanitize('MATCH (n) RETURN n')
        assert isinstance(result, SanitizedQuery)
        assert 'LIMIT' in result.query
        assert any('LIMIT' in f for f in result.auto_fixes)

    def test_existing_limit_preserved(self):
        result = validate_and_sanitize('MATCH (n) RETURN n LIMIT 50')
        assert isinstance(result, SanitizedQuery)
        assert 'LIMIT 50' in result.query
        assert not any('LIMIT' in f for f in result.auto_fixes)

    def test_code_block_plus_missing_return_plus_limit(self):
        result = validate_and_sanitize('```cypher\nMATCH (n:Occurrence)\n```')
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
        assert result['error']['stage'] == 'security'
        assert result['execution_ms'] == 0


# ---------------------------------------------------------------------------
# Tool description & server instruction tests
# ---------------------------------------------------------------------------


def _make_test_profile():
    entity_types = {
        'Occurrence': SimpleNamespace(
            label='Occurrence', count=10, description='Aviation incident',
            sample_names=['2024-0975-EU'],
        ),
        'Aircraft': SimpleNamespace(
            label='Aircraft', count=15, description='Aircraft entity',
            sample_names=['Boeing 737'],
        ),
        'Operator': SimpleNamespace(
            label='Operator', count=8, description='Airline operator',
            sample_names=['KLM'],
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
        from tool_descriptions import build_run_cypher_description

        desc = build_run_cypher_description(_make_test_profile())
        assert 'FalkorDB' in desc
        assert 'APOC' in desc

    def test_run_cypher_description_includes_examples(self):
        from tool_descriptions import build_run_cypher_description

        desc = build_run_cypher_description(_make_test_profile())
        assert 'Example' in desc or 'example' in desc
        assert 'MATCH' in desc

    def test_run_cypher_description_includes_chained_workflow(self):
        from tool_descriptions import build_run_cypher_description

        desc = build_run_cypher_description(_make_test_profile())
        assert 'search' in desc.lower()

    def test_run_cypher_description_includes_where_in_example(self):
        from tool_descriptions import build_run_cypher_description

        desc = build_run_cypher_description(_make_test_profile())
        assert 'IN [' in desc or 'found via search' in desc.lower()


class TestServerInstructions:
    def test_instructions_mention_get_schema(self):
        from tool_descriptions import build_instructions

        assert 'get_schema' in build_instructions(_make_test_profile())

    def test_instructions_mention_run_cypher(self):
        from tool_descriptions import build_instructions

        assert 'run_cypher' in build_instructions(_make_test_profile())

    def test_instructions_mention_chained_workflow(self):
        from tool_descriptions import build_instructions

        instructions = build_instructions(_make_test_profile())
        assert 'search' in instructions.lower()
        assert 'cypher' in instructions.lower() or 'run_cypher' in instructions.lower()

    def test_instructions_mention_semantic_vs_analytical(self):
        from tool_descriptions import build_instructions

        instructions = build_instructions(_make_test_profile())
        assert 'count' in instructions.lower() or 'aggregat' in instructions.lower()


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
