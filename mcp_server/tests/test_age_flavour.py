"""Unit tests for AgeFlavour (offline — no DB).

AGE-unsupported constructs (a variable named `id`, relationship-type disjunction [:A|B|C]) are
REJECT-with-hint via check_dialect — never auto-rewritten (a reliable structural rewrite of
arbitrary openCypher is not feasible with pattern matching, and a wrong rewrite silently returns
wrong data; the agent writes the correct form in one step from the hint). auto_fix applies three
bench-observed rewrites (`!=` -> `<>`, strip PROFILE/EXPLAIN, rename the reserved `count` alias);
LIMIT injection remains the shared pipeline's job.
"""
import pytest

from flavours.age import AgeFlavour
from utils.cypher import CypherError


def test_identity():
    f = AgeFlavour()
    assert f.dialect_id == "age-opencypher"
    assert "Apache AGE" in f.dialect_reference
    assert "attributes" in f.dialect_reference  # nested-map guidance present


# --- id-variable reject (precise: pattern var / alias only) ---

def test_check_dialect_rejects_id_variable():
    err = AgeFlavour().check_dialect("MATCH (id) RETURN id")
    assert isinstance(err, CypherError)
    assert err.reason == "reserved_id_variable"
    assert "ident" in err.suggestion  # suggests a rename


def test_check_dialect_rejects_id_alias():
    err = AgeFlavour().check_dialect("MATCH (n) RETURN n.name AS id")
    assert err is not None and err.reason == "reserved_id_variable"


def test_check_dialect_allows_normal_query():
    assert AgeFlavour().check_dialect("MATCH (n) RETURN n.name") is None


def test_check_dialect_allows_id_as_property():
    # n.id (property access) is fine; only a bare `id` variable collides with graphid.
    assert AgeFlavour().check_dialect("MATCH (n) WHERE n.id = '5' RETURN n") is None


def test_check_dialect_I1_ignores_id_inside_string_literal():
    # `(id: 5)` inside a string literal must not trigger the id-variable reject.
    assert AgeFlavour().check_dialect(
        "MATCH (n) WHERE n.text CONTAINS '(id: 5)' RETURN n.name"
    ) is None


def test_check_dialect_I2_allows_id_function_call():
    # AGE's own id() function is legitimate — do not reject it.
    assert AgeFlavour().check_dialect("MATCH (n) RETURN id(n)") is None
    assert AgeFlavour().check_dialect("MATCH (n) RETURN count(id(n))") is None


def test_check_dialect_allows_id_map_key():
    # A map literal key named `id` ({id: 5}) is not a variable — must be allowed.
    assert AgeFlavour().check_dialect("MATCH (n {id: 5}) RETURN n") is None


# --- relationship-type disjunction [:A|B|C] reject-with-hint (NOT auto-rewrite) ---

def test_check_dialect_rejects_reltype_disjunction():
    err = AgeFlavour().check_dialect("MATCH (a)-[:DETIENE|INVESTIGA]->(b) RETURN b")
    assert isinstance(err, CypherError)
    assert err.reason == "reltype_disjunction_unsupported"
    assert "type(r) IN" in err.suggestion   # the hint teaches the correct form


def test_check_dialect_rejects_reltype_disjunction_with_var():
    err = AgeFlavour().check_dialect("MATCH (a)-[r:A|B]->(b) WHERE b.x = 1 RETURN b")
    assert err is not None and err.reason == "reltype_disjunction_unsupported"


def test_check_dialect_allows_single_reltype():
    assert AgeFlavour().check_dialect("MATCH (a)-[r:DETIENE]->(b) RETURN b") is None


def test_check_dialect_disjunction_inside_string_not_rejected():
    # A bracket/pipe decoy inside a string literal must not trip the disjunction reject.
    assert AgeFlavour().check_dialect(
        "MATCH (a)-[r:KNOWS]->(b) WHERE b.raw CONTAINS '[:X|Y]' RETURN b"
    ) is None


def test_previously_corrupting_query_is_now_cleanly_rejected():
    # A property named like a clause keyword (b.limit) used to CORRUPT the old regex rewrite.
    # With reject-with-hint there is no rewrite: the disjunction is simply rejected, cleanly.
    err = AgeFlavour().check_dialect("MATCH (a)-[:A|B]->(b) WHERE b.limit > 5 RETURN b")
    assert isinstance(err, CypherError) and err.reason == "reltype_disjunction_unsupported"


def test_no_disjunction_with_keyword_property_is_allowed():
    # Without a disjunction, a keyword-shaped property (b.limit) is untouched and allowed —
    # confirming the old clause-scan corruption is gone entirely.
    assert AgeFlavour().check_dialect("MATCH (a)-[r:KNOWS]->(b) WHERE b.limit > 5 RETURN b") is None


# --- auto_fix: bench-observed AGE rewrites (LIMIT is still the shared pipeline's job) ---

def test_auto_fix_leaves_a_clean_query_untouched():
    # Replaces the former test_auto_fix_is_noop: AGE now HAS auto-fixes, so the invariant
    # under test changed from "never rewrites" to "never rewrites a query that is already
    # AGE-valid". Deliberate behaviour change, not a relaxed assertion.
    q, fixes = AgeFlavour().auto_fix("MATCH (a)-[r:DETIENE]->(b) RETURN b.name AS name")
    assert q == "MATCH (a)-[r:DETIENE]->(b) RETURN b.name AS name"
    assert fixes == []


# --- execution-error classification (post-hoc net) ---

def test_classify_execution_error_graphid():
    err = AgeFlavour().classify_execution_error("column notation .id applied to type graphid")
    assert isinstance(err, CypherError)
    assert "id" in err.suggestion.lower()


def test_classify_execution_error_pipe_syntax():
    err = AgeFlavour().classify_execution_error('syntax error at or near "|"')
    assert "type(r) IN" in err.suggestion


# --- AgeFlavour.attribute_keys reads the nested `attributes` agtype map (offline) ---

@pytest.mark.asyncio
async def test_age_attribute_keys_unions_nested_map_keys():
    class _StubDriver:
        async def execute_query(self, query, *a, **k):
            assert "keys(n.attributes)" in query   # queries the nested map, not top-level keys
            return ([{"ks": ["documento", "nombre"]}, {"ks": ["documento", "telefono"]}], None, None)

    keys = await AgeFlavour().attribute_keys(_StubDriver(), "Persona")
    assert keys == ["documento", "nombre", "telefono"]   # unioned across sample + sorted


@pytest.mark.asyncio
async def test_age_attribute_keys_swallows_non_map_label():
    class _BoomDriver:
        async def execute_query(self, query, *a, **k):
            raise RuntimeError("attributes is not a map on this label")

    assert await AgeFlavour().attribute_keys(_BoomDriver(), "Weird") == []


# --- shared span masking (utils.cypher._strip_non_code_spans), not a bespoke copy ---

def test_check_dialect_ignores_constructs_inside_a_line_comment():
    q = "MATCH (a)-[r:KNOWS]->(b) // avoid MATCH (id) and [:A|B] here\nRETURN b"
    assert AgeFlavour().check_dialect(q) is None


def test_check_dialect_ignores_constructs_inside_a_block_comment():
    q = "MATCH (a)-[r:KNOWS]->(b) /* was: MATCH (id)-[:A|B]->(x) */ RETURN b"
    assert AgeFlavour().check_dialect(q) is None


def test_module_level_check_age_dialect_is_the_implementation():
    from flavours.age import check_age_dialect

    err = check_age_dialect("MATCH (a)-[:A|B]->(b) RETURN b")
    assert err is not None and err.reason == "reltype_disjunction_unsupported"
    assert check_age_dialect("MATCH (n) RETURN n.name") is None


# --- execution-error table + synthesized errCtx (Postgres echoes no query context) ---

def test_classify_migrated_graphid_pattern():
    err = AgeFlavour().classify_execution_error(
        "column notation .id applied to type graphid",
        query="MATCH (id) RETURN id.name LIMIT 5",
    )
    assert err.reason == "reserved_id_variable"
    assert "ident" in err.suggestion
    assert err.doc_hint != ""


def test_classify_migrated_reltype_disjunction_pattern():
    err = AgeFlavour().classify_execution_error('syntax error at or near "|"')
    assert err.reason == "reltype_disjunction_unsupported"
    assert "type(r) IN" in err.suggestion
    assert err.doc_hint != ""


def test_classify_synthesizes_errctx_around_the_failing_token():
    query = (
        "MATCH (a)-[r:DETIENE|INVESTIGA]->(b) "
        "WHERE b.name = 'OMAR MOHAMED' RETURN b.name LIMIT 25"
    )
    err = AgeFlavour().classify_execution_error('syntax error at or near "|"', query=query)
    assert "errCtx:" in err.explanation
    assert "DETIENE|INVESTIGA" in err.explanation


def test_classify_errctx_falls_back_to_the_query_head_when_no_token_named():
    query = "MATCH (n) WHERE n.name = 'x' RETURN n.name LIMIT 25"
    err = AgeFlavour().classify_execution_error("something went sideways", query=query)
    assert "MATCH (n) WHERE n.name = 'x'" in err.explanation


def test_classify_without_a_query_emits_no_errctx():
    err = AgeFlavour().classify_execution_error('syntax error at or near "|"')
    assert "errCtx:" not in err.explanation


def test_classify_unknown_error_returns_the_generic_envelope():
    err = AgeFlavour().classify_execution_error("totally novel failure")
    assert err.stage == "execution"
    assert err.reason == "query_failed"
    assert "get_schema" in err.suggestion


# --- unbound $parameter: 8 of the 11 AGE bench failures ---

_UNBOUND_PARAM_MSG = "parameters argument is missing from cypher() function call"


def test_classify_unbound_parameter():
    err = AgeFlavour().classify_execution_error(
        _UNBOUND_PARAM_MSG, query="MATCH (n) WHERE n.name = $nm RETURN n.name LIMIT 25"
    )
    assert err.reason == "unbound_parameter"
    assert "$nm" in err.suggestion
    assert "inline" in err.suggestion.lower()
    assert err.doc_hint != ""


def test_classify_unbound_parameter_names_every_parameter_once():
    query = "MATCH (n) WHERE n.a = $one AND n.b = $two AND n.c = $one RETURN n LIMIT 25"
    err = AgeFlavour().classify_execution_error(_UNBOUND_PARAM_MSG, query=query)
    assert "$one" in err.suggestion and "$two" in err.suggestion
    assert err.suggestion.count("$one") == 1


def test_classify_unbound_parameter_ignores_dollars_inside_string_literals():
    query = "MATCH (n) WHERE n.note = 'costs $50 total' AND n.x = $real RETURN n LIMIT 25"
    err = AgeFlavour().classify_execution_error(_UNBOUND_PARAM_MSG, query=query)
    assert "$real" in err.suggestion
    assert "$50" not in err.suggestion


def test_classify_unbound_parameter_without_a_query_still_classifies():
    err = AgeFlavour().classify_execution_error(_UNBOUND_PARAM_MSG)
    assert err.reason == "unbound_parameter"
    assert "inline" in err.suggestion.lower()


# --- bare label test outside a MATCH pattern (live: `at or near ":"` / `at or near "OR"`) ---

def test_classify_boolean_label_test_colon_token():
    err = AgeFlavour().classify_execution_error(
        'syntax error at or near ":"',
        query="MATCH (n) WHERE n:Persona OR n:Ubicacion RETURN n.name LIMIT 25",
    )
    assert err.reason == "boolean_label_test"
    assert "(n:Persona)" in err.suggestion or "parenthes" in err.suggestion.lower()
    assert "label(n)" in err.suggestion
    assert err.doc_hint != ""


def test_classify_boolean_label_test_or_token_from_in_pattern_disjunction():
    err = AgeFlavour().classify_execution_error(
        'syntax error at or near "OR"',
        query="MATCH (n:Persona OR n:Ubicacion) RETURN n.name LIMIT 25",
    )
    assert err.reason == "boolean_label_test"


def test_classify_boolean_label_test_covers_not_and_case_forms():
    for query in (
        "MATCH (n) WHERE NOT n:Persona RETURN count(n) AS c",
        "MATCH (n) RETURN CASE WHEN n:Persona THEN 1 ELSE 0 END AS c LIMIT 25",
    ):
        err = AgeFlavour().classify_execution_error('syntax error at or near ":"', query=query)
        assert err.reason == "boolean_label_test", query


def test_classify_boolean_label_test_not_tripped_by_a_plain_pattern_label():
    # `MATCH (n:Persona)` is valid AGE; a colon error on such a query is something else.
    err = AgeFlavour().classify_execution_error(
        'syntax error at or near ":"', query="MATCH (n:Persona) RETURN n.name LIMIT 25"
    )
    assert err.reason == "query_failed"


def test_classify_boolean_label_test_not_tripped_by_a_map_key():
    err = AgeFlavour().classify_execution_error(
        'syntax error at or near ":"', query="MATCH (n {id: 5}) RETURN n LIMIT 25"
    )
    assert err.reason == "query_failed"


# --- reserved-word alias (live sweep: count/exists/all/any/none/single/distinct/end/
#     contains/starts/ends/null/true/false/coalesce all fail as a bare identifier) ---

_RESERVED_ALIAS_QUERY = (
    "MATCH (n) RETURN label(n) AS type, count(n) AS count ORDER BY count DESC LIMIT 25"
)


def test_classify_reserved_alias_from_desc_token():
    err = AgeFlavour().classify_execution_error(
        'syntax error at or near "DESC"', query=_RESERVED_ALIAS_QUERY
    )
    assert err.reason == "reserved_alias"
    assert "count" in err.suggestion
    assert "cnt" in err.suggestion
    assert err.doc_hint != ""


def test_classify_reserved_alias_from_asc_limit_and_order_tokens():
    for token in ("ASC", "LIMIT", "ORDER"):
        err = AgeFlavour().classify_execution_error(
            f'syntax error at or near "{token}"', query=_RESERVED_ALIAS_QUERY
        )
        assert err.reason == "reserved_alias", token


def test_classify_reserved_alias_covers_exists():
    err = AgeFlavour().classify_execution_error(
        'syntax error at or near "DESC"',
        query="MATCH (n) RETURN n.name AS nm, count(n) AS exists ORDER BY exists DESC LIMIT 5",
    )
    assert err.reason == "reserved_alias"
    assert "exists" in err.suggestion


def test_classify_reserved_alias_requires_a_reserved_alias_in_the_query():
    err = AgeFlavour().classify_execution_error(
        'syntax error at or near "DESC"',
        query="MATCH (n) RETURN label(n) AS type, count(n) AS cnt ORDER BY cnt DESC LIMIT 25",
    )
    assert err.reason == "query_failed"


def test_classify_reserved_alias_ignores_a_reserved_word_inside_a_string():
    err = AgeFlavour().classify_execution_error(
        'syntax error at or near "DESC"',
        query="MATCH (n) WHERE n.note = 'AS count' RETURN n.name AS nm ORDER BY nm DESC LIMIT 5",
    )
    assert err.reason == "query_failed"


# --- projection / column-derivation failures (AGEDriver._columns_from_return) ---

def test_classify_duplicate_return_column():
    err = AgeFlavour().classify_execution_error(
        'column name "parte_uuid" specified more than once',
        query="MATCH (p)-[]->(r) RETURN p.uuid AS parte_uuid, r.uuid AS parte_uuid LIMIT 25",
    )
    assert err.reason == "duplicate_return_column"
    assert "parte_uuid" in err.explanation
    assert "distinct" in err.suggestion.lower() or "unique" in err.suggestion.lower()
    assert err.doc_hint != ""


def test_classify_projection_column_mismatch():
    err = AgeFlavour().classify_execution_error(
        "return row and column definition list do not match",
        query="MATCH (n) RETURN n.name, n.uuid LIMIT 25",
    )
    assert err.reason == "projection_column_mismatch"
    assert "AS" in err.suggestion
    assert err.doc_hint != ""


def test_classify_mixed_case_alias():
    # str(KeyError('TipoDelito')) is exactly "'TipoDelito'" — the whole message.
    err = AgeFlavour().classify_execution_error(
        "'TipoDelito'",
        query="MATCH (n) RETURN label(n) AS TipoDelito, count(n) AS cnt LIMIT 25",
    )
    assert err.reason == "mixed_case_alias"
    assert "TipoDelito" in err.suggestion
    assert "lowercase" in err.suggestion.lower() or "snake_case" in err.suggestion


def test_classify_mixed_case_alias_requires_an_uppercase_alias():
    err = AgeFlavour().classify_execution_error(
        "'tipodelito'", query="MATCH (n) RETURN label(n) AS tipodelito LIMIT 25"
    )
    assert err.reason == "query_failed"


def test_classify_mixed_case_alias_names_every_offender():
    err = AgeFlavour().classify_execution_error(
        "'TipoDelito'",
        query="MATCH (n) RETURN label(n) AS TipoDelito, n.name AS nombreCompleto LIMIT 25",
    )
    assert "TipoDelito" in err.suggestion
    assert "nombreCompleto" in err.suggestion


# --- internal / operator errors ---

def test_classify_internal_resource_owner_error_is_retryable():
    # Observed once in the v0.31.0 bench (report line 123); not reproducible on demand.
    msg = "tupdesc reference 0x7f8b2c0a1234 is not owned by resource owner Portal"
    err = AgeFlavour().classify_execution_error(msg, query="MATCH (n) RETURN n LIMIT 25")
    assert err.reason == "internal_resource_owner"
    assert "retry" in err.suggestion.lower()
    assert err.doc_hint != ""


def test_classify_not_equals_operator():
    err = AgeFlavour().classify_execution_error(
        "operator does not exist: agtype != agtype",
        query="MATCH (n) WHERE n.name != 'zzz' RETURN n.name LIMIT 25",
    )
    assert err.reason == "not_equals_operator"
    assert "<>" in err.suggestion


def test_pattern_table_order_puts_specific_fingerprints_before_generic_ones():
    from flavours.age import _AGE_EXECUTION_ERROR_PATTERNS

    names = [p.name for p in _AGE_EXECUTION_ERROR_PATTERNS]
    # The two `syntax error at or near ...` patterns must be last, most-specific first.
    assert names[-1] == "reserved_alias"
    assert names[-2] == "boolean_label_test"
    assert names.index("reltype_disjunction_unsupported") < names.index("boolean_label_test")
    assert len(names) == len(set(names)), "duplicate pattern names"


# --- stage 2a rejects: $parameter and $$ ---

def test_check_dialect_rejects_dollar_parameter():
    err = AgeFlavour().check_dialect("MATCH (n) WHERE n.name = $name RETURN n.name")
    assert isinstance(err, CypherError)
    assert err.stage == "age_dialect"
    assert err.reason == "unbound_parameter"
    assert "$name" in err.suggestion
    assert "inline" in err.suggestion.lower()
    assert err.doc_hint != ""


def test_check_dialect_reject_names_every_parameter():
    err = AgeFlavour().check_dialect(
        "MATCH (n) WHERE n.a = $one AND n.b = $two RETURN n"
    )
    assert "$one" in err.suggestion and "$two" in err.suggestion


def test_check_dialect_rejects_double_dollar():
    err = AgeFlavour().check_dialect("MATCH (n) WHERE n.x = $$ RETURN n")
    assert isinstance(err, CypherError)
    assert err.reason == "dollar_quote_unsupported"
    assert "inline" in err.suggestion.lower()
    assert err.doc_hint != ""


def test_check_dialect_dollar_quote_is_checked_before_parameters():
    # A query carrying both must report the security-relevant one.
    err = AgeFlavour().check_dialect("MATCH (n) WHERE n.a = $one AND n.b = $$ RETURN n")
    assert err.reason == "dollar_quote_unsupported"


def test_check_dialect_allows_dollar_inside_a_string_literal():
    assert AgeFlavour().check_dialect(
        "MATCH (n) WHERE n.note = 'costs $50 and $$ too' RETURN n.name"
    ) is None


def test_check_dialect_allows_dollar_inside_a_comment():
    assert AgeFlavour().check_dialect(
        "MATCH (n) // was: WHERE n.x = $foo and $$\nRETURN n.name"
    ) is None


def test_auto_fix_rewrites_not_equals():
    # Live: `!=` -> `operator does not exist: agtype != agtype`; `<>` works.
    q, fixes = AgeFlavour().auto_fix("MATCH (n) WHERE n.name != 'zzz' RETURN n.name AS name")
    assert "<>" in q and "!=" not in q
    assert any("<>" in f for f in fixes)


def test_auto_fix_preserves_not_equals_inside_a_string_literal():
    q, fixes = AgeFlavour().auto_fix("MATCH (n) WHERE n.note = 'a != b' RETURN n.name AS name")
    assert "'a != b'" in q
    assert fixes == []


def test_auto_fix_strips_a_profile_prefix():
    # Live: `PROFILE MATCH ...` -> `syntax error at or near "PROFILE"`.
    q, fixes = AgeFlavour().auto_fix("PROFILE MATCH (n) RETURN n.name AS name")
    assert q.startswith("MATCH")
    assert any("PROFILE" in f for f in fixes)


def test_auto_fix_strips_an_explain_prefix():
    # Live: EXPLAIN is ACCEPTED and escalates to a SQL-level EXPLAIN returning a QUERY PLAN
    # column, which then breaks the driver's column lookup. Stripping it is required.
    q, fixes = AgeFlavour().auto_fix("EXPLAIN MATCH (n) RETURN n.name AS name")
    assert q.startswith("MATCH")
    assert any("EXPLAIN" in f for f in fixes)


def test_auto_fix_renames_the_reserved_count_alias_and_its_references():
    q, fixes = AgeFlavour().auto_fix(
        "MATCH (n) RETURN label(n) AS type, count(n) AS count ORDER BY count DESC"
    )
    assert "AS count_" in q
    assert "ORDER BY count_ DESC" in q
    assert "count(n)" in q          # the aggregate call is untouched
    assert any("count" in f for f in fixes)


def test_auto_fix_does_not_touch_count_without_a_reserved_alias():
    # No `AS count` -> no rename; a property or map key named count stays as written.
    q, fixes = AgeFlavour().auto_fix(
        "MATCH (n) WHERE n.count > 5 RETURN n.count AS total ORDER BY total DESC"
    )
    assert q == "MATCH (n) WHERE n.count > 5 RETURN n.count AS total ORDER BY total DESC"
    assert fixes == []


# --- dialect reference: FalkorDB-style sections, not a prose blob ---

def test_dialect_reference_is_sectioned():
    ref = AgeFlavour().dialect_reference
    assert ref.startswith("## Cypher Quick Reference (Apache AGE)")
    for heading in (
        "### Labels",
        "### Parameters",
        "### Aliases",
        "### Label Tests",
        "### Relationship Types",
        "### Attributes & agtype",
        "### Projections",
        "### Known Limitations",
    ):
        assert heading in ref, heading


def test_dialect_reference_keeps_the_load_bearing_facts():
    ref = AgeFlavour().dialect_reference
    assert "Apache AGE" in ref
    assert "n.attributes" in ref
    assert "label(n)" in ref
    assert "n.labels" in ref
    assert "col0" in ref
    assert "count" in ref


def test_dialect_summary_names_parameters_and_label_tests():
    s = AgeFlavour().dialect_summary
    assert "Apache AGE openCypher" in s
    assert "n.attributes" in s
    assert "never name a variable `id`" in s
    assert "$param" in s
    assert "(n:Label)" in s


# --- F1: errCtx must centre on the REAL failing site, not the first substring match ---

def _errctx_of(err) -> str:
    """The synthesized `errCtx:` line of a CypherError explanation ('' when absent)."""
    for line in err.explanation.splitlines():
        if line.startswith("errCtx:"):
            return line[len("errCtx:"):].strip()
    return ""


def test_errctx_skips_a_token_embedded_in_a_longer_word():
    # Live shape: `DESC` also occurs inside `descripcion`, which is NOT the failing site.
    query = (
        "MATCH (n) RETURN n.attributes.descripcion AS descripcion, count(n) AS end "
        "ORDER BY end DESC LIMIT 5"
    )
    err = AgeFlavour().classify_execution_error('syntax error at or near "DESC"', query=query)
    ctx = _errctx_of(err)
    assert ctx, "expected a synthesized errCtx"
    assert "ORDER BY end DESC" in ctx, ctx


def test_errctx_ignores_a_token_occurring_only_inside_a_string_literal():
    query = (
        "MATCH (n) WHERE n.note = 'DESC ORDER' RETURN n.name AS nm, count(n) AS end "
        "ORDER BY end DESC LIMIT 5"
    )
    err = AgeFlavour().classify_execution_error('syntax error at or near "DESC"', query=query)
    ctx = _errctx_of(err)
    assert "ORDER BY end DESC" in ctx, ctx


def test_errctx_prefers_the_last_code_occurrence_of_the_token():
    query = "MATCH (n) RETURN n.a AS a, n.b AS b ORDER BY a DESC, b DESC LIMIT 5"
    err = AgeFlavour().classify_execution_error('syntax error at or near "DESC"', query=query)
    ctx = _errctx_of(err)
    assert ctx.endswith("b DESC LIMIT 5"), ctx


def test_errctx_handles_a_token_at_position_zero_without_a_leading_ellipsis():
    query = "MATCH (n) RETURN n.name AS name LIMIT 5"
    err = AgeFlavour().classify_execution_error('syntax error at or near "MATCH"', query=query)
    ctx = _errctx_of(err)
    assert ctx.startswith("MATCH (n)"), ctx
    assert not ctx.startswith("..."), ctx
