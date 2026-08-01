"""Unit tests for AgeFlavour (offline — no DB).

AGE-unsupported constructs (a variable named `id`, relationship-type disjunction [:A|B|C]) are
REJECT-with-hint via check_dialect — never auto-rewritten (a reliable structural rewrite of
arbitrary openCypher is not feasible with pattern matching, and a wrong rewrite silently returns
wrong data; the agent writes the correct form in one step from the hint). The only AGE auto-fix
is the shared pipeline's LIMIT injection.
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


# --- auto_fix: no AGE-specific rewrite (inherited no-op; LIMIT is the shared pipeline's job) ---

def test_auto_fix_is_noop():
    q, fixes = AgeFlavour().auto_fix("MATCH (a)-[r:DETIENE]->(b) RETURN b")
    assert q == "MATCH (a)-[r:DETIENE]->(b) RETURN b"
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
