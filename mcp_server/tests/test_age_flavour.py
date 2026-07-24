"""Unit tests for AgeFlavour (offline — no DB)."""
import pytest

from flavours.age import AgeFlavour
from utils.cypher import CypherError


def test_identity():
    f = AgeFlavour()
    assert f.dialect_id == "age-opencypher"
    assert "Apache AGE" in f.dialect_reference
    assert "attributes" in f.dialect_reference  # nested-map guidance present


def test_check_dialect_rejects_id_variable():
    err = AgeFlavour().check_dialect("MATCH (id) RETURN id")
    assert isinstance(err, CypherError)
    assert err.reason == "reserved_id_variable"
    assert "ident" in err.suggestion  # suggests a rename


def test_check_dialect_allows_normal_query():
    assert AgeFlavour().check_dialect("MATCH (n) RETURN n.name") is None


def test_check_dialect_allows_id_as_property():
    # n.id (property access) is fine; only a bare `id` variable collides with graphid.
    assert AgeFlavour().check_dialect("MATCH (n) WHERE n.id = '5' RETURN n") is None


def test_check_dialect_does_not_reject_reltype_disjunction():
    # [:A|B|C] is AUTO-FIXED (see auto_fix), not rejected.
    assert AgeFlavour().check_dialect("MATCH (a)-[:A|B|C]->(b) RETURN b") is None


def test_auto_fix_rewrites_reltype_disjunction():
    q, fixes = AgeFlavour().auto_fix("MATCH (a)-[:DETIENE|INVESTIGA]->(b) RETURN b")
    assert "|" not in q
    assert "type(r)" in q
    assert "'DETIENE'" in q and "'INVESTIGA'" in q
    assert any("disjunction" in f.lower() or "type(r)" in f for f in fixes)


def test_auto_fix_reltype_disjunction_with_existing_var():
    q, _ = AgeFlavour().auto_fix("MATCH (a)-[r:A|B]->(b) WHERE b.x = 1 RETURN b")
    assert "|" not in q
    assert "type(r) IN ['A', 'B']" in q


def test_auto_fix_noop_when_clean():
    q, fixes = AgeFlavour().auto_fix("MATCH (a)-[r:DETIENE]->(b) RETURN b")
    assert q == "MATCH (a)-[r:DETIENE]->(b) RETURN b"
    assert fixes == []


def test_classify_execution_error_graphid():
    err = AgeFlavour().classify_execution_error("column notation .id applied to type graphid")
    assert isinstance(err, CypherError)
    assert "id" in err.suggestion.lower()


def test_classify_execution_error_pipe_syntax():
    err = AgeFlavour().classify_execution_error('syntax error at or near "|"')
    assert "type(r) IN" in err.suggestion


# --- Regression tests for focused-review findings (C1/C2/C3/I1/I2) ---

def test_auto_fix_C1_attaches_filter_to_the_disjunction_clause_not_a_preceding_where():
    # C1: a preceding MATCH...WHERE must NOT capture the type filter; it attaches to the
    # clause that actually contains the disjunction (before the following RETURN).
    q, _ = AgeFlavour().auto_fix(
        "MATCH (x) WHERE x.name='Ana' MATCH (a)-[:DETIENE|INVESTIGA]->(b) RETURN a,b"
    )
    assert "|" not in q
    assert "WHERE x.name='Ana'" in q                 # the preceding WHERE is left intact
    assert "type(r) IN ['DETIENE', 'INVESTIGA']" in q
    # the type filter sits after the rewritten [r] edge and before RETURN
    assert q.index("type(r)") > q.index("[r]")
    assert q.index("type(r)") < q.index("RETURN")


def test_auto_fix_C2_parenthesizes_existing_or_where():
    # C2: AND-ing into an existing top-level OR must parenthesize it, or the filter is defeated.
    q, _ = AgeFlavour().auto_fix(
        "MATCH (a)-[:A|B]->(b) WHERE b.name='Ana' OR b.age>30 RETURN b"
    )
    assert "|" not in q
    assert "type(r) IN ['A', 'B'] AND (b.name='Ana' OR b.age>30)" in q


def test_auto_fix_C3_bails_on_multiple_disjunctions():
    # C3: >1 disjunction is ambiguous to rewrite safely — leave unchanged for the hint net.
    src = "MATCH (a)-[:A|B]->(b) MATCH (c)-[:C|D]->(d) RETURN b,d"
    q, fixes = AgeFlavour().auto_fix(src)
    assert q == src
    assert fixes == []


def test_auto_fix_C3_uses_collision_free_variable():
    # C3: introduced rel var must not clobber an existing variable named `r`.
    q, _ = AgeFlavour().auto_fix("MATCH (r:Persona)-[:A|B]->(b) RETURN r,b")
    assert "|" not in q
    assert "(r:Persona)" in q          # the pre-existing node var `r` is untouched
    assert "type(rt)" in q             # a fresh, non-colliding rel var was introduced


def test_check_dialect_I1_ignores_id_inside_string_literal():
    # I1: `(id: 5)` inside a string literal must not trigger the id-variable reject.
    assert AgeFlavour().check_dialect(
        "MATCH (n) WHERE n.text CONTAINS '(id: 5)' RETURN n.name"
    ) is None


def test_check_dialect_I2_allows_id_function_call():
    # I2: AGE's own id() function is legitimate — do not reject it.
    assert AgeFlavour().check_dialect("MATCH (n) RETURN id(n)") is None
    assert AgeFlavour().check_dialect("MATCH (n) RETURN count(id(n))") is None


def test_check_dialect_rejects_id_alias():
    # A column aliased to `id` is still a variable named id — reject.
    err = AgeFlavour().check_dialect("MATCH (n) RETURN n.name AS id")
    assert err is not None and err.reason == "reserved_id_variable"


# --- Regression tests for RE-REVIEW findings (string-masking + map keys) ---

def test_auto_fix_rereview_keyword_inside_string_literal_not_split():
    # A clause keyword ('with') inside a string literal must not split the WHERE condition.
    q, _ = AgeFlavour().auto_fix(
        "MATCH (a)-[:A|B]->(b) WHERE b.description CONTAINS 'person with a knife' RETURN b"
    )
    assert "|" not in q.replace("'person with a knife'", "")  # the disjunction | is gone
    assert "type(r) IN ['A', 'B'] AND (b.description CONTAINS 'person with a knife')" in q
    assert q.count("(") == q.count(")")   # parens balanced (the string keyword didn't corrupt)
    assert "'person with a knife'" in q   # the string literal is intact


def test_auto_fix_rereview_bracket_text_in_string_not_counted_as_disjunction():
    # A bracket-shaped decoy inside a string must not make a single disjunction look like >1.
    q, fixes = AgeFlavour().auto_fix(
        "MATCH (a)-[:A|B]->(b) WHERE b.raw CONTAINS '[:X|Y]' RETURN b"
    )
    assert fixes != []                       # the REAL disjunction was rewritten
    assert "type(r) IN ['A', 'B']" in q
    assert "'[:X|Y]'" in q                    # the string literal is preserved verbatim


def test_check_dialect_rereview_allows_id_map_key():
    # A map literal key named `id` ({id: 5}) is not a variable — must be allowed.
    assert AgeFlavour().check_dialect("MATCH (n {id: 5}) RETURN n") is None
