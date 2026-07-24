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
