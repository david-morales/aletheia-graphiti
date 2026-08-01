"""Unit tests for the Flavour abstraction (base + factory)."""
import pytest

from flavours import build_flavour
from flavours.base import BaseFlavour, Flavour
from utils.cypher import CypherError


def test_base_flavour_satisfies_protocol():
    assert isinstance(BaseFlavour(), Flavour)


def test_base_flavour_identity():
    f = BaseFlavour()
    assert f.name == "opencypher"
    assert f.dialect_id == "opencypher"
    assert f.dialect_reference == ""


def test_base_check_dialect_never_rejects():
    # Generic openCypher: no backend-specific rejects.
    assert BaseFlavour().check_dialect("MATCH (n) RETURN n") is None
    assert BaseFlavour().check_dialect("MATCH (n)-[:A|B]->(m) RETURN m") is None


def test_base_auto_fix_is_noop():
    q, fixes = BaseFlavour().auto_fix("MATCH (n) RETURN n")
    assert q == "MATCH (n) RETURN n"
    assert fixes == []


def test_base_classify_execution_error_is_generic():
    err = BaseFlavour().classify_execution_error("boom happened")
    assert isinstance(err, CypherError)
    assert err.stage == "execution"
    assert "boom happened" in err.explanation


def test_build_flavour_neo4j_is_base():
    assert type(build_flavour("neo4j")) is BaseFlavour


def test_build_flavour_unknown_falls_back_to_base():
    assert isinstance(build_flavour("something-else"), BaseFlavour)


# --- FalkorDbFlavour (relocated dialect machinery) ---

def test_build_flavour_falkordb():
    from flavours.falkordb import FalkorDbFlavour
    assert type(build_flavour("falkordb")) is FalkorDbFlavour


def test_falkordb_flavour_identity():
    from flavours.falkordb import FalkorDbFlavour
    f = FalkorDbFlavour()
    assert f.dialect_id == "falkordb-cypher"
    assert "FalkorDB" in f.dialect_reference


def test_falkordb_check_dialect_rejects_apoc():
    from flavours.falkordb import FalkorDbFlavour
    err = FalkorDbFlavour().check_dialect("MATCH (n) CALL apoc.path.expand(n) RETURN n")
    assert err is not None and err.reason == "apoc_unsupported"


def test_falkordb_auto_fix_lowercases_helper():
    from flavours.falkordb import FalkorDbFlavour
    q, fixes = FalkorDbFlavour().auto_fix("MATCH (n) RETURN lower(n.name)")
    assert "toLower(" in q
    assert any("toLower" in f for f in fixes)


def test_falkordb_classify_execution_error_is_cypher_error():
    from flavours.falkordb import FalkorDbFlavour
    err = FalkorDbFlavour().classify_execution_error("Invalid input 'x'")
    assert isinstance(err, CypherError)


# --- validate_and_sanitize is flavour-driven ---

def test_validate_and_sanitize_uses_flavour_check_and_fix():
    from flavours.falkordb import FalkorDbFlavour
    from utils.cypher import validate_and_sanitize, SanitizedQuery, CypherError as _CE

    rejected = validate_and_sanitize("MATCH (n) CALL apoc.x() RETURN n", FalkorDbFlavour())
    assert isinstance(rejected, _CE)

    ok = validate_and_sanitize("MATCH (n) RETURN lower(n.name)", FalkorDbFlavour())
    assert isinstance(ok, SanitizedQuery)
    assert "toLower(" in ok.query
    assert "LIMIT" in ok.query.upper()


def test_validate_and_sanitize_base_flavour_no_dialect_fixes():
    from utils.cypher import validate_and_sanitize, SanitizedQuery

    ok = validate_and_sanitize("MATCH (n) RETURN n", BaseFlavour())
    assert isinstance(ok, SanitizedQuery)
    assert "LIMIT" in ok.query.upper()   # shared safety stage still runs


# --- GraphitiService selects the flavour from its provider ---

def test_graphiti_service_selects_flavour_from_provider():
    from config.schema import GraphitiConfig
    from graphiti_mcp_server import GraphitiService
    from flavours.falkordb import FalkorDbFlavour

    cfg = GraphitiConfig()
    cfg.database.provider = "falkordb"
    svc = GraphitiService(config=cfg)   # light ctor: sets self.config + self.flavour, no I/O
    assert isinstance(svc.flavour, FalkorDbFlavour)


def test_build_flavour_age():
    from flavours.age import AgeFlavour
    assert type(build_flavour("age")) is AgeFlavour


# --- BaseFlavour.execute_graph_query normalizes driver return shapes ---

@pytest.mark.asyncio
async def test_base_execute_handles_neo4j_eager_result():
    class _FakeEager:
        records = [{"a": 1}, {"a": 2}]  # dict-able like neo4j Record
        keys = ["a"]
        summary = object()

    class _Neo4jStub:
        async def execute_query(self, q):
            return _FakeEager()

    records, header = await BaseFlavour().execute_graph_query(_Neo4jStub(), "MATCH (n) RETURN n.a AS a")
    assert header == ["a"]
    assert records == [{"a": 1}, {"a": 2}]


@pytest.mark.asyncio
async def test_base_execute_handles_tuple_shape():
    class _TupleStub:  # FalkorDB/AGE-style (records, header, summary)
        async def execute_query(self, q):
            return ([{"c": 5}], ["c"], None)

    records, header = await BaseFlavour().execute_graph_query(_TupleStub(), "RETURN 5 AS c")
    assert header == ["c"]
    assert records == [{"c": 5}]


# --- classify_execution_error(message, query=None): the contract every flavour honours ---

def test_all_flavours_accept_the_query_argument():
    from flavours.age import AgeFlavour
    from flavours.falkordb import FalkorDbFlavour

    for flavour in (BaseFlavour(), FalkorDbFlavour(), AgeFlavour()):
        err = flavour.classify_execution_error("boom", query="MATCH (n) RETURN n")
        assert isinstance(err, CypherError), f"{flavour.name} did not return a CypherError"
        assert err.stage == "execution"


def test_falkordb_classify_ignores_the_query_argument():
    # FalkorDB already carries errCtx inside the message; passing a query must not change it.
    from flavours.falkordb import FalkorDbFlavour

    msg = "errMsg: Invalid input '-': expected '=' errCtx: OPTIONAL MATCH parte-[:R]->(t)"
    without = FalkorDbFlavour().classify_execution_error(msg)
    with_query = FalkorDbFlavour().classify_execution_error(msg, query="OPTIONAL MATCH parte-[:R]->(t)")
    assert without == with_query


def test_classify_execution_error_query_argument_is_optional():
    # Legacy positional-only callers must keep working (backward-compatible default).
    assert BaseFlavour().classify_execution_error("boom").reason == "execution_error"
