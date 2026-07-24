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
