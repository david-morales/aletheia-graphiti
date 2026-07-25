"""Offline get_schema canonical-shape test (ADR-019 R5) using a stub driver + flavour."""
import pytest

import graphiti_mcp_server as srv
from flavours.falkordb import FalkorDbFlavour


class _StubDriver:
    """Answers the get_schema probe queries with a tiny fixed graph."""

    async def execute_query(self, query: str, *a, **k):
        if "count(n) AS cnt" in query:
            return [{"lbls": ["Entity", "Persona"], "cnt": 3}], None, None
        if "keys(n)" in query and "UNWIND" in query:
            # top-level keys: a domain field + bookkeeping + an embedding
            return (
                [{"key": "documento"}, {"key": "name"}, {"key": "uuid"}, {"key": "name_embedding"}],
                None,
                None,
            )
        if "type(r) AS rel_type" in query:
            return [{"rel_type": "ES_DETENIDO", "cnt": 2}], None, None
        if "labels(s) AS source_labels" in query:
            return (
                [{"source_labels": ["Entity", "Persona"], "target_labels": ["Entity", "Detencion"]}],
                None,
                None,
            )
        return [], None, None


class _StubClient:
    def __init__(self, driver):
        self.driver = driver


class _StubService:
    def __init__(self, flavour):
        self.flavour = flavour
        self._schema_cache = None
        self._schema_dirty = True
        self.domain_profile = None
        self.config = srv.GraphitiConfig()
        self.config.graphiti.group_id = "policia"

    async def get_client(self):
        return _StubClient(_StubDriver())


@pytest.mark.asyncio
async def test_get_schema_canonical_falkordb(monkeypatch):
    monkeypatch.setattr(srv, "graphiti_service", _StubService(FalkorDbFlavour()))
    schema = await srv.get_schema()

    # ADR-019 R5 canonical top-level fields
    assert schema["dialect"] == "falkordb-cypher"
    assert "FalkorDB" in schema["dialect_reference"]
    # cypher_reference retained as a back-compat alias, now sourced from the flavour
    assert schema["cypher_reference"] == schema["dialect_reference"]

    persona = schema["node_labels"]["Persona"]
    assert persona["count"] == 3
    # attribute_keys = canonical, domain-only (reserved bookkeeping + embeddings excluded)
    assert "documento" in persona["attribute_keys"]
    assert "name" not in persona["attribute_keys"]
    assert "uuid" not in persona["attribute_keys"]
    assert "name_embedding" not in persona["attribute_keys"]
    # properties = full top-level keys (feeds cypher_quality schema_match) — bookkeeping kept,
    # only the embedding stripped
    assert "documento" in persona["properties"]
    assert "name" in persona["properties"]
    assert "uuid" in persona["properties"]
    assert "name_embedding" not in persona["properties"]

    assert "ES_DETENIDO" in schema["relationship_types"]
    assert schema["relationship_types"]["ES_DETENIDO"]["count"] == 2
