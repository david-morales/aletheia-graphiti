"""Offline get_schema canonical-shape test (ADR-019 R5) using a stub driver + flavour."""
import pytest

import graphiti_mcp_server as srv
from flavours.age import AgeFlavour
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


class _AgeStubDriver:
    """A driver with Apache AGE's label semantics, not FalkorDB's.

    On AGE a vertex carries exactly ONE stored label (its ontology leaf), so
    `labels(n)` answers `['Persona']`; the full hierarchy lives in the stored
    list property `n.labels` (`['Entity', 'Actor', 'Persona']`). The stub
    answers each census according to which of the two the query asked for —
    that is what makes the abstract supertype visible or invisible.
    """

    _LEAF = ["Persona"]
    _HIERARCHY = ["Entity", "Actor", "Persona"]

    def __init__(self):
        self.queries: list[str] = []

    async def execute_query(self, query: str, *a, **k):
        self.queries.append(query)
        if "count(n) AS cnt" in query:
            lbls = self._HIERARCHY if "n.labels AS lbls" in query else self._LEAF
            return [{"lbls": lbls, "cnt": 5}], None, None
        if "keys(n.attributes)" in query:
            return [{"ks": ["documento"]}], None, None
        if "keys(n)" in query and "UNWIND" in query:
            return [{"key": "name"}, {"key": "uuid"}], None, None
        if "type(r) AS rel_type" in query:
            return [{"rel_type": "ES_DETENIDO", "cnt": 2}], None, None
        if "AS source_labels" in query:
            src = self._HIERARCHY if "s.labels AS source_labels" in query else self._LEAF
            return (
                [{"source_labels": src, "target_labels": ["Detencion"]}],
                None,
                None,
            )
        return [], None, None


class _StubService:
    def __init__(self, flavour, driver=None):
        self.flavour = flavour
        self._driver = driver or _StubDriver()
        self._schema_cache = None
        self._schema_dirty = True
        self.domain_profile = None
        self.config = srv.GraphitiConfig()
        self.config.graphiti.group_id = "policia"

    async def get_client(self):
        return _StubClient(self._driver)


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


@pytest.mark.asyncio
async def test_get_schema_age_census_surfaces_abstract_supertypes(monkeypatch):
    """On AGE the census must read the stored `n.labels` list, not `labels(n)`.

    `labels(n)` returns only the leaf on AGE, so every abstract supertype
    (Actor) was invisible in the schema while the leaf (Persona) was present —
    the UI and the agent saw a truncated ontology. Reading the stored list
    counts both, and `Entity` stays filtered as internal bookkeeping.
    """
    driver = _AgeStubDriver()
    monkeypatch.setattr(srv, "graphiti_service", _StubService(AgeFlavour(), driver))
    schema = await srv.get_schema()

    assert schema["dialect"] == "age-opencypher"
    labels = schema["node_labels"]
    assert "Actor" in labels, "abstract supertype missing from the AGE census"
    assert "Persona" in labels
    assert "Entity" not in labels  # internal bookkeeping label, filtered as before
    assert labels["Actor"]["count"] == 5
    assert labels["Persona"]["count"] == 5

    # The census text is the flavour's, not the base literal.
    census = [q for q in driver.queries if "count(n) AS cnt" in q]
    assert census and all("n.labels AS lbls" in q for q in census), census

    # ...and the relationship-pattern probe, whose rows the SHARED loop parses.
    patterns = schema["relationship_types"]["ES_DETENIDO"]["patterns"]
    assert patterns == [["Actor", "Detencion"]]
    probes = [q for q in driver.queries if "AS source_labels" in q]
    assert probes and all("s.labels AS source_labels" in q for q in probes), probes
