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

    `unmatchable` names labels that exist in the hierarchy but have NO label
    table on this backend — precisely the supertypes the census now surfaces.
    A label-scoped probe against one of those either raises (AGE's own
    behaviour for an unknown label, `unmatchable_raises=True`) or returns no
    rows; both are reproduced, because they are different failure modes for
    get_schema.

    One census row carries `lbls: None`: AGE returns label-less vertices with a
    null labels list, and a `.get(k, [])` default does NOT cover an explicit null.
    """

    _LEAF = ["Persona"]
    _HIERARCHY = ["Entity", "Actor", "Persona"]

    def __init__(self, unmatchable: set[str] | None = None, unmatchable_raises: bool = True):
        self.queries: list[str] = []
        self.unmatchable = unmatchable or set()
        self.unmatchable_raises = unmatchable_raises

    def _probes_an_unmatchable_label(self, query: str) -> bool:
        return any(f"(n:`{label}`)" in query for label in self.unmatchable)

    async def execute_query(self, query: str, *a, **k):
        self.queries.append(query)
        # Checked FIRST: a label-scoped probe is recognisable by its label, not
        # by what it projects, and this is what the real backend does to it.
        if self._probes_an_unmatchable_label(query):
            if self.unmatchable_raises:
                raise RuntimeError('label "Actor" does not exist')
            return [], None, None
        if "count(n) AS cnt" in query:
            lbls = self._HIERARCHY if "n.labels AS lbls" in query else self._LEAF
            return [{"lbls": lbls, "cnt": 5}, {"lbls": None, "cnt": 7}], None, None
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
async def test_get_schema_falkordb_adds_no_census_notes(monkeypatch):
    """The flavour-announced caveat is AGE-only; FalkorDB's schema is unchanged.

    On FalkorDB every label in `node_labels` IS matchable with `(n:Label)`, so
    there is nothing to announce — and with no domain profile there are no
    description-derived notes either, so the key stays absent entirely.
    """
    monkeypatch.setattr(srv, "graphiti_service", _StubService(FalkorDbFlavour()))
    schema = await srv.get_schema()
    assert schema.get("analysis_notes") is None


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
    # The label-less census row (lbls: None) contributes nothing and crashes nothing.
    assert len(labels) == 2, labels
    assert all(info["count"] == 5 for info in labels.values()), labels

    # The census text is the flavour's, not the base literal.
    census = [q for q in driver.queries if "count(n) AS cnt" in q]
    assert census and all("n.labels AS lbls" in q for q in census), census

    # ...and the relationship-pattern probe, whose rows the SHARED loop parses.
    # Assert MEMBERSHIP, not position: the hierarchy list is explicitly unordered
    # (SubgraphResponse in models/response_types.py), so which non-internal label
    # `src[0]` lands on is arbitrary. RECORDED LIMITATION: the endpoint pick is
    # hierarchy-arbitrary — a pattern may name the supertype where the leaf would
    # read better. A leaf-preferring pick is a possible future improvement; it is
    # pre-existing behaviour shared with FalkorDB and out of scope here.
    patterns = schema["relationship_types"]["ES_DETENIDO"]["patterns"]
    assert len(patterns) == 1, patterns
    src_endpoint, tgt_endpoint = patterns[0]
    assert src_endpoint in {"Actor", "Persona"}, patterns
    assert tgt_endpoint in {"Detencion"}, patterns
    probes = [q for q in driver.queries if "AS source_labels" in q]
    assert probes and all("s.labels AS source_labels" in q for q in probes), probes


@pytest.mark.asyncio
async def test_get_schema_age_announces_the_hierarchy_label_caveat(monkeypatch):
    """The new AGE entries are not matchable with `(n:Label)` — say so (ADR-019 R6).

    Surfacing the supertypes is only half the fix: an agent that reads
    `Actor: {count: 5}` and writes `MATCH (n:Actor)` gets 0 rows on AGE and
    concludes there are no Actors. The flavour announces the caveat as data.
    """
    monkeypatch.setattr(srv, "graphiti_service", _StubService(AgeFlavour(), _AgeStubDriver()))
    schema = await srv.get_schema()

    notes = schema.get("analysis_notes") or []
    assert notes, "AGE schema must carry the flavour's census note"
    for note in AgeFlavour().census_notes():
        assert note in notes, notes
    joined = " ".join(notes)
    assert "n.labels" in joined  # the form that DOES match a hierarchy label
    assert "(n:" in joined       # ...named as the form that does not


@pytest.mark.asyncio
async def test_get_schema_age_survives_a_label_probe_that_raises(monkeypatch):
    """A supertype has no label table on AGE — probing it must not sink get_schema.

    The census now feeds `MATCH (n:\\`{label}\\`)` labels that exist only in the
    stored hierarchy. On AGE that probe can raise, and an unhandled raise here
    returns `{'error': ...}` for the WHOLE call — a total get_schema outage on
    AGE, the exact class of failure tests/live/test_two_flavour_parity_live.py
    guards. Mirrors AgeFlavour.attribute_keys: a probe that cannot answer
    degrades one entry, it does not break the schema.
    """
    driver = _AgeStubDriver(unmatchable={"Actor"}, unmatchable_raises=True)
    monkeypatch.setattr(srv, "graphiti_service", _StubService(AgeFlavour(), driver))
    schema = await srv.get_schema()

    assert "error" not in schema, schema
    labels = schema["node_labels"]
    assert set(labels) == {"Actor", "Persona"}
    # The degraded entry: census count survives, probe output is empty and honest.
    assert labels["Actor"]["count"] == 5
    assert labels["Actor"]["properties"] == []
    assert labels["Actor"]["attribute_keys"] == []
    assert labels["Actor"]["sampled"] is False
    # ...and the matchable leaf is untouched by its neighbour's failure.
    assert labels["Persona"]["properties"] == ["name", "uuid"]
    assert labels["Persona"]["attribute_keys"] == ["documento"]
    assert labels["Persona"]["sampled"] is True


@pytest.mark.asyncio
async def test_get_schema_marks_an_empty_label_probe_unsampled(monkeypatch):
    """The other AGE outcome: the probe answers, with no rows. Same honesty rule."""
    driver = _AgeStubDriver(unmatchable={"Actor"}, unmatchable_raises=False)
    monkeypatch.setattr(srv, "graphiti_service", _StubService(AgeFlavour(), driver))
    schema = await srv.get_schema()

    labels = schema["node_labels"]
    assert labels["Actor"]["properties"] == []
    assert labels["Actor"]["sampled"] is False, "an unsampled entry must not claim sampled"
    assert labels["Actor"]["count"] == 5
    assert labels["Persona"]["sampled"] is True
