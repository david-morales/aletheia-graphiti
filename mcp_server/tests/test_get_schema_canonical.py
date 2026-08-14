"""Offline get_schema canonical-shape test (ADR-019 R5) using a stub driver + flavour."""
import pytest

import graphiti_mcp_server as srv
from flavours.age import AgeFlavour
from flavours.falkordb import FalkorDbFlavour


class _StubDriver:
    """Answers the get_schema probe queries with a tiny fixed graph."""

    def __init__(self):
        self.queries: list[str] = []

    async def execute_query(self, query: str, *a, **k):
        self.queries.append(query)
        if "AS storage_label" in query:
            # FalkorDB stores every label it censuses — the sets are equal.
            return [{"storage_label": "Entity"}, {"storage_label": "Persona"}], None, None
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
        if "AS storage_label" in query:
            # Only the LEAF is a stored label on AGE; Actor exists in the
            # hierarchy list alone, which is the whole distinction under test.
            return [{"storage_label": lbl} for lbl in self._LEAF], None, None
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
            # Leaf columns only when the census asked for them, so this one stub
            # exercises both the leaf-preferring path and the positional fallback.
            src = self._HIERARCHY if "s.labels AS source_labels" in query else self._LEAF
            row = {"source_labels": src, "target_labels": ["Detencion"]}
            if "AS source_leaf" in query:
                row["source_leaf"] = src[-1]
                row["target_leaf"] = "Detencion"
            return [row], None, None
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
    # The POSITIONAL fallback in _pick_endpoint, pinned on the arm that uses it.
    # FalkorDB announces no `source_leaf` column, so the endpoint comes from the
    # filtered label list — ["Entity","Persona"] -> "Persona". Nothing covered this
    # on the falkor side: a _pick_endpoint that returned None with no leaf column
    # left `patterns` EMPTY here and only the AGE test noticed.
    assert schema["relationship_types"]["ES_DETENIDO"]["patterns"] == [["Persona", "Detencion"]]


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
    # EQUALITY, not membership: the hierarchy list is explicitly unordered
    # (SubgraphResponse in models/response_types.py), so a positional pick over it
    # could name any non-internal member — including the abstract supertype, which
    # is not `(n:X)`-matchable and which merges distinct leaf patterns. The census
    # now ALSO announces `label(s) AS source_leaf`, and the leaf is what makes this
    # endpoint deterministic. The positional pick survives as the fallback for
    # flavours that announce no leaf column (FalkorDB / openCypher).
    patterns = schema["relationship_types"]["ES_DETENIDO"]["patterns"]
    assert patterns == [["Persona", "Detencion"]], patterns
    probes = [q for q in driver.queries if "AS source_labels" in q]
    assert probes and all("s.labels AS source_labels" in q for q in probes), probes
    assert probes and all("label(s) AS source_leaf" in q for q in probes), probes


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


# ---------------------------------------------------------------------------
# hierarchy-only labels — censusable/searchable, but not storage types
# ---------------------------------------------------------------------------


class _AgeRichStubDriver(_AgeStubDriver):
    """Two ontology branches, so a LOW-COUNT LEAF is in the picture.

    `Arma` has one vertex on the live bench graph. A consumer guessing which
    entries are abstract from counts, or from "does it appear in patterns",
    would mis-flag exactly that kind of label — it is a real storage type that
    happens to be rare and unconnected. The producer must flag on the storage
    census alone.
    """

    _LEAF = ["Persona", "Arma"]
    _BRANCHES = (["Entity", "Actor", "Persona"], ["Entity", "PhysicalObject", "Arma"])

    async def execute_query(self, query: str, *a, **k):
        self.queries.append(query)
        if "AS storage_label" in query:
            return [{"storage_label": lbl} for lbl in self._LEAF], None, None
        if "count(n) AS cnt" in query:
            if "n.labels AS lbls" in query:
                rows = [{"lbls": list(b), "cnt": 5} for b in self._BRANCHES]
            else:
                rows = [{"lbls": [lbl], "cnt": 5} for lbl in self._LEAF]
            return rows + [{"lbls": None, "cnt": 7}], None, None
        return await super().execute_query(query, *a, **k)


@pytest.mark.asyncio
async def test_get_schema_age_flags_hierarchy_only_labels(monkeypatch):
    """The abstracts are in node_labels on purpose (searchable) but no vertex is
    STORED under them, so `MATCH (n:Actor)` matches nothing and they belong in no
    pattern. The UI rendered them as disconnected schema nodes because nothing in
    the payload told it they are not storage types. Now the producer says so.
    """
    driver = _AgeRichStubDriver()
    monkeypatch.setattr(srv, "graphiti_service", _StubService(AgeFlavour(), driver))
    schema = await srv.get_schema()

    labels = schema["node_labels"]
    flagged = sorted(k for k, v in labels.items() if v.get("hierarchy"))
    assert flagged == ["Actor", "PhysicalObject"], labels
    # ...and every STORAGE type is unflagged — including the rare, unconnected one.
    for leaf in ("Persona", "Arma"):
        assert not labels[leaf].get("hierarchy"), (leaf, labels[leaf])
    # The counts and the rest of the entry are untouched by the flag.
    assert labels["Actor"]["count"] == 5
    assert labels["Arma"]["count"] == 5

    census = [q for q in driver.queries if "AS storage_label" in q]
    assert census and all("label(n) AS storage_label" in q for q in census), census


@pytest.mark.asyncio
async def test_get_schema_falkordb_flags_nothing_as_hierarchy(monkeypatch):
    """Every censused label on FalkorDB is a storage label, so the flag must be
    absent/None on every entry — the proven arm's payload does not change."""
    driver = _StubDriver()
    monkeypatch.setattr(srv, "graphiti_service", _StubService(FalkorDbFlavour(), driver))
    schema = await srv.get_schema()

    labels = schema["node_labels"]
    assert labels, labels
    assert all(not v.get("hierarchy") for v in labels.values()), labels


@pytest.mark.asyncio
async def test_a_failing_storage_census_flags_NOTHING(monkeypatch):
    """The dangerous degrade: an empty storage set makes EVERY label look
    hierarchy-only, which would blank the whole schema view. A census that cannot
    answer must leave every entry unflagged, exactly as before the flag existed.
    """

    class _NoStorageCensus(_AgeRichStubDriver):
        async def execute_query(self, query: str, *a, **k):
            if "AS storage_label" in query:
                raise RuntimeError('relation "storage" does not exist')
            return await super().execute_query(query, *a, **k)

    monkeypatch.setattr(srv, "graphiti_service", _StubService(AgeFlavour(), _NoStorageCensus()))
    schema = await srv.get_schema()

    assert "error" not in schema, schema
    labels = schema["node_labels"]
    assert labels, labels
    assert all(not v.get("hierarchy") for v in labels.values()), labels


@pytest.mark.parametrize(
    "case,rows",
    [
        ("zero_rows", []),
        # AGE names an UNALIASED projection `col0`; every rec.get('storage_label')
        # is then None and the set comes out empty while the query looks healthy.
        ("mis_keyed_col0", [{"col0": "Persona"}, {"col0": "Arma"}]),
    ],
)
@pytest.mark.asyncio
async def test_an_empty_storage_census_flags_NOTHING(monkeypatch, case, rows):
    """The OTHER half of the safe degrade: the census SUCCEEDS but yields nothing
    usable. `found` is then an empty set, and diffing against it flags EVERY
    label as hierarchy-only — the same schema-blanking inversion as a raise, and
    it reaches the diff through the happy path where no except clause guards it.

    Two ways that happens, both live-plausible:

    * zero rows — a graph whose vertices are all label-less, or a backend that
      answers the census with nothing;
    * rows whose column is not `storage_label`. AGE names an UNALIASED projection
      `col0` (the failure mode this module has already been bitten by twice: the
      `RETURN DISTINCT key AS key` fix and the `RETURN n AS n` fix). Every
      `rec.get('storage_label')` is then None and the set comes out empty while
      the query itself looks perfectly healthy.
    """

    class _EmptyStorageCensus(_AgeRichStubDriver):
        def __init__(self, rows):
            super().__init__()
            self._storage_rows = rows

        async def execute_query(self, query: str, *a, **k):
            if "AS storage_label" in query:
                self.queries.append(query)
                return self._storage_rows, None, None
            return await super().execute_query(query, *a, **k)

    driver = _EmptyStorageCensus(rows)
    monkeypatch.setattr(srv, "graphiti_service", _StubService(AgeFlavour(), driver))
    schema = await srv.get_schema()

    assert "error" not in schema, (case, schema)
    labels = schema["node_labels"]
    # The schema is INTACT — same entries, counts untouched...
    assert set(labels) == {"Actor", "Persona", "PhysicalObject", "Arma"}, (case, labels)
    assert all(info["count"] == 5 for info in labels.values()), (case, labels)
    # ...and NOT ONE of them is flagged.
    flagged = sorted(k for k, v in labels.items() if v.get("hierarchy"))
    assert flagged == [], (case, flagged)


# ---------------------------------------------------------------------------
# The dialect-sensitive probes come from the flavour (BLK-1 / H-F4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flavour_cls", [FalkorDbFlavour, AgeFlavour])
@pytest.mark.asyncio
async def test_get_schema_issues_the_flavours_property_keys_probe(monkeypatch, flavour_cls):
    """The top-level `properties` probe was the ONE census text still hardcoded.

    Its literal sat in the server module carrying an AGE-driven fix (`AS key`)
    while its sibling `attribute_keys` probe was already flavour-routed — the
    asymmetry BLK-1 came out of. Pin that get_schema sends the flavour's text
    verbatim, so a flavour that needs a different one can say so.
    """
    flavour = flavour_cls()
    driver = _StubDriver() if flavour_cls is FalkorDbFlavour else _AgeStubDriver()
    monkeypatch.setattr(srv, "graphiti_service", _StubService(flavour, driver))
    await srv.get_schema()

    expected = flavour.property_keys_query("Persona")
    assert expected in driver.queries, (expected, driver.queries)


@pytest.mark.asyncio
async def test_get_schema_announces_the_attribute_container_on_age(monkeypatch):
    """ADR-019 R6, dialect as data: the backend names its own nesting container.

    Consumers (cypher_quality here, agents downstream) must not have to know
    that "AGE" implies "attributes" — the producer says so in the payload.
    """
    monkeypatch.setattr(srv, "graphiti_service", _StubService(AgeFlavour(), _AgeStubDriver()))
    schema = await srv.get_schema()
    assert schema["attribute_container"] == "attributes"


@pytest.mark.asyncio
async def test_get_schema_announces_no_container_on_a_flat_backend(monkeypatch):
    monkeypatch.setattr(srv, "graphiti_service", _StubService(FalkorDbFlavour(), _StubDriver()))
    schema = await srv.get_schema()
    assert schema.get("attribute_container") is None


@pytest.mark.asyncio
async def test_an_age_correct_nested_query_assesses_clean(monkeypatch):
    """BLK-1 end to end: schema OUT of get_schema, verdict INTO the caller.

    The two halves are only worth anything composed. `n.attributes.<key>` is the
    exact access form AGE's own `dialect_reference` teaches, and against a schema
    this connector produced it used to come back `suspect / schema_mismatch` on
    correct rows — a verdict `refine_verdict` then refuses to downgrade, so the
    flag survived to the agent no matter how good the result was.
    """
    from utils.cypher_quality import assess_quality

    monkeypatch.setattr(srv, "graphiti_service", _StubService(AgeFlavour(), _AgeStubDriver()))
    schema = await srv.get_schema()
    assert "documento" in schema["node_labels"]["Persona"]["attribute_keys"], schema

    good = assess_quality(
        "MATCH (n:Persona) RETURN n.name AS name, n.attributes.documento AS documento",
        schema=schema,
    )
    assert good.verdict == "success", good.to_dict()
    assert good.outcome == "ok", good.to_dict()

    # ...and a hallucinated field on the same path is still caught.
    bad = assess_quality(
        "MATCH (n:Persona) RETURN n.attributes.numero_de_serie AS serie", schema=schema
    )
    assert bad.verdict == "schema_mismatch", bad.to_dict()
