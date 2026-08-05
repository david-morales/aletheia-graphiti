"""Flavour-owned subgraph sampling queries (Step 2 UI alignment).

The Knowledge-tab sample must be flavour-correct: FalkorDB nodes are
multi-label (labels(n) IS the hierarchy, :Entity scoping works); AGE nodes
carry one leaf label plus the stored n.labels list (labels(n) is NOT the
hierarchy, :Entity matches almost nothing, Episodic vertices have no labels
list). These tests pin the query TEXT contracts per flavour.
"""
from flavours.age import AgeFlavour
from flavours.base import BaseFlavour
from flavours.falkordb import FalkorDbFlavour


def test_base_node_query_scopes_by_entity_and_uses_labels_fn():
    q = BaseFlavour().subgraph_node_query()
    assert "MATCH (n:Entity)" in q
    assert "labels(n) AS labels" in q
    assert "$limit" in q


def test_age_node_query_never_scopes_by_entity_and_reads_the_stored_list():
    q = AgeFlavour().subgraph_node_query()
    assert ":Entity" not in q
    assert "labels(n)" not in q
    assert "n.labels IS NOT NULL" in q  # excludes Episodic (no labels list)
    assert "n.labels AS labels" in q
    assert "$limit" in q


def test_falkordb_inherits_the_base_subgraph_queries():
    assert FalkorDbFlavour().subgraph_node_query() == BaseFlavour().subgraph_node_query()


def test_age_node_query_diversifies_the_sample_with_an_order_by():
    """AGE keeps ONE TABLE PER LABEL, so an unordered scan is label-sequential:
    live at limit 25 that was 1 label set and 0 edges (falkor: 10 sets, 26 edges).

    The diversifier is `created_at`, NOT the uuid. Both break the per-label
    grouping, but ingestion time also PRESERVES cluster locality — a parte is
    written with its roles and actors — which is what puts both endpoints of an
    edge in the sample. Measured live at limit 25: uuid -> 9 label sets but 0
    edges; created_at -> 6 sets and 23 edges."""
    q = AgeFlavour().subgraph_node_query()
    assert "ORDER BY n.created_at" in q, q
    # Before the LIMIT, or it orders the already-truncated page — i.e. nothing.
    assert q.index("ORDER BY n.created_at") < q.index("$limit"), q


def test_base_node_query_keeps_storage_order():
    """Byte-identical pin: falkor's storage order already interleaves types (10 label
    sets at limit 25 live), so it needs no ORDER BY — and a sort would cost a scan."""
    for flavour in (BaseFlavour(), FalkorDbFlavour()):
        assert "ORDER BY" not in flavour.subgraph_node_query(), flavour.name
    assert BaseFlavour().subgraph_node_query() == (
        "MATCH (n:Entity) "
        "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, "
        "n.created_at AS created_at, n.summary AS summary, n.group_id AS group_id "
        "LIMIT $limit"
    )


# ---------------------------------------------------------------------------
# flatten_node_props / property_accessor — the AGE nested `attributes` map
# ---------------------------------------------------------------------------


def test_age_node_sample_query_aliases_the_projection():
    """`RETURN n` is an UNALIASED variable projection, and AGE names those `col0`
    — live: header `['col0']`, so a reader keyed on `n` gets nothing while rows
    still exist (that is `sampled: true` beside `properties: {}`). Same class as
    get_schema's `RETURN DISTINCT key AS key`."""
    q = AgeFlavour().node_sample_query()
    assert "RETURN n AS n" in q, q
    assert "{label}" in q and "{limit}" in q, q


def test_base_node_sample_query_is_byte_identical():
    for flavour in (BaseFlavour(), FalkorDbFlavour()):
        assert flavour.node_sample_query() == "MATCH (n:`{label}`) RETURN n LIMIT {limit}"


def test_age_flatten_node_props_unwraps_the_vertex_envelope():
    """AGE hands back the whole vertex — `{'id', 'label', 'properties': {...}}` —
    not a flat property dict (live repr from the driver)."""
    flat = AgeFlavour().flatten_node_props(
        {
            "id": 2814749767106562,
            "label": "Persona",
            "properties": {
                "name": "OMAR MOHAMED",
                "uuid": "365fd6da",
                "attributes": {"documento": "NIE Z8217547C", "filiacion": "hijo de IBRAHIM"},
            },
        }
    )
    assert flat["documento"] == "NIE Z8217547C"
    assert flat["filiacion"] == "hijo de IBRAHIM"
    assert flat["name"] == "OMAR MOHAMED"
    assert "properties" not in flat and "attributes" not in flat, flat


def test_base_flatten_node_props_is_identity():
    props = {"name": "OMAR", "documento": "X1234567L", "attributes": {"ignored": 1}}
    # Base keeps every key exactly as stored — including a literal `attributes`
    # property, which on openCypher/FalkorDB is just another top-level field.
    assert BaseFlavour().flatten_node_props(props) == props
    assert FalkorDbFlavour().flatten_node_props(props) == props


def test_age_flatten_node_props_merges_the_nested_attributes_map():
    """On AGE the descriptive fields live in a queryable `attributes` MAP, so a
    reader that only walks the top level sees no domain properties at all."""
    flat = AgeFlavour().flatten_node_props(
        {
            "uuid": "u1",
            "name": "OMAR MOHAMED",
            "attributes": {"documento": "X1234567L", "edad": 34},
        }
    )
    assert flat["documento"] == "X1234567L"
    assert flat["edad"] == 34
    assert flat["name"] == "OMAR MOHAMED"
    # The container itself is not a domain property — it is the transport.
    assert "attributes" not in flat, flat


def test_age_flatten_node_props_lets_the_top_level_win_on_collision():
    """Non-destructive: a nested field never overwrites a stored top-level one.
    `name`/`summary` are Graphiti's own columns and stay authoritative."""
    flat = AgeFlavour().flatten_node_props(
        {"name": "top-level", "attributes": {"name": "nested", "documento": "X1"}}
    )
    assert flat["name"] == "top-level", flat
    assert flat["documento"] == "X1"


def test_age_flatten_node_props_tolerates_a_missing_or_scalar_attributes():
    for props in ({"name": "a"}, {"name": "a", "attributes": None}, {"attributes": "str"}):
        assert AgeFlavour().flatten_node_props(props) == props


def test_base_property_accessor_is_the_plain_top_level_path():
    assert BaseFlavour().property_accessor("documento") == "n.`documento`"
    assert FalkorDbFlavour().property_accessor("name") == "n.`name`"


def test_age_property_accessor_reaches_into_the_attributes_map():
    """A full-scan probe on the TOP-level path is valid Cypher that matches nothing
    — it returns 0/0 and silently overwrites the sample-based coverage with 0.0."""
    assert AgeFlavour().property_accessor("documento") == "n.attributes.`documento`"
    # ...but Graphiti's own columns really are top-level on AGE.
    for bookkeeping in ("name", "summary", "labels", "uuid", "group_id", "created_at"):
        assert AgeFlavour().property_accessor(bookkeeping) == f"n.`{bookkeeping}`", bookkeeping


def test_edge_query_inlines_only_sanitized_uuids():
    uuids = ["abc-123", 'evil"uuid', "def-456"]
    q = BaseFlavour().subgraph_edge_query(uuids)
    assert '"abc-123"' in q and '"def-456"' in q
    assert "evil" not in q  # non-uuid charset dropped, not escaped
    # The offending uuid is DROPPED WHOLE, never char-stripped: a stripping impl would
    # keep its safe chars as a third entry ("ed") and still satisfy `"evil" not in q`.
    # Pin the exact literal so only whole-entry rejection passes.
    assert '["abc-123", "def-456"]' in q
    # ...and exactly two quoted entries overall. 8, not 4: the literal is inlined twice
    # (once per endpoint), 4 quotes each.
    assert q.count('"') == 8
    assert "$limit" in q


def test_age_edge_query_matches_unlabelled_endpoints():
    q = AgeFlavour().subgraph_edge_query(["abc-123"])
    assert ":Entity" not in q
    assert "MATCH (s)-[r]->(t)" in q
    assert "s.uuid IN" in q and "t.uuid IN" in q


def test_both_node_queries_project_the_same_columns():
    # The per-flavour tests above pin only the MATCH scope and the labels source, so a
    # column dropped from ONE flavour's projection (e.g. `n.group_id AS group_id` on AGE)
    # would slip through and silently None that SubgraphNode field on that flavour alone.
    cols = ("AS uuid", "AS name", "AS labels", "AS created_at", "AS summary", "AS group_id")
    for flavour in (BaseFlavour(), AgeFlavour()):
        q = flavour.subgraph_node_query()
        for c in cols:
            assert c in q, (flavour.name, c)


def test_both_edge_queries_project_the_same_columns():
    cols = ("AS uuid", "AS name", "AS fact", "AS source_node_uuid", "AS target_node_uuid", "AS created_at")
    for flavour in (BaseFlavour(), AgeFlavour()):
        q = flavour.subgraph_edge_query(["abc-123"])
        for c in cols:
            assert c in q, (flavour.name, c)


# --- get_schema censuses -----------------------------------------------------
# Same seam, one layer up: get_schema's label census and relationship-pattern
# probe are dialect-sensitive for exactly the same reason as the subgraph
# sample, so the flavour owns their text too.


def test_base_census_uses_labels_fn():
    q = BaseFlavour().census_queries()
    assert "labels(n) AS lbls" in q["label_counts"]
    assert "{rel_type}" in q["rel_patterns"]
    assert "labels(s) AS source_labels" in q["rel_patterns"]


def test_age_census_reads_the_stored_list_so_supertypes_are_counted():
    q = AgeFlavour().census_queries()
    assert "n.labels AS lbls" in q["label_counts"]
    assert "n.labels IS NOT NULL" in q["label_counts"]
    assert "s.labels AS source_labels" in q["rel_patterns"]
    assert "s.labels IS NOT NULL" in q["rel_patterns"]


def test_falkordb_inherits_the_base_censuses():
    assert FalkorDbFlavour().census_queries() == BaseFlavour().census_queries()


def test_both_censuses_project_the_same_aliases():
    # get_schema parses both flavours' rows with ONE loop (rec['lbls'],
    # rec['source_labels'], rec['target_labels'], rec['cnt']), so a renamed
    # alias on one flavour silently empties that flavour's schema.
    for flavour in (BaseFlavour(), AgeFlavour()):
        q = flavour.census_queries()
        assert "AS lbls" in q["label_counts"], flavour.name
        assert "count(n) AS cnt" in q["label_counts"], flavour.name
        assert "AS source_labels" in q["rel_patterns"], flavour.name
        assert "AS target_labels" in q["rel_patterns"], flavour.name


def test_age_rel_patterns_also_announces_the_endpoint_leaf():
    """On AGE `source_labels` is the whole HIERARCHY, so a positional pick lands
    on an abstract supertype: unmatchable, and it collapses distinct leaf patterns
    into one. `label(n)` returns exactly the leaf, so the census announces it as an
    OPTIONAL extra column the consumers prefer when present."""
    q = AgeFlavour().census_queries()["rel_patterns"]
    assert "label(s) AS source_leaf" in q, q
    assert "label(t) AS target_leaf" in q, q
    # ...and the shared aliases are still there: the leaf columns are additive.
    assert "s.labels AS source_labels" in q, q
    assert "t.labels AS target_labels" in q, q


def test_base_rel_patterns_announces_no_leaf_columns():
    """The base/FalkorDB text stays byte-identical: `labels(s)` is already the
    matchable label set there, so the consumers keep the positional pick."""
    for flavour in (BaseFlavour(), FalkorDbFlavour()):
        q = flavour.census_queries()["rel_patterns"]
        assert "source_leaf" not in q, flavour.name
        assert "target_leaf" not in q, flavour.name
    assert BaseFlavour().census_queries()["rel_patterns"] == (
        "MATCH (s)-[r:`{rel_type}`]->(t) "
        "RETURN DISTINCT labels(s) AS source_labels, labels(t) AS target_labels LIMIT 20"
    )


def test_base_announces_no_census_caveat():
    # Nothing to warn about: on FalkorDB/Neo4j every censused label is matchable.
    assert BaseFlavour().census_notes() == []
    assert FalkorDbFlavour().census_notes() == []


def test_age_announces_that_hierarchy_labels_are_not_matchable():
    notes = AgeFlavour().census_notes()
    assert notes, "AGE must announce that censused supertypes are not (n:X)-matchable"
    joined = " ".join(notes)
    assert "n.labels" in joined     # the form that works
    assert "(n:" in joined          # the form that silently returns 0 rows
    # Same convention as the description-derived notes get_schema already emits,
    # so a consumer filtering on the prefix sees this one too.
    assert all(n.startswith("IMPORTANT:") for n in notes), notes


def test_rel_patterns_placeholder_formats_with_the_rel_type():
    # The placeholder is the only substitution get_schema makes; if a flavour
    # dropped or renamed it, .format() would raise or return an unfiltered probe.
    for flavour in (BaseFlavour(), AgeFlavour()):
        formatted = flavour.census_queries()["rel_patterns"].format(rel_type="ES_DETENIDO")
        assert "`ES_DETENIDO`" in formatted, flavour.name
        assert "{rel_type}" not in formatted, flavour.name


# --- ontology read path ------------------------------------------------------
# Same seam again, for the three ontology tiers. Two dialect axes, not one:
#   (a) STORAGE — FalkorDB keeps every descriptive ontology field as a top-level
#       node property; AGE keeps only uuid/name/summary/group_id/labels/created_at
#       there and nests the rest in an `attributes` agtype map.
#   (b) RELATIONSHIPS — FalkorDB stores them as RELATES_TO edges whose `name`
#       property carries the real relation name; AGE materializes that name as the
#       edge LABEL (age_graph_operations._edge_label), so it has ZERO RELATES_TO
#       edges and the relation name is `type(r)`.

BASE_CLASS_CONTEXT_QUERY = (
    "MATCH (n:OntologyClass) "
    "RETURN n.uuid AS uuid, "
    "n.name AS name, "
    "n.ontology_type AS ontology_type, "
    "n.inherits_from AS inherits_from, "
    "n.summary AS summary, "
    "n.alt_labels AS alt_labels, "
    "n.source_entity AS source_entity, "
    "n.target_entity AS target_entity, "
    "n.examples AS examples, "
    "n.properties AS properties, "
    "n.identity AS identity"
)

BASE_STRUCTURE_QUERY = (
    "MATCH (n:OntologyClass) "
    "RETURN n.name AS name, "
    "n.ontology_type AS ontology_type, "
    "n.inherits_from AS inherits_from, "
    "n.summary AS summary, "
    "n.alt_labels AS alt_labels, "
    "n.source_entity AS source_entity, "
    "n.target_entity AS target_entity, "
    "n.examples AS examples"
)

BASE_RELATES_QUERY = (
    "MATCH (a:OntologyClass)-[r:RELATES_TO]->(b:OntologyClass) "
    "RETURN a.name AS source, r.name AS name, r.fact AS fact, b.name AS target"
)

# Every attribute-backed column, i.e. everything the AGE arm returned empty.
_ONTOLOGY_ATTR_COLUMNS = (
    "ontology_type",
    "inherits_from",
    "alt_labels",
    "source_entity",
    "target_entity",
    "examples",
)


def test_base_ontology_queries_project_top_level_attrs():
    q = BaseFlavour().ontology_queries()
    assert "n.ontology_type AS ontology_type" in q["class_context"]
    assert "RELATES_TO" in q["relates"]


def test_age_ontology_queries_read_the_nested_attributes_map_and_typed_edges():
    q = AgeFlavour().ontology_queries()
    assert "n.attributes.ontology_type AS ontology_type" in q["class_context"]
    assert "n.attributes.inherits_from AS inherits_from" in q["class_context"]
    # The edge pattern loses its TYPE CONSTRAINT (AGE materializes the relation
    # name as the label, so `[r:RELATES_TO]` reaches almost nothing)...
    assert "[r:RELATES_TO]" not in q["relates"]
    assert "-[r]->" in q["relates"]


def test_age_relates_reads_the_stored_name_not_the_edge_label():
    """`r.name`, not `type(r)` — the label is lossy and the property never is.

    `_edge_label`'s `_IDENT_RE` is ASCII-only (`^[A-Za-z_][A-Za-z0-9_]*$`) and the
    ontology loader does no accent folding, so an accented relation
    (INVOLUCRA_MUNICIÓN) is stored under the RELATES_TO FALLBACK label while
    `r.name` keeps the true name — which BOTH AGE edge write paths persist
    top-level, in the same props dict the `fact` read relies on. Reading `type(r)`
    would rename that relation to "RELATES_TO" and, because
    _edge_relationship_entry rebuilds the fact prefix from this name, also leave
    its summary unstripped.

    It also makes the AGE text a token-for-token mirror of the base: the ONLY
    difference between the two `relates` queries is the edge type constraint.
    """
    q = AgeFlavour().ontology_queries()["relates"]
    assert "r.name AS name" in q, q
    assert "type(r)" not in q, q
    assert q == BASE_RELATES_QUERY.replace("[r:RELATES_TO]", "[r]"), q


def test_base_ontology_queries_are_byte_identical_to_the_shipped_texts():
    """The FalkorDB arm is PROVEN correct — the seam must not perturb it by a byte.

    A reworded-but-equivalent base text would pass every alias/substring pin above
    while changing what the live FalkorDB connector issues, so pin the whole string.
    """
    for flavour in (BaseFlavour(), FalkorDbFlavour()):
        q = flavour.ontology_queries()
        assert q["class_context"] == BASE_CLASS_CONTEXT_QUERY, flavour.name
        assert q["structure"] == BASE_STRUCTURE_QUERY, flavour.name
        assert q["relates"] == BASE_RELATES_QUERY, flavour.name


def test_falkordb_inherits_the_base_ontology_queries():
    assert FalkorDbFlavour().ontology_queries() == BaseFlavour().ontology_queries()


def test_age_reads_every_attribute_backed_column_from_the_nested_map():
    """Not just the two the spike named: EVERY field AGE nests must move.

    A partial fix leaves `alt_labels`/`examples`/`source_entity`/`target_entity`
    empty on the AGE arm — the same silent hole, one field narrower.
    """
    q = AgeFlavour().ontology_queries()
    for key in ("class_context", "structure"):
        for col in _ONTOLOGY_ATTR_COLUMNS:
            assert f"n.attributes.{col} AS {col}" in q[key], (key, col)
            # ...and the broken top-level form is GONE, not merely shadowed.
            assert f"n.{col} AS {col}" not in q[key].replace(f"n.attributes.{col}", ""), (key, col)
    # class_context alone carries the two full-detail extras.
    for col in ("properties", "identity"):
        assert f"n.attributes.{col} AS {col}" in q["class_context"], col
        assert col not in q["structure"], col


def test_age_keeps_the_genuinely_top_level_columns_top_level():
    """uuid/name/summary ARE top-level on AGE — nesting them would break the arm
    that currently works (name/summary were the only two fields the AGE payload
    ever populated)."""
    q = AgeFlavour().ontology_queries()
    assert "n.uuid AS uuid" in q["class_context"]
    assert "n.name AS name" in q["class_context"]
    assert "n.summary AS summary" in q["class_context"]
    assert "n.attributes.name" not in q["class_context"]
    assert "n.attributes.summary" not in q["class_context"]


def test_both_ontology_queries_project_the_same_aliases():
    # The parsing is shared (_ontology_full_entry, _combine_relationship_entries,
    # the structure loop), so a renamed alias on one flavour silently empties that
    # flavour's field — exactly the failure mode this task exists to remove.
    for flavour in (BaseFlavour(), AgeFlavour()):
        q = flavour.ontology_queries()
        for col in ("uuid", "name", "summary", "properties", "identity", *_ONTOLOGY_ATTR_COLUMNS):
            assert f"AS {col}" in q["class_context"], (flavour.name, col)
        for col in ("name", "summary", *_ONTOLOGY_ATTR_COLUMNS):
            assert f"AS {col}" in q["structure"], (flavour.name, col)
        for col in ("source", "name", "fact", "target"):
            assert f"AS {col}" in q["relates"], (flavour.name, col)


def test_both_relates_queries_scope_both_endpoints_to_ontology_classes():
    # The endpoint scope is what excludes the individual-level edges (INSTANCE_OF,
    # BROADER) and Graphiti's Episodic bookkeeping from the relationship listing.
    for flavour in (BaseFlavour(), AgeFlavour()):
        q = flavour.ontology_queries()["relates"]
        assert "(a:OntologyClass)" in q, flavour.name
        assert "(b:OntologyClass)" in q, flavour.name


def test_neither_relates_query_filters_subclass_of_at_the_flavour():
    """Mirror the base ROW SEMANTICS: hierarchy edges are IN this row set.

    On FalkorDB every ontology edge is a RELATES_TO whose `name` may be
    SUBCLASS_OF, so the base row set includes them and
    `_combine_relationship_entries` drops them downstream — one filter, shared by
    both arms. An AGE-side `WHERE type(r) <> 'SUBCLASS_OF'` would move that
    decision into the flavour and make the two arms disagree about what the query
    returns, which is what the shared parsing cannot survive.
    """
    for flavour in (BaseFlavour(), AgeFlavour()):
        assert "SUBCLASS_OF" not in flavour.ontology_queries()["relates"], flavour.name


def test_age_relates_reads_the_edge_fact_top_level():
    """`fact` is NOT nested on AGE edges: age_graph_operations.edge_save writes it
    into the edge props (`SET r += {...}`), and only custom `attributes` nest. The
    relationship summary is derived from it, so reading `r.attributes.fact` would
    blank every AGE relationship doc."""
    q = AgeFlavour().ontology_queries()["relates"]
    assert "r.fact AS fact" in q, q
    assert "r.attributes" not in q, q
