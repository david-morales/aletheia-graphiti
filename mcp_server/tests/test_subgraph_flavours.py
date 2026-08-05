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


def test_rel_patterns_placeholder_formats_with_the_rel_type():
    # The placeholder is the only substitution get_schema makes; if a flavour
    # dropped or renamed it, .format() would raise or return an unfiltered probe.
    for flavour in (BaseFlavour(), AgeFlavour()):
        formatted = flavour.census_queries()["rel_patterns"].format(rel_type="ES_DETENIDO")
        assert "`ES_DETENIDO`" in formatted, flavour.name
        assert "{rel_type}" not in formatted, flavour.name
