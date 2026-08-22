"""Census-validated property references in the graph_query response.

A property name the graph does not carry is not an error on either backend: the
query runs, the column comes back null for every row, and nothing says so. The
agent reads a clean-looking empty answer and concludes the fact is absent.

These tests pin the warning channel that closes that: it fires only from
MEASURED knowledge (a positive census), it never blocks or rewrites the query,
and it stays silent wherever the census cannot support a verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.schema_warnings import build_schema_warnings


# ---------------------------------------------------------------------------
# Fixtures — census shapes as get_schema publishes them
# ---------------------------------------------------------------------------


def _flat_census(**overrides) -> dict:
    """A FLAT census: domain fields live at the top level, no container."""
    schema = {
        'attribute_container': None,
        'node_labels': {
            'Parte': {
                'count': 20,
                'properties': ['fecha_de_inicio', 'localidad', 'name', 'uuid'],
                'attribute_keys': ['fecha_de_inicio', 'localidad'],
                'sampled': True,
            },
            'Persona': {
                'count': 40,
                'properties': ['fecha_de_nacimiento', 'name', 'nombre_completo', 'uuid'],
                'attribute_keys': ['fecha_de_nacimiento', 'nombre_completo'],
                'sampled': True,
            },
        },
    }
    schema.update(overrides)
    return schema


def _nesting_census(**overrides) -> dict:
    """A NESTING census: the backend announces the map its domain fields sit in."""
    schema = {
        'attribute_container': 'attributes',
        'node_labels': {
            'Parte': {
                'count': 20,
                'properties': ['name', 'uuid'],
                'attribute_keys': ['fecha_de_inicio', 'localidad'],
                'sampled': True,
            },
        },
    }
    schema.update(overrides)
    return schema


def _joined(warnings: list[str]) -> str:
    return '\n'.join(warnings)


# ---------------------------------------------------------------------------
# Unknown property references
# ---------------------------------------------------------------------------


class TestUnknownProperties:
    def test_property_on_no_censused_label_warns(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_inicio', _flat_census()
        )
        assert w, 'a property no censused label carries must be reported'
        assert 'fecha_inicio' in _joined(w)

    def test_warning_offers_nearest_census_matches(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_inicio', _flat_census()
        )
        assert 'fecha_de_inicio' in _joined(w), 'did-you-mean must name the real property'

    def test_property_known_on_another_label_does_not_warn(self):
        """The check is a label-agnostic UNION: no alias->label resolution."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.nombre_completo', _flat_census()
        )
        assert w == []

    def test_known_property_does_not_warn(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_de_inicio', _flat_census()
        )
        assert w == []

    def test_where_clause_reference_is_checked(self):
        w = build_schema_warnings(
            "MATCH (p:Parte) WHERE p.fecha_inicio = '2026-01-23' RETURN p.name",
            _flat_census(),
        )
        assert 'fecha_inicio' in _joined(w)

    def test_reference_after_with_rebinding_is_checked(self):
        """No label binding survives a WITH — the union check does not need one."""
        w = build_schema_warnings(
            'MATCH (p:Parte) WITH p AS q RETURN q.fecha_inicio', _flat_census()
        )
        assert 'fecha_inicio' in _joined(w)

    def test_each_unknown_property_reported_once(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) WHERE p.fecha_inicio IS NOT NULL RETURN p.fecha_inicio',
            _flat_census(),
        )
        assert len([x for x in w if 'fecha_inicio' in x]) == 1

    def test_output_is_deterministic(self):
        query = 'MATCH (p:Parte) RETURN p.zzz_unknown, p.aaa_unknown'
        assert build_schema_warnings(query, _flat_census()) == build_schema_warnings(
            query, _flat_census()
        )


# ---------------------------------------------------------------------------
# Always-allowed properties
# ---------------------------------------------------------------------------


class TestInternalProperties:
    def test_internal_node_properties_never_warn(self):
        for prop in ('uuid', 'name', 'group_id', 'summary', 'created_at', 'labels'):
            w = build_schema_warnings(f'MATCH (n) RETURN n.{prop}', _flat_census())
            assert w == [], f'internal property {prop} must never warn'

    def test_embedding_properties_never_warn(self):
        w = build_schema_warnings('MATCH (n) RETURN n.name_embedding', _flat_census())
        assert w == []

    def test_episodic_properties_never_warn(self):
        for prop in ('content', 'source', 'source_description', 'valid_at'):
            w = build_schema_warnings(f'MATCH (e:Episodic) RETURN e.{prop}', _flat_census())
            assert w == [], f'episodic property {prop} must never warn'

    def test_edge_internal_properties_never_warn(self):
        for prop in ('fact', 'valid_at', 'invalid_at', 'episodes', 'expired_at'):
            w = build_schema_warnings(
                f'MATCH (a)-[r:REL]->(b) RETURN r.{prop}', _flat_census()
            )
            assert w == [], f'edge property {prop} must never warn'

    def test_edge_bound_variable_is_not_judged_against_a_node_census(self):
        """The census describes NODE labels; an edge property is out of its scope."""
        w = build_schema_warnings(
            'MATCH (a)-[r:REL]->(b) RETURN r.peso_relativo', _flat_census()
        )
        assert w == []

    def test_node_reference_still_checked_alongside_an_edge_one(self):
        w = build_schema_warnings(
            'MATCH (a:Parte)-[r:REL]->(b) RETURN r.peso_relativo, a.fecha_inicio',
            _flat_census(),
        )
        assert 'fecha_inicio' in _joined(w)
        assert 'peso_relativo' not in _joined(w)


# ---------------------------------------------------------------------------
# Nested-chain references
# ---------------------------------------------------------------------------


class TestNestedChains:
    def test_nested_chain_warns_on_a_flat_census(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _flat_census()
        )
        assert w, 'a nested read against flat storage must be reported'
        assert 'attributes' in _joined(w)

    def test_nested_chain_warns_even_when_the_leaf_is_real(self):
        """The chain itself is the defect — the leaf being censused does not save it."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _flat_census()
        )
        assert any('fecha_de_inicio' in x and 'attributes' in x for x in w)

    def test_nested_chain_names_the_flat_form_to_use_instead(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _flat_census()
        )
        assert 'p.fecha_de_inicio' in _joined(w)

    def test_nested_chain_with_an_unknown_leaf_reports_both_defects(self):
        """The measured failure: the shape AND the name were both guessed."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_inicio', _flat_census()
        )
        joined = _joined(w)
        assert 'attributes' in joined, 'the nested shape must be reported'
        assert 'fecha_de_inicio' in joined, 'the unknown leaf must still get did-you-mean'

    def test_announced_container_chain_does_not_warn(self):
        """Where the backend ANNOUNCES the container, the nested form is correct."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _nesting_census()
        )
        assert w == []

    def test_wrong_container_warns_even_on_a_nesting_census(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.props.fecha_de_inicio', _nesting_census()
        )
        assert w, 'a map the backend never announced is still a defect'
        assert 'attributes' in _joined(w), 'the announced container must be named'

    def test_deeper_chain_warns(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.a.b.fecha_de_inicio', _flat_census()
        )
        assert w


# ---------------------------------------------------------------------------
# The positive-census gate — a warning may only fire from measured knowledge
# ---------------------------------------------------------------------------


class TestPositiveCensusGate:
    def test_no_schema_means_no_warnings_at_all(self):
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.attributes.whatever', None)
        assert w == []

    def test_an_unsampled_label_silences_unknown_property_warnings(self):
        census = _flat_census()
        census['node_labels']['Persona']['sampled'] = False
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.fecha_inicio', census)
        assert w == [], 'one unsampled label makes the union incomplete evidence'

    def test_an_empty_property_union_silences_unknown_property_warnings(self):
        census = _flat_census(
            node_labels={
                'Parte': {'count': 20, 'properties': [], 'attribute_keys': [], 'sampled': True}
            }
        )
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.fecha_inicio', census)
        assert w == []

    def test_no_censused_labels_silences_unknown_property_warnings(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_inicio', _flat_census(node_labels={})
        )
        assert w == []

    def test_the_chain_warning_survives_a_failed_census_gate(self):
        """The nested-shape defect is schema-independent, so the gate does not bind it."""
        census = _flat_census()
        census['node_labels']['Persona']['sampled'] = False
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.attributes.x', census)
        assert w, 'the chain warning does not depend on the property union'
        assert 'attributes' in _joined(w)

    def test_unparseable_query_yields_no_warnings(self):
        w = build_schema_warnings('this is not cypher at all {{{', _flat_census())
        assert w == []


# ---------------------------------------------------------------------------
# False-positive guards
# ---------------------------------------------------------------------------


class TestNoFalsePositives:
    def test_string_literal_containing_dots_is_not_a_reference(self):
        w = build_schema_warnings(
            "MATCH (p:Parte) WHERE p.localidad = 'a.bogus.value' RETURN p", _flat_census()
        )
        assert w == []

    def test_function_calls_are_not_references(self):
        w = build_schema_warnings(
            'MATCH (n)-[r]->(m) RETURN type(r), keys(n), labels(m), count(*)', _flat_census()
        )
        assert w == []

    def test_map_literal_keys_are_not_references(self):
        w = build_schema_warnings(
            'MATCH (p:Parte {localidad: "x"}) RETURN p.name', _flat_census()
        )
        assert w == []

    def test_parameters_are_not_references(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) WHERE p.localidad = $localidad RETURN p.name', _flat_census()
        )
        assert w == []

    def test_label_predicate_is_not_a_reference(self):
        w = build_schema_warnings(
            'MATCH (n) WHERE n:Parte RETURN n.name', _flat_census()
        )
        assert w == []

    def test_aggregation_alias_is_not_a_reference(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN count(p) AS total ORDER BY total DESC', _flat_census()
        )
        assert w == []


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class TestContract:
    def test_returns_plain_strings(self):
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.fecha_inicio', _flat_census())
        assert all(isinstance(x, str) for x in w)

    def test_warnings_are_domain_agnostic(self):
        """The wording is produced by the module; only the NAMES come from the graph."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_inicio', _flat_census()
        )
        joined = _joined(w).lower()
        for banned in ('aircraft', 'aviation', 'police', 'crime', 'delito', 'parte'):
            assert banned not in joined.replace('p.fecha_inicio', ''), (
                f'warning text must not name a domain: {banned}'
            )
