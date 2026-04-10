"""Tests for schema-aware Cypher quality assessment."""
from __future__ import annotations

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cypher_quality import UnknownProp, WrongLabelProp, assess_quality

SCHEMA = {
    'node_labels': {
        'Evento': {
            'count': 100,
            'properties': ['name', 'summary', 'date', 'group_id', 'created_at'],
        },
        'Persona': {
            'count': 50,
            'properties': ['name', 'summary', 'edad', 'group_id'],
        },
        'Ubicacion': {
            'count': 30,
            'properties': ['name', 'summary', 'municipio', 'group_id'],
        },
    },
    'relationship_types': {
        'INVOLUCRADO_EN': {
            'count': 80,
            'patterns': [['Persona', 'Evento']],
        },
        'OCURRIO_EN': {
            'count': 60,
            'patterns': [['Evento', 'Ubicacion']],
        },
    },
}


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


class TestLabelValidation:
    def test_all_labels_found(self):
        q = assess_quality('MATCH (n:Evento) RETURN n', schema=SCHEMA)
        assert q.schema_match is not None
        assert q.schema_match.labels.found == ['Evento']
        assert q.schema_match.labels.unknown == []

    def test_unknown_label(self):
        q = assess_quality('MATCH (n:Accidente) RETURN n', schema=SCHEMA)
        assert 'Accidente' in q.schema_match.labels.unknown
        assert q.verdict == 'schema_mismatch'

    def test_unknown_label_suggestion(self):
        q = assess_quality('MATCH (n:Event) RETURN n', schema=SCHEMA)
        assert 'Event' in q.schema_match.labels.unknown
        assert q.schema_match.labels.suggestions.get('Event') == 'Evento'

    def test_no_labels_skips_validation(self):
        q = assess_quality('CALL db.labels()', schema=SCHEMA)
        assert q.schema_match is not None
        assert q.schema_match.labels.found == []
        assert q.schema_match.labels.unknown == []


# ---------------------------------------------------------------------------
# Relationship validation
# ---------------------------------------------------------------------------


class TestRelationshipValidation:
    def test_known_rel_type(self):
        q = assess_quality(
            'MATCH (p:Persona)-[:INVOLUCRADO_EN]->(e:Evento) RETURN p, e',
            schema=SCHEMA,
        )
        assert 'INVOLUCRADO_EN' in q.schema_match.relationships.found
        assert q.schema_match.relationships.unknown == []

    def test_unknown_rel_type(self):
        q = assess_quality(
            'MATCH (p:Persona)-[:TIENE]->(e:Evento) RETURN p, e',
            schema=SCHEMA,
        )
        assert 'TIENE' in q.schema_match.relationships.unknown
        assert q.verdict == 'schema_mismatch'

    def test_wrong_direction(self):
        # Schema says INVOLUCRADO_EN goes Persona -> Evento
        # This query has Evento -> Persona (wrong direction)
        q = assess_quality(
            'MATCH (e:Evento)-[:INVOLUCRADO_EN]->(p:Persona) RETURN e, p',
            schema=SCHEMA,
        )
        assert 'INVOLUCRADO_EN' in q.schema_match.relationships.wrong_direction
        assert q.verdict == 'schema_mismatch'


# ---------------------------------------------------------------------------
# Property validation
# ---------------------------------------------------------------------------


class TestPropertyValidation:
    def test_known_property(self):
        q = assess_quality(
            'MATCH (n:Evento) RETURN n.name, n.date',
            schema=SCHEMA,
        )
        assert 'name' in q.schema_match.properties.found or any(
            p in q.schema_match.properties.found for p in ['name', 'date']
        )
        assert q.schema_match.properties.unknown == []
        assert q.schema_match.properties.wrong_label == []

    def test_unknown_property(self):
        q = assess_quality(
            'MATCH (n:Evento) RETURN n.nombre',
            schema=SCHEMA,
        )
        unknown_props = q.schema_match.properties.unknown
        assert any(u.property_name == 'nombre' and u.on_label == 'Evento' for u in unknown_props)
        assert q.verdict == 'schema_mismatch'

    def test_wrong_label_property(self):
        # 'edad' exists on Persona, not on Evento
        q = assess_quality(
            'MATCH (n:Evento) RETURN n.edad',
            schema=SCHEMA,
        )
        wrong = q.schema_match.properties.wrong_label
        assert any(
            w.property_name == 'edad' and w.on_label == 'Evento' and 'Persona' in w.exists_on
            for w in wrong
        )
        assert q.verdict == 'schema_mismatch'

    def test_unbound_variable_skipped(self):
        q = assess_quality('MATCH (n) RETURN n.name', schema=SCHEMA)
        # No label binding for 'n', so property validation is skipped
        assert q.schema_match.properties.unknown == []
        assert q.schema_match.properties.wrong_label == []

    def test_internal_properties_valid(self):
        q = assess_quality(
            'MATCH (n:Evento) RETURN n.group_id, n.created_at, n.uuid, n.updated_at',
            schema=SCHEMA,
        )
        assert q.schema_match.properties.unknown == []
        assert q.schema_match.properties.wrong_label == []
        assert q.verdict == 'success'


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_clean_query_success(self):
        q = assess_quality(
            'MATCH (p:Persona)-[:INVOLUCRADO_EN]->(e:Evento) RETURN p.name, e.date',
            schema=SCHEMA,
        )
        assert q.outcome == 'ok'
        assert q.verdict == 'success'

    def test_schema_mismatch_on_label(self):
        q = assess_quality('MATCH (n:Accidente) RETURN n', schema=SCHEMA)
        assert q.outcome == 'suspect'
        assert q.verdict == 'schema_mismatch'

    def test_schema_mismatch_on_property(self):
        q = assess_quality('MATCH (n:Evento) RETURN n.nombre', schema=SCHEMA)
        assert q.outcome == 'suspect'
        assert q.verdict == 'schema_mismatch'

    def test_no_schema_returns_success(self):
        q = assess_quality('MATCH (n:Evento) RETURN n', schema=None)
        assert q.outcome == 'ok'
        assert q.verdict == 'success'
        assert q.schema_match is None

    def test_parse_error_verdict(self):
        q = assess_quality('MATCH (n:Evento RETURN n', schema=SCHEMA)
        assert q.outcome == 'suspect'
        assert q.verdict == 'parse_failed'


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_structure(self):
        q = assess_quality(
            'MATCH (p:Persona)-[:INVOLUCRADO_EN]->(e:Evento) RETURN p.name, e.date',
            schema=SCHEMA,
        )
        d = q.to_dict()
        assert d['outcome'] == 'ok'
        assert d['verdict'] == 'success'
        assert 'labels' in d['schema_match']
        assert 'relationships' in d['schema_match']
        assert 'properties' in d['schema_match']

    def test_to_dict_unknown_prop(self):
        q = assess_quality('MATCH (n:Evento) RETURN n.nombre', schema=SCHEMA)
        d = q.to_dict()
        unknown = d['schema_match']['properties']['unknown']
        assert any(u['property'] == 'nombre' for u in unknown)

    def test_to_dict_wrong_label_prop(self):
        q = assess_quality('MATCH (n:Evento) RETURN n.edad', schema=SCHEMA)
        d = q.to_dict()
        wrong = d['schema_match']['properties']['wrong_label']
        assert any(
            w['property'] == 'edad' and 'Persona' in w['exists_on']
            for w in wrong
        )

    def test_to_dict_no_schema(self):
        q = assess_quality('MATCH (n:Evento) RETURN n', schema=None)
        d = q.to_dict()
        assert d['schema_match'] is None
