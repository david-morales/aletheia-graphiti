"""Tests for AST-based Cypher element extraction."""
from __future__ import annotations

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cypher_extractor import PropertyAccess, extract_elements


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


class TestLabelExtraction:
    def test_single_label(self):
        result = extract_elements('MATCH (n:Evento) RETURN n')
        assert result.labels == ['Evento']
        assert result.var_labels == {'n': 'Evento'}

    def test_multiple_labels(self):
        result = extract_elements(
            'MATCH (p:Persona)-[:REL]->(e:Evento) RETURN p, e'
        )
        assert sorted(result.labels) == ['Evento', 'Persona']

    def test_backtick_label(self):
        result = extract_elements('MATCH (n:`Tipo Delito`) RETURN n')
        assert result.labels == ['Tipo Delito']
        assert result.var_labels == {'n': 'Tipo Delito'}

    def test_no_label(self):
        result = extract_elements('MATCH (n) RETURN n')
        assert result.labels == []

    def test_optional_match_labels(self):
        result = extract_elements(
            'MATCH (n:Evento) OPTIONAL MATCH (n)-[:REL]->(p:Persona) RETURN n, p'
        )
        assert sorted(result.labels) == ['Evento', 'Persona']

    def test_call_procedure(self):
        result = extract_elements('CALL db.labels()')
        assert result.labels == []
        assert result.parse_errors == 0


# ---------------------------------------------------------------------------
# Relationship extraction
# ---------------------------------------------------------------------------


class TestRelationshipExtraction:
    def test_single_rel_type(self):
        result = extract_elements(
            'MATCH (p:Persona)-[:INVOLUCRADO_EN]->(e:Evento) RETURN p, e'
        )
        assert result.rel_types == ['INVOLUCRADO_EN']

    def test_untyped_rel(self):
        result = extract_elements(
            'MATCH (a:A)-[*1..3]->(b:B) RETURN a, b'
        )
        assert result.rel_types == []

    def test_multiple_rel_types(self):
        result = extract_elements(
            'MATCH (p:Persona)-[:INVOLUCRADO_EN]->(e:Evento) '
            'MATCH (e)-[:OCURRIDO_EN]->(l:Lugar) '
            'RETURN p, e, l'
        )
        assert sorted(result.rel_types) == ['INVOLUCRADO_EN', 'OCURRIDO_EN']


# ---------------------------------------------------------------------------
# Relationship patterns (directional)
# ---------------------------------------------------------------------------


class TestRelPatterns:
    def test_right_direction(self):
        result = extract_elements(
            'MATCH (p:Persona)-[:R]->(e:Evento) RETURN p, e'
        )
        assert len(result.rel_patterns) == 1
        rp = result.rel_patterns[0]
        assert rp.source_var == 'p'
        assert rp.rel_type == 'R'
        assert rp.target_var == 'e'
        assert rp.direction == 'right'

    def test_left_direction(self):
        result = extract_elements(
            'MATCH (e:Evento)<-[:R]-(p:Persona) RETURN e, p'
        )
        assert len(result.rel_patterns) == 1
        rp = result.rel_patterns[0]
        assert rp.source_var == 'e'
        assert rp.rel_type == 'R'
        assert rp.target_var == 'p'
        assert rp.direction == 'left'

    def test_undirected(self):
        result = extract_elements(
            'MATCH (a:A)-[:R]-(b:B) RETURN a, b'
        )
        assert len(result.rel_patterns) == 1
        rp = result.rel_patterns[0]
        assert rp.source_var == 'a'
        assert rp.rel_type == 'R'
        assert rp.target_var == 'b'
        assert rp.direction == 'undirected'

    def test_multi_hop_chain(self):
        result = extract_elements(
            'MATCH (a:A)-[:R1]->(b:B)-[:R2]->(c:C) RETURN a, b, c'
        )
        assert len(result.rel_patterns) == 2
        r1 = result.rel_patterns[0]
        r2 = result.rel_patterns[1]
        assert r1.source_var == 'a'
        assert r1.rel_type == 'R1'
        assert r1.target_var == 'b'
        assert r2.source_var == 'b'
        assert r2.rel_type == 'R2'
        assert r2.target_var == 'c'


# ---------------------------------------------------------------------------
# Property extraction
# ---------------------------------------------------------------------------


class TestPropertyExtraction:
    def test_simple_property(self):
        result = extract_elements('MATCH (n:Evento) RETURN n.name')
        assert PropertyAccess(variable='n', property_name='name') in result.properties

    def test_multiple_properties(self):
        result = extract_elements('MATCH (n:Evento) RETURN n.name, n.date')
        assert len(result.properties) >= 2
        assert PropertyAccess(variable='n', property_name='name') in result.properties
        assert PropertyAccess(variable='n', property_name='date') in result.properties

    def test_where_property(self):
        result = extract_elements(
            "MATCH (n:Evento) WHERE n.date > '2024' RETURN n"
        )
        assert PropertyAccess(variable='n', property_name='date') in result.properties

    def test_property_in_aggregation(self):
        result = extract_elements(
            'MATCH (p:Persona)-[:R]->(s:Suceso) '
            'WITH p, count(s) AS cnt '
            'RETURN p.name, cnt'
        )
        assert PropertyAccess(variable='p', property_name='name') in result.properties

    def test_string_literal_not_captured(self):
        result = extract_elements(
            "MATCH (n:Evento) WHERE n.date > '2024-01-01' RETURN n"
        )
        # '2024-01-01' is a string literal, not a property access
        for prop in result.properties:
            assert prop.variable != "'2024-01-01'"
            assert '2024' not in prop.variable
