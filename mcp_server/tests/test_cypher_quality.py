"""Tests for schema-aware Cypher quality assessment."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cypher_quality import (
    UnknownProp,
    WrongLabelProp,
    assess_quality,
    compute_result_signals,
    refine_verdict,
    ResultSignals,
)

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
# Nested-attribute backends (BLK-1)
# ---------------------------------------------------------------------------

# A schema as get_schema builds it on a backend that NESTS its domain fields —
# Apache AGE. `properties` is the honest TOP-LEVEL key set (Graphiti's own
# bookkeeping columns plus the `attributes` container itself); the domain fields
# live one level down and are announced in `attribute_keys`. The backend names
# the container in `attribute_container` — dialect as DATA (ADR-019 R6), so the
# validator stays dialect-blind and no backend name is hardcoded here.
NESTED_SCHEMA = {
    'dialect': 'age-opencypher',
    'attribute_container': 'attributes',
    'node_labels': {
        'Persona': {
            'count': 500,
            'properties': ['attributes', 'created_at', 'group_id', 'labels', 'name', 'uuid'],
            'attribute_keys': ['documento', 'edad'],
        },
        'Ubicacion': {
            'count': 30,
            'properties': ['attributes', 'created_at', 'group_id', 'labels', 'name', 'uuid'],
            'attribute_keys': ['municipio'],
        },
    },
    'relationship_types': {
        'VIVE_EN': {'count': 40, 'patterns': [['Persona', 'Ubicacion']]},
    },
}


class TestNestedAttributeValidation:
    """BLK-1: the query form the AGE dialect_reference TEACHES must assess CLEAN.

    Measured before this guard: `n.attributes.edad` — the exact nested-path
    access `dialect_reference` documents as the way to read a domain field on
    AGE — came back `outcome: suspect / verdict: schema_mismatch` on 500
    correct rows, because `_validate_properties` diffed against `properties`
    only. `properties` is top-level, so both halves of the path (`attributes`
    and `edad`) were unknown, and `refine_verdict` never downgrades
    `schema_mismatch` — so the whole result was flagged for the life of the call.
    """

    def test_nested_path_access_is_clean(self):
        q = assess_quality(
            'MATCH (n:Persona) WHERE n.attributes.edad > 30 '
            'RETURN n.name AS name, n.attributes.edad AS edad',
            schema=NESTED_SCHEMA,
        )
        assert q.schema_match.properties.unknown == [], q.schema_match.properties.unknown
        assert q.schema_match.properties.wrong_label == []
        assert q.verdict == 'success'
        assert q.outcome == 'ok'

    def test_the_container_alone_is_clean(self):
        """`RETURN n.attributes` returns the whole map — valid, not a domain field."""
        q = assess_quality('MATCH (n:Persona) RETURN n.attributes', schema=NESTED_SCHEMA)
        assert q.schema_match.properties.unknown == []
        assert q.verdict == 'success'

    def test_the_container_is_valid_even_when_the_probe_never_sampled_it(self):
        """The container is a BACKEND fact, not a sampling outcome.

        A label whose top-level probe degrades comes back `properties: []` /
        `sampled: false` — get_schema reports one entry unsampled rather than
        failing the whole call. If the container's validity were read off
        `properties`, that degraded entry would flag every correct nested query
        against it. It is read off `attribute_container` instead, which the
        flavour announces unconditionally.
        """
        schema = {
            'attribute_container': 'attributes',
            'node_labels': {
                'Persona': {'count': 5, 'properties': [], 'attribute_keys': ['edad'],
                            'sampled': False},
            },
            'relationship_types': {},
        }
        q = assess_quality('MATCH (n:Persona) RETURN n.attributes.edad', schema=schema)
        assert q.schema_match.properties.unknown == [], q.schema_match.properties.unknown
        assert q.verdict == 'success'

    def test_a_genuinely_wrong_nested_field_is_still_flagged(self):
        """The fix must not blanket-accept everything under the container."""
        q = assess_quality(
            'MATCH (n:Persona) RETURN n.attributes.no_such_field',
            schema=NESTED_SCHEMA,
        )
        unknown = q.schema_match.properties.unknown
        assert any(
            u.property_name == 'no_such_field' and u.on_label == 'Persona' for u in unknown
        ), unknown
        assert q.verdict == 'schema_mismatch'

    def test_a_nested_field_on_the_wrong_label_is_flagged(self):
        """`municipio` is a Ubicacion attribute — naming it on Persona is a mismatch."""
        q = assess_quality(
            'MATCH (n:Persona) RETURN n.attributes.municipio',
            schema=NESTED_SCHEMA,
        )
        wrong = q.schema_match.properties.wrong_label
        assert any(
            w.property_name == 'municipio'
            and w.on_label == 'Persona'
            and 'Ubicacion' in w.exists_on
            for w in wrong
        ), wrong
        assert q.verdict == 'schema_mismatch'

    def test_top_level_bookkeeping_still_validates(self):
        q = assess_quality(
            'MATCH (n:Persona) RETURN n.uuid, n.name, n.created_at',
            schema=NESTED_SCHEMA,
        )
        assert q.schema_match.properties.unknown == []
        assert q.verdict == 'success'

    @pytest.mark.parametrize(
        'query,expected,note',
        [
            (
                'MATCH (n:Persona) RETURN n.attributes.edad',
                'success',
                'the form the dialect teaches — clean',
            ),
            (
                'MATCH (n:Persona) RETURN n.attributes.no_such_field',
                'schema_mismatch',
                'nested, but no such attribute',
            ),
            (
                'MATCH (n:Persona) RETURN n.edad',
                'schema_mismatch',
                'F4: right key, WRONG PATH — addresses nothing on this backend',
            ),
        ],
    )
    def test_the_access_PATH_is_checked_not_just_the_key(self, query, expected, note):
        """F4: unioning the two key sets made the validator container-BLIND.

        `edad` lives only inside the container on this backend, so `n.edad`
        returns null for every row — a silently empty column, which is the
        failure mode `cypher_quality` exists to catch. Before the union it was
        correctly flagged; the union accepted it. The schema carries enough to
        tell the two apart: a key in `attribute_keys` but NOT in top-level
        `properties` is reachable ONLY through the announced container.
        """
        q = assess_quality(query, schema=NESTED_SCHEMA)
        assert q.verdict == expected, (note, q.to_dict())

    def test_the_bare_access_is_reported_against_the_right_label(self):
        q = assess_quality('MATCH (n:Persona) RETURN n.edad', schema=NESTED_SCHEMA)
        assert any(
            u.property_name == 'edad' and u.on_label == 'Persona'
            for u in q.schema_match.properties.unknown
        ), q.to_dict()

    def test_a_top_level_key_is_reachable_without_the_container(self):
        """The rule is about keys that ONLY exist nested. `name` is top-level on
        every backend, so `n.name` stays clean — an over-broad path rule would
        have flagged it."""
        q = assess_quality('MATCH (n:Persona) RETURN n.name', schema=NESTED_SCHEMA)
        assert q.verdict == 'success', q.to_dict()

    def test_a_flat_backend_is_unaffected_by_the_path_rule(self):
        """F4 no-change guard: with no container announced, nothing is
        path-restricted and FalkorDB's verdicts are exactly what they were."""
        for query in (
            'MATCH (n:Persona) RETURN n.edad',
            'MATCH (n:Persona) RETURN n.name, n.summary',
        ):
            assert assess_quality(query, schema=SCHEMA).verdict == 'success', query

    def test_an_UNSAMPLED_label_does_not_accuse_a_nested_access(self):
        """F8: a probe that could not answer must not become an accusation.

        get_schema degrades ONE label rather than failing the whole call, and
        says so with `sampled: False`; `attribute_keys` comes back empty by the
        same route (AgeFlavour.attribute_keys swallows its exception). With no
        evidence about this label's attributes, a container-nested access gets
        no field-level verdict — the container name itself is still checked.
        """
        for entry in (
            {'count': 5, 'properties': [], 'attribute_keys': ['edad'], 'sampled': False},
            {'count': 5, 'properties': ['attributes'], 'attribute_keys': [], 'sampled': True},
        ):
            schema = {
                'attribute_container': 'attributes',
                'node_labels': {'Persona': entry},
                'relationship_types': {},
            }
            q = assess_quality(
                'MATCH (n:Persona) RETURN n.attributes.edad', schema=schema
            )
            assert q.verdict == 'success', (entry, q.to_dict())

    def test_the_SAMPLED_path_is_not_weakened_by_the_unsampled_allowance(self):
        """F8 must not become a blanket amnesty: a label that WAS sampled and
        does have attributes still catches a hallucinated one."""
        q = assess_quality(
            'MATCH (n:Persona) RETURN n.attributes.no_such_field', schema=NESTED_SCHEMA
        )
        assert q.verdict == 'schema_mismatch', q.to_dict()

    def test_a_flat_backend_does_not_inherit_the_nested_allowance(self):
        """No `attribute_container` announced -> the nested form is a real mismatch.

        On FalkorDB `n.attributes.edad` addresses nothing: there is no container
        and `edad` is a top-level property. Accepting it there would trade one
        silent wrong answer for another.
        """
        q = assess_quality('MATCH (n:Persona) RETURN n.attributes.edad', schema=SCHEMA)
        unknown = {u.property_name for u in q.schema_match.properties.unknown}
        assert 'attributes' in unknown, unknown
        assert q.verdict == 'schema_mismatch'


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


# ---------------------------------------------------------------------------
# Result signals (post-execution)
# ---------------------------------------------------------------------------


class TestResultSignals:
    def test_normal_result(self):
        records = [{'name': 'Alice', 'date': '2024-01-01'}, {'name': 'Bob', 'date': '2024-02-01'}]
        header = ['name', 'date']
        rs = compute_result_signals(records, header, truncated=False)
        assert rs.row_count == 2
        assert rs.null_ratio == 0.0
        assert rs.truncated is False

    def test_empty_result(self):
        rs = compute_result_signals([], [], truncated=False)
        assert rs.row_count == 0
        assert rs.null_ratio == 0.0

    def test_null_ratio(self):
        records = [
            {'name': 'Alice', 'date': None},
            {'name': None, 'date': None},
        ]
        header = ['name', 'date']
        rs = compute_result_signals(records, header, truncated=False)
        # 3 nulls out of 4 cells = 0.75
        assert rs.null_ratio == pytest.approx(0.75)

    def test_truncated(self):
        records = [{'name': 'Alice'}]
        header = ['name']
        rs = compute_result_signals(records, header, truncated=True)
        assert rs.truncated is True


# ---------------------------------------------------------------------------
# Verdict refinement (post-execution)
# ---------------------------------------------------------------------------


class TestRefineVerdict:
    def test_schema_mismatch_preserved(self):
        q = assess_quality('MATCH (n:Accidente) RETURN n', schema=SCHEMA)
        assert q.verdict == 'schema_mismatch'
        q.result_signals = compute_result_signals([], [], truncated=False)
        refined = refine_verdict(q)
        assert refined.verdict == 'schema_mismatch'

    def test_parse_failed_preserved_on_empty(self):
        """parse_failed stays when execution also returned 0 rows."""
        q = assess_quality('MATCH (n:Evento RETURN n', schema=SCHEMA)
        assert q.verdict == 'parse_failed'
        q.result_signals = compute_result_signals([], [], truncated=False)
        refined = refine_verdict(q)
        assert refined.verdict == 'parse_failed'

    def test_parse_failed_downgraded_on_success(self):
        """parse_failed downgrades to success when query actually returned rows.

        Our ANTLR parser doesn't cover all valid Cypher (e.g. consecutive
        WITH clauses). A parse error with successful execution is a false
        positive in the parser, not a query problem.
        """
        q = assess_quality('MATCH (n:Evento RETURN n', schema=SCHEMA)
        assert q.verdict == 'parse_failed'
        q.result_signals = compute_result_signals(
            [{'name': 'Alice'}, {'name': 'Bob'}], ['name'], truncated=False,
        )
        refined = refine_verdict(q)
        assert refined.verdict == 'success'
        assert refined.outcome == 'ok'

    def test_empty_with_clean_schema(self):
        q = assess_quality('MATCH (n:Evento) RETURN n.name', schema=SCHEMA)
        q.result_signals = compute_result_signals([], [], truncated=False)
        refined = refine_verdict(q)
        assert refined.verdict == 'empty_legit'
        assert refined.outcome == 'ok'

    def test_high_null_ratio(self):
        records = [
            {'name': None, 'date': None},
            {'name': None, 'date': None},
        ]
        q = assess_quality('MATCH (n:Evento) RETURN n.name', schema=SCHEMA)
        q.result_signals = compute_result_signals(records, ['name', 'date'], truncated=False)
        refined = refine_verdict(q)
        assert refined.verdict == 'degraded'
        assert refined.outcome == 'suspect'

    def test_no_signals_unchanged(self):
        q = assess_quality('MATCH (n:Evento) RETURN n.name', schema=SCHEMA)
        assert q.result_signals is None
        refined = refine_verdict(q)
        assert refined.verdict == 'success'
        assert refined.outcome == 'ok'

    def test_normal_result_unchanged(self):
        records = [
            {'name': 'Alice', 'date': '2024-01-01'},
            {'name': 'Bob', 'date': '2024-02-01'},
        ]
        q = assess_quality('MATCH (n:Evento) RETURN n.name', schema=SCHEMA)
        q.result_signals = compute_result_signals(records, ['name', 'date'], truncated=False)
        refined = refine_verdict(q)
        assert refined.verdict == 'success'
        assert refined.outcome == 'ok'
