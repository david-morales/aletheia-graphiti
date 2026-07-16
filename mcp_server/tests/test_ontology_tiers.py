"""Tests for the ontology access tiers (v1.0.3).

Covers:
- get_ontology_documentation — full reference payload (summary + parsed
  properties + identity), tolerant of old graphs missing the new attributes.
- get_ontology_structure — FROZEN shape: Cypher and per-entry dict must not
  change (the lightweight map tier).
- explore_ontology — redesigned class-context payload:
  {center, relationships: {outgoing, incoming},
   hierarchy: {parents, children, siblings}, neighbors}.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.schema import GraphitiConfig

# ---------------------------------------------------------------------------
# Fixture data — fake OntologyClass rows as the FalkorDB driver returns them
# ---------------------------------------------------------------------------

WIDGET_SUMMARY = (
    'A Widget is the core manufactured component tracked by the system. '
    'It carries a unique serial number assigned at production time.\n\n'
    'Widgets move through a lifecycle of assembly, testing, and deployment; '
    'each stage is recorded with full provenance.'
)
GADGET_SUMMARY = (
    'A Gadget is an auxiliary device attached to widgets. '
    'It extends widget behaviour at runtime.'
)
GIZMO_SUMMARY = 'A Gizmo is an experimental component. It rarely leaves the lab.'
DOOHICKEY_SUMMARY = 'A Doohickey stores gadget telemetry. It is written continuously.'
BASE_SUMMARY = (
    'Base is the abstract root of the component hierarchy. '
    'Every concrete component inherits from it.'
)
LINKS_TO_SUMMARY = (
    'Connects a Widget to the Gadget it powers. '
    'Recorded whenever a physical attachment is observed.'
)
FEEDS_SUMMARY = 'Connects a Gadget to the Doohickey it feeds. Telemetry flows along it.'

# Contract: stored `properties` JSON entries carry FIVE keys.
WIDGET_PROPERTIES = [
    {
        'name': 'serial_number',
        'label': 'Serial Number',
        'range': 'string',
        'comment': 'Unique manufacturing serial.',
        'required': True,
    },
    {
        'name': 'weight_kg',
        'label': 'Weight (kg)',
        'range': 'float',
        'comment': 'Total weight in kilograms.',
        'required': False,
    },
]


def widget_row() -> dict:
    return {
        'uuid': 'widget-uuid',
        'name': 'Widget',
        'ontology_type': 'class',
        'summary': WIDGET_SUMMARY,
        'alt_labels': ['Artefact'],
        'inherits_from': ['Base'],
        'examples': ['Widget WX-1000'],
        'source_entity': None,
        'target_entity': None,
        'properties': json.dumps(WIDGET_PROPERTIES),
        'identity': True,
    }


def gadget_row() -> dict:
    return {
        'uuid': 'gadget-uuid',
        'name': 'Gadget',
        'ontology_type': 'class',
        'summary': GADGET_SUMMARY,
        'alt_labels': [],
        'inherits_from': ['Base'],
        'examples': [],
        'source_entity': None,
        'target_entity': None,
        'properties': None,   # attribute null on the graph
        'identity': None,
    }


def gizmo_row() -> dict:
    return {
        'uuid': 'gizmo-uuid',
        'name': 'Gizmo',
        'ontology_type': 'class',
        'summary': GIZMO_SUMMARY,
        'alt_labels': [],
        'inherits_from': ['Base'],
        'examples': [],
        'source_entity': None,
        'target_entity': None,
        'properties': None,
        'identity': None,
    }


def doohickey_row() -> dict:
    return {
        'uuid': 'doohickey-uuid',
        'name': 'Doohickey',
        'ontology_type': 'class',
        'summary': DOOHICKEY_SUMMARY,
        'alt_labels': [],
        'inherits_from': [],
        'examples': [],
        'source_entity': None,
        'target_entity': None,
        'properties': None,
        'identity': None,
    }


def base_row() -> dict:
    return {
        'uuid': 'base-uuid',
        'name': 'Base',
        'ontology_type': 'abstract_class',
        'summary': BASE_SUMMARY,
        'alt_labels': [],
        'inherits_from': [],
        'examples': [],
        'source_entity': None,
        'target_entity': None,
        'properties': None,
        'identity': None,
    }


def links_to_row() -> dict:
    return {
        'uuid': 'links-to-uuid',
        'name': 'LINKS_TO',
        'ontology_type': 'relationship_class',
        'summary': LINKS_TO_SUMMARY,
        'alt_labels': [],
        'inherits_from': [],
        'examples': [],
        'source_entity': 'Widget',
        'target_entity': 'Gadget',
        'properties': None,
        'identity': None,
    }


def feeds_row() -> dict:
    return {
        'uuid': 'feeds-uuid',
        'name': 'FEEDS',
        'ontology_type': 'relationship_class',
        'summary': FEEDS_SUMMARY,
        'alt_labels': [],
        'inherits_from': [],
        'examples': [],
        'source_entity': 'Gadget',
        'target_entity': 'Doohickey',
        'properties': None,
        'identity': None,
    }


# ---------------------------------------------------------------------------
# Object-property-style ontology fixtures — relationships live as RELATES_TO
# EDGES between OntologyClass nodes; there are ZERO relationship_class nodes.
# Edge rows are what the RELATES_TO Cypher returns: {source, name, fact, target}.
# ---------------------------------------------------------------------------

PERSONA_SUMMARY = 'A person appearing in a report. Identified by their document number.'
DETENCION_SUMMARY = 'An arrest event recorded by officers. Carries date, place, and grounds.'
LUGAR_SUMMARY = 'A place referenced by an event. Geocodable to an address.'

ES_DETENIDO_PROSE = (
    'Relationship: A person was arrested — links Persona to Detencion.'
)
ES_DETENIDO_FACT = f'Persona ES_DETENIDO Detencion: {ES_DETENIDO_PROSE}'
# A fact WITHOUT the "<source> <name> <target>: " prefix — served verbatim.
OCURRE_EN_FACT = 'Where the arrest took place.'


def persona_row() -> dict:
    return {
        'uuid': 'persona-uuid',
        'name': 'Persona',
        'ontology_type': 'class',
        'summary': PERSONA_SUMMARY,
        'alt_labels': [],
        'inherits_from': [],
        'examples': [],
        'source_entity': None,
        'target_entity': None,
        'properties': None,
        'identity': None,
    }


def detencion_row() -> dict:
    return {
        'uuid': 'detencion-uuid',
        'name': 'Detencion',
        'ontology_type': 'class',
        'summary': DETENCION_SUMMARY,
        'alt_labels': [],
        'inherits_from': [],
        'examples': [],
        'source_entity': None,
        'target_entity': None,
        'properties': None,
        'identity': None,
    }


def lugar_row() -> dict:
    return {
        'uuid': 'lugar-uuid',
        'name': 'Lugar',
        'ontology_type': 'class',
        'summary': LUGAR_SUMMARY,
        'alt_labels': [],
        'inherits_from': [],
        'examples': [],
        'source_entity': None,
        'target_entity': None,
        'properties': None,
        'identity': None,
    }


def es_detenido_edge() -> dict:
    return {
        'source': 'Persona',
        'name': 'ES_DETENIDO',
        'fact': ES_DETENIDO_FACT,
        'target': 'Detencion',
    }


def ocurre_en_edge() -> dict:
    return {
        'source': 'Detencion',
        'name': 'OCURRE_EN',
        'fact': OCURRE_EN_FACT,
        'target': 'Lugar',
    }


# ---------------------------------------------------------------------------
# Service factory — follows test_ontology_resilience.py conventions
# ---------------------------------------------------------------------------

def make_service(rows: list[dict], edges: list[dict] | None = None):
    """Fake GraphitiService whose ontology driver returns the given rows.

    `rows` answers the OntologyClass node query; `edges` (default none)
    answers the RELATES_TO edge query used by object-property ontologies.
    """
    svc = MagicMock()
    svc.config = GraphitiConfig()
    svc.config.graphiti.ontology_graph = 'test_ontology'
    svc._ensure_ontology_client = AsyncMock(return_value=True)

    edge_rows = edges or []

    async def dispatch_query(query, **kwargs):
        if 'RELATES_TO' in query:
            return (edge_rows, None, None)
        return (rows, None, None)

    client = MagicMock()
    client.driver.execute_query = AsyncMock(side_effect=dispatch_query)
    # Semantic name-resolution fallback finds nothing unless a test overrides.
    client.search_ = AsyncMock(return_value=MagicMock(nodes=[]))
    svc.ontology_client = client
    return svc


# ---------------------------------------------------------------------------
# get_ontology_documentation — the full-reference tier
# ---------------------------------------------------------------------------

class TestGetOntologyDocumentation:
    @pytest.mark.asyncio
    async def test_documentation_returns_full_payload(self):
        from graphiti_mcp_server import get_ontology_documentation

        svc = make_service([widget_row(), gadget_row(), links_to_row()])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_documentation()

        assert 'error' not in result
        assert result['ontology_graph'] == 'test_ontology'

        entities = {e['name']: e for e in result['entity_classes']}
        relationships = {r['name']: r for r in result['relationship_classes']}
        assert set(entities) == {'Widget', 'Gadget'}
        assert set(relationships) == {'LINKS_TO'}

        widget = entities['Widget']
        # Full summary — never truncated.
        assert widget['summary'] == WIDGET_SUMMARY
        # Parsed properties, passed through untouched (5-key contract).
        assert widget['properties'] == WIDGET_PROPERTIES
        for prop in widget['properties']:
            assert set(prop.keys()) == {'name', 'label', 'range', 'comment', 'required'}
        assert widget['identity'] is True
        assert widget['alt_labels'] == ['Artefact']
        assert widget['inherits_from'] == ['Base']
        assert widget['examples'] == ['Widget WX-1000']

        gadget = entities['Gadget']
        assert gadget['properties'] == []
        assert gadget['identity'] is False

        rel = relationships['LINKS_TO']
        assert rel['source_entity'] == 'Widget'
        assert rel['target_entity'] == 'Gadget'
        assert rel['summary'] == LINKS_TO_SUMMARY
        assert rel['properties'] == []
        assert rel['identity'] is False

    @pytest.mark.asyncio
    async def test_documentation_handles_missing_new_attributes(self):
        """Old graphs: rows WITHOUT properties/identity keys → safe defaults."""
        from graphiti_mcp_server import get_ontology_documentation

        old_widget = widget_row()
        del old_widget['properties']
        del old_widget['identity']
        old_rel = links_to_row()
        del old_rel['properties']
        del old_rel['identity']
        # And a row whose properties attribute holds malformed JSON.
        bad_gadget = gadget_row()
        bad_gadget['properties'] = 'not-json{'

        svc = make_service([old_widget, bad_gadget, old_rel])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_documentation()

        assert 'error' not in result
        for entry in result['entity_classes'] + result['relationship_classes']:
            assert entry['properties'] == []
            assert entry['identity'] is False

    @pytest.mark.asyncio
    async def test_documentation_error_when_no_ontology(self):
        from graphiti_mcp_server import get_ontology_documentation

        svc = make_service([])
        svc._ensure_ontology_client = AsyncMock(return_value=False)

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_documentation()

        assert 'error' in result
        assert 'No ontology graph configured' in result['error']


# ---------------------------------------------------------------------------
# get_ontology_structure — FROZEN shape (the map tier)
# ---------------------------------------------------------------------------

# The exact Cypher get_ontology_structure has always issued. Fattening the
# projection (e.g. with properties/identity) must fail this test loudly.
FROZEN_STRUCTURE_QUERY = (
    'MATCH (n:OntologyClass) '
    'RETURN n.name AS name, '
    'n.ontology_type AS ontology_type, '
    'n.inherits_from AS inherits_from, '
    'n.summary AS summary, '
    'n.alt_labels AS alt_labels, '
    'n.source_entity AS source_entity, '
    'n.target_entity AS target_entity, '
    'n.examples AS examples'
)

FROZEN_ENTITY_KEYS = {'name', 'ontology_type', 'summary', 'alt_labels', 'inherits_from', 'examples'}
FROZEN_RELATIONSHIP_KEYS = FROZEN_ENTITY_KEYS | {'source_entity', 'target_entity'}


class TestStructureShapeFrozen:
    @pytest.mark.asyncio
    async def test_structure_shape_frozen(self):
        from graphiti_mcp_server import get_ontology_structure

        # Rows deliberately CARRY the new attributes — the structure tool
        # must not leak them into its entries.
        svc = make_service([widget_row(), gadget_row(), links_to_row()])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_structure()

        assert 'error' not in result

        issued_query = svc.ontology_client.driver.execute_query.call_args[0][0]
        assert issued_query == FROZEN_STRUCTURE_QUERY

        assert len(result['entity_classes']) == 2
        assert len(result['relationship_classes']) == 1
        for entry in result['entity_classes']:
            assert set(entry.keys()) == FROZEN_ENTITY_KEYS
        for entry in result['relationship_classes']:
            assert set(entry.keys()) == FROZEN_RELATIONSHIP_KEYS


# ---------------------------------------------------------------------------
# explore_ontology — the class-context tier
# ---------------------------------------------------------------------------

class TestExploreOntologyClassContext:
    @pytest.mark.asyncio
    async def test_explore_center_full_and_neighbors_trimmed(self):
        from graphiti_mcp_server import explore_ontology

        svc = make_service([widget_row(), gadget_row(), links_to_row()])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Widget')

        assert 'error' not in result

        center = result['center']
        assert center['name'] == 'Widget'
        assert center['summary'] == WIDGET_SUMMARY  # full text, not trimmed
        assert center['properties'] == WIDGET_PROPERTIES
        assert center['identity'] is True

        assert result['relationships']['outgoing'] == [
            {'name': 'LINKS_TO', 'target': 'Gadget', 'summary': LINKS_TO_SUMMARY}
        ]
        assert result['relationships']['incoming'] == []

        assert result['neighbors'] == [
            {
                'name': 'Gadget',
                'summary_line': 'A Gadget is an auxiliary device attached to widgets.',
                'via': 'LINKS_TO',
            }
        ]

    @pytest.mark.asyncio
    async def test_explore_incoming_relationships(self):
        from graphiti_mcp_server import explore_ontology

        svc = make_service([widget_row(), gadget_row(), links_to_row()])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Gadget')

        assert 'error' not in result
        assert result['relationships']['incoming'] == [
            {'name': 'LINKS_TO', 'source': 'Widget', 'summary': LINKS_TO_SUMMARY}
        ]
        assert result['relationships']['outgoing'] == []
        assert result['neighbors'] == [
            {
                'name': 'Widget',
                'summary_line': 'A Widget is the core manufactured component tracked by the system.',
                'via': 'LINKS_TO',
            }
        ]

    @pytest.mark.asyncio
    async def test_explore_hierarchy_parents_children_siblings(self):
        from graphiti_mcp_server import explore_ontology

        rows = [base_row(), widget_row(), gadget_row(), gizmo_row()]

        # Exploring Widget: parent Base, siblings Gadget + Gizmo.
        svc = make_service(rows)
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Widget')

        assert 'error' not in result
        hierarchy = result['hierarchy']
        assert hierarchy['parents'] == [
            {
                'name': 'Base',
                'summary_line': 'Base is the abstract root of the component hierarchy.',
            }
        ]
        assert [s['name'] for s in hierarchy['siblings']] == ['Gadget', 'Gizmo']
        for sibling in hierarchy['siblings']:
            assert set(sibling.keys()) == {'name', 'summary_line'}
        assert hierarchy['children'] == []

        # Exploring Base: children are every class inheriting from it.
        svc = make_service(rows)
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Base')

        assert 'error' not in result
        assert [c['name'] for c in result['hierarchy']['children']] == [
            'Widget',
            'Gadget',
            'Gizmo',
        ]
        assert result['hierarchy']['parents'] == []

        # Siblings respect the limit.
        svc = make_service(rows)
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Widget', limit=1)

        assert 'error' not in result
        assert len(result['hierarchy']['siblings']) == 1

    @pytest.mark.asyncio
    async def test_explore_depth_two_hop_names_only(self):
        from graphiti_mcp_server import explore_ontology

        rows = [widget_row(), gadget_row(), doohickey_row(), links_to_row(), feeds_row()]

        # depth=2 (default): the second hop appends name-only entries.
        svc = make_service(rows)
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Widget', depth=2)

        assert 'error' not in result
        assert result['neighbors'] == [
            {
                'name': 'Gadget',
                'summary_line': 'A Gadget is an auxiliary device attached to widgets.',
                'via': 'LINKS_TO',
            },
            {'name': 'Doohickey', 'via': 'FEEDS'},
        ]

        # depth=1: direct neighbors only.
        svc = make_service(rows)
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Widget', depth=1)

        assert 'error' not in result
        assert [n['name'] for n in result['neighbors']] == ['Gadget']

    @pytest.mark.asyncio
    async def test_explore_unknown_class_error_contract(self):
        from graphiti_mcp_server import explore_ontology

        svc = make_service([widget_row()])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Nonexistent')

        assert 'error' in result
        assert 'Nonexistent' in result['error']

    @pytest.mark.asyncio
    async def test_explore_resolves_by_uuid(self):
        from graphiti_mcp_server import explore_ontology

        svc = make_service([widget_row(), gadget_row(), links_to_row()])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_uuid='widget-uuid')

        assert 'error' not in result
        assert result['center']['name'] == 'Widget'

    @pytest.mark.asyncio
    async def test_explore_requires_name_or_uuid(self):
        from graphiti_mcp_server import explore_ontology

        svc = make_service([widget_row()])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology()

        assert 'error' in result
        assert 'node_name or node_uuid' in result['error']


# ---------------------------------------------------------------------------
# Object-property ontologies — relationships derived from RELATES_TO edges
# ---------------------------------------------------------------------------

class TestEdgeDerivedRelationships:
    """Ontology families that model relationships as owl:ObjectProperty store
    them as RELATES_TO edges between OntologyClass nodes, with ZERO
    relationship_class nodes. The read tiers must derive relationship entries
    from those edges."""

    @pytest.mark.asyncio
    async def test_documentation_derives_relationships_from_edges(self):
        from graphiti_mcp_server import get_ontology_documentation

        svc = make_service(
            rows=[persona_row(), detencion_row(), lugar_row()],
            edges=[es_detenido_edge(), ocurre_en_edge()],
        )

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_documentation()

        assert 'error' not in result
        assert {e['name'] for e in result['entity_classes']} == {
            'Persona',
            'Detencion',
            'Lugar',
        }

        relationships = {r['name']: r for r in result['relationship_classes']}
        assert set(relationships) == {'ES_DETENIDO', 'OCURRE_EN'}

        es_detenido = relationships['ES_DETENIDO']
        assert es_detenido['source_entity'] == 'Persona'
        assert es_detenido['target_entity'] == 'Detencion'
        # The "<source> <name> <target>: " prefix is stripped from the fact.
        assert es_detenido['summary'] == ES_DETENIDO_PROSE
        # Shape consistency with node-derived entries.
        assert es_detenido['properties'] == []
        assert es_detenido['identity'] is False
        assert es_detenido['alt_labels'] == []
        assert es_detenido['inherits_from'] == []
        assert es_detenido['examples'] == []

        # A fact without the prefix is served verbatim.
        assert relationships['OCURRE_EN']['summary'] == OCURRE_EN_FACT

    @pytest.mark.asyncio
    async def test_explore_uses_edge_derived_relationships(self):
        from graphiti_mcp_server import explore_ontology

        rows = [persona_row(), detencion_row(), lugar_row()]
        edges = [es_detenido_edge(), ocurre_en_edge()]

        # Outgoing + neighbors from edges; depth 2 adds the name-only hop.
        svc = make_service(rows, edges)
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Persona')

        assert 'error' not in result
        assert result['relationships']['outgoing'] == [
            {'name': 'ES_DETENIDO', 'target': 'Detencion', 'summary': ES_DETENIDO_PROSE}
        ]
        assert result['relationships']['incoming'] == []
        assert result['neighbors'] == [
            {
                'name': 'Detencion',
                'summary_line': 'An arrest event recorded by officers.',
                'via': 'ES_DETENIDO',
            },
            {'name': 'Lugar', 'via': 'OCURRE_EN'},
        ]

        # Incoming side.
        svc = make_service(rows, edges)
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Detencion')

        assert 'error' not in result
        assert result['relationships']['incoming'] == [
            {'name': 'ES_DETENIDO', 'source': 'Persona', 'summary': ES_DETENIDO_PROSE}
        ]
        assert result['relationships']['outgoing'] == [
            {'name': 'OCURRE_EN', 'target': 'Lugar', 'summary': OCURRE_EN_FACT}
        ]

    @pytest.mark.asyncio
    async def test_mixed_reified_and_edge_relationships_node_wins(self):
        """Same (name, source, target) as both a reified relationship_class
        node and a RELATES_TO edge → the node-derived entry wins (richer)."""
        from graphiti_mcp_server import explore_ontology, get_ontology_documentation

        rows = [widget_row(), gadget_row(), links_to_row()]
        duplicate_edge = {
            'source': 'Widget',
            'name': 'LINKS_TO',
            'fact': 'Widget LINKS_TO Gadget: short edge fact.',
            'target': 'Gadget',
        }

        svc = make_service(rows, [duplicate_edge])
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_documentation()

        assert 'error' not in result
        links = [r for r in result['relationship_classes'] if r['name'] == 'LINKS_TO']
        assert len(links) == 1
        assert links[0]['summary'] == LINKS_TO_SUMMARY  # node-derived prose

        svc = make_service(rows, [duplicate_edge])
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Widget')

        assert 'error' not in result
        assert result['relationships']['outgoing'] == [
            {'name': 'LINKS_TO', 'target': 'Gadget', 'summary': LINKS_TO_SUMMARY}
        ]
        assert [n['name'] for n in result['neighbors']] == ['Gadget']


# ---------------------------------------------------------------------------
# Registration — the new tool is registered next to get_ontology_structure
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_get_ontology_documentation_registered(self):
        from domain_profile import DomainProfile, EntityTypeInfo
        from graphiti_mcp_server import mcp, register_dynamic_tools

        profile = DomainProfile(
            group_id='test_graph',
            entity_types={
                'Widget': EntityTypeInfo('Widget', 3, 'Test widget', ['WX-1000']),
            },
            edge_types={},
            time_range=None,
        )

        register_dynamic_tools(profile)

        tools = mcp._tool_manager._tools
        assert 'get_ontology_documentation' in tools
        assert 'get_ontology_structure' in tools
