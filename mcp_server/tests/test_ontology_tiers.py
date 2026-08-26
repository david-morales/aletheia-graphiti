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
from flavours.age import AgeFlavour
from flavours.falkordb import FalkorDbFlavour

# These are served-payload guards — the FROZEN `get_ontology_structure` shape,
# `explore_ontology`'s class-context envelope and its not-found taxonomy, plus
# which of the four tools this arm registers at all. All ADR-015/019 surface,
# all on stubs: no database, no API key. Nothing in CI selected them.
pytestmark = pytest.mark.contract

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


def subclass_of_edge() -> dict:
    # Builders store class hierarchy as RELATES_TO edges named SUBCLASS_OF.
    # Hierarchy is served via inherits_from — these edges must never surface
    # as relationship entries.
    return {
        'source': 'Persona',
        'name': 'SUBCLASS_OF',
        'fact': 'Persona SUBCLASS_OF Base: Persona inherits from Base.',
        'target': 'Base',
    }


# ---------------------------------------------------------------------------
# Service factory — follows test_ontology_resilience.py conventions
# ---------------------------------------------------------------------------

def make_service(rows: list[dict], edges: list[dict] | None = None):
    """Fake GraphitiService whose ontology driver returns the given rows.

    `rows` answers the OntologyClass node query; `edges` (default none)
    answers the RELATES_TO edge query used by object-property ontologies.

    Carries a REAL FalkorDbFlavour: the ontology query texts are flavour-owned,
    so a MagicMock flavour would hand the driver a MagicMock instead of Cypher
    and every dispatch below would silently fall through to the node rows.
    """
    svc = MagicMock()
    svc.config = GraphitiConfig()
    svc.config.graphiti.ontology_graph = 'test_ontology'
    svc._ensure_ontology_client = AsyncMock(return_value=True)
    svc.flavour = FalkorDbFlavour()

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
# AGE-shaped ontology storage — the Task 1 spike's DIVERGENT arm
# ---------------------------------------------------------------------------
#
# Two storage differences, both proven on the live bench graph:
#
#  (a) NESTED ATTRIBUTES. Only uuid/name/summary/group_id/labels/created_at are
#      top-level node properties on AGE; every descriptive ontology field lives in
#      an `attributes` agtype map. `n.ontology_type` is NULL on 30/30 classes while
#      `n.attributes.ontology_type` is populated on 30/30.
#
#  (b) TYPED EDGES. AGE materializes the relationship NAME as the edge LABEL
#      (graphiti_core age_graph_operations._edge_label), so the bench ontology
#      graph has ZERO RELATES_TO edges — `MATCH (a:OntologyClass)-[r:RELATES_TO]->
#      (b:OntologyClass)` returned 0 while `MATCH ()-[r]->()` returned 918.
#
# Shipped result on AGE: every structured field empty, relationship_classes [].

AGE_VEHICULO_PROPERTIES = [
    {
        'name': 'matricula',
        'label': 'Matrícula',
        'range': 'string',
        'comment': 'Plate number.',
        'required': True,
    },
]

VEHICULO_SUMMARY = 'A vehicle involved in an intervention. Identified by its plate.'


def age_vehiculo_node() -> dict:
    """One AGE OntologyClass vertex AS STORED: top-level + nested `attributes`."""
    return {
        'uuid': 'vehiculo-uuid',
        'name': 'Vehiculo',
        'summary': VEHICULO_SUMMARY,
        'attributes': {
            'ontology_type': 'class',
            'inherits_from': ['PhysicalObject'],
            'alt_labels': ['Vehículo', 'Coche'],
            'examples': ['Seat León 1234-ABC'],
            'identity': True,
            'properties': json.dumps(AGE_VEHICULO_PROPERTIES),
            'source_entity': None,
            'target_entity': None,
        },
    }


def age_physical_object_node() -> dict:
    return {
        'uuid': 'physicalobject-uuid',
        'name': 'PhysicalObject',
        'summary': 'An abstract physical thing. Root of the object hierarchy.',
        'attributes': {
            'ontology_type': 'abstract_class',
            'inherits_from': [],
            'alt_labels': [],
            'examples': [],
            'identity': False,
            'properties': None,
            'source_entity': None,
            'target_entity': None,
        },
    }


def age_ubicacion_node() -> dict:
    return {
        'uuid': 'ubicacion-uuid',
        'name': 'Ubicacion',
        'summary': 'A place referenced by an event. Geocodable to an address.',
        'attributes': {
            'ontology_type': 'class',
            'inherits_from': [],
            'alt_labels': [],
            'examples': [],
            'identity': False,
            'properties': None,
            'source_entity': None,
            'target_entity': None,
        },
    }


def age_municion_node() -> dict:
    return {
        'uuid': 'municion-uuid',
        'name': 'Municion',
        'summary': 'Ammunition seized during an intervention. Counted by calibre.',
        'attributes': {
            'ontology_type': 'class',
            'inherits_from': [],
            'alt_labels': [],
            'examples': [],
            'identity': False,
            'properties': None,
            'source_entity': None,
            'target_entity': None,
        },
    }


OCURRE_EN_PROSE = 'Where the intervention involving the vehicle took place.'
OCURRE_EN_AGE_FACT = f'Vehiculo OCURRE_EN Ubicacion: {OCURRE_EN_PROSE}'

# An ACCENTED relation name. `_edge_label`'s `_IDENT_RE` is ASCII-only
# (`^[A-Za-z_][A-Za-z0-9_]*$`) and the aletheia ontology loader does NO accent
# folding — rdfs:label "Involucra Munición" becomes INVOLUCRA_MUNICIÓN, which
# fails the regex — so AGE stores this edge under the RELATES_TO FALLBACK label
# while `r.name` keeps the true name. `type(r)` reports the fallback; `r.name`
# does not. This fixture is what makes the difference observable.
INVOLUCRA_MUNICION_NAME = 'INVOLUCRA_MUNICIÓN'
INVOLUCRA_MUNICION_PROSE = 'Ammunition found in the vehicle.'
INVOLUCRA_MUNICION_FACT = (
    f'Vehiculo {INVOLUCRA_MUNICION_NAME} Municion: {INVOLUCRA_MUNICION_PROSE}'
)


def age_typed_edges() -> list[dict]:
    """AGE edges AS STORED.

    `label` is the AGE edge LABEL (`_edge_label`: the relation name when
    identifier-safe, else the RELATES_TO fallback); `name` and `fact` are TOP-LEVEL
    edge properties written by both AGE edge write paths
    (age_graph_operations.py edge_save + the bulk field writer). Only custom
    `attributes` nest.
    """
    return [
        {
            'label': 'OCURRE_EN',
            'name': 'OCURRE_EN',
            'source': 'Vehiculo',
            'fact': OCURRE_EN_AGE_FACT,
            'target': 'Ubicacion',
        },
        {
            # The accent case: LABEL degraded to the fallback, `name` intact.
            'label': 'RELATES_TO',
            'name': INVOLUCRA_MUNICION_NAME,
            'source': 'Vehiculo',
            'fact': INVOLUCRA_MUNICION_FACT,
            'target': 'Municion',
        },
        {
            # Hierarchy edge — present in the row set on BOTH flavours, dropped by
            # the SHARED downstream filter, never by a flavour-side WHERE.
            'label': 'SUBCLASS_OF',
            'name': 'SUBCLASS_OF',
            'source': 'Vehiculo',
            'fact': 'Vehiculo SUBCLASS_OF PhysicalObject: Vehiculo inherits from it.',
            'target': 'PhysicalObject',
        },
    ]


class _AgeOntologyDriver:
    """DISCRIMINATING AGE-shaped ontology driver.

    Answers a TOP-LEVEL projection the way the live AGE graph answered the shipped
    FalkorDB-shaped query — nulls for every attribute-backed column — and a NESTED
    projection with the stored values. So a server that issues the wrong flavour's
    class query does not merely miss an assertion: it reproduces the shipped bug's
    payload exactly, and the population assertions fail loudly.

    Likewise for relationships: a `[r:RELATES_TO]` match answers only with edges
    actually stored under that LABEL, and the relation name comes from whichever
    source the query asked for — `type(r)` (the label) or `r.name` (the property).
    """

    # Columns the base/FalkorDB query reads top-level and AGE stores nested.
    _NESTED_COLUMNS = (
        'ontology_type',
        'inherits_from',
        'alt_labels',
        'examples',
        'source_entity',
        'target_entity',
        'properties',
        'identity',
    )

    def __init__(self, nodes: list[dict], edges: list[dict] | None = None):
        self.nodes = nodes
        self.edges = edges if edges is not None else []
        self.queries: list[str] = []

    def _project_class(self, node: dict, query: str) -> dict:
        row = {
            'uuid': node['uuid'],
            'name': node['name'],
            'summary': node['summary'],
        }
        attrs = node.get('attributes') or {}
        for col in self._NESTED_COLUMNS:
            # Keyed PER COLUMN, not off one sentinel: a partial regression that
            # reverts a single column to the top-level form must show up in the
            # PAYLOAD, not only in the query-text pins.
            # The whole bug in one line: a top-level read of a nested field is
            # NULL on AGE, it is not an error.
            nested = f'n.attributes.{col} AS {col}' in query
            row[col] = attrs.get(col) if nested else None
        return row

    async def execute_query(self, query: str, *a, **k):
        self.queries.append(query)

        if 'OntologyClass)-[' in query:
            # A type-constrained match reaches only edges stored under that LABEL.
            edges = (
                [e for e in self.edges if e['label'] == 'RELATES_TO']
                if '[r:RELATES_TO]' in query
                else self.edges
            )
            # `r.name AS name` reads the stored property (always the true relation
            # name); `type(r) AS name` reads the LABEL, which is the RELATES_TO
            # fallback whenever the name is not ASCII-identifier-safe.
            from_property = 'r.name AS name' in query
            rows = [
                {
                    'source': e['source'],
                    'name': e['name'] if from_property else e['label'],
                    'fact': e['fact'],
                    'target': e['target'],
                }
                for e in edges
            ]
            return rows, None, None

        if 'MATCH (n:OntologyClass)' in query:
            return [self._project_class(n, query) for n in self.nodes], None, None

        return [], None, None


def make_age_service(nodes: list[dict], edges: list[dict] | None = None):
    """Fake GraphitiService on the AGE flavour, backed by _AgeOntologyDriver."""
    svc = MagicMock()
    svc.config = GraphitiConfig()
    svc.config.graphiti.ontology_graph = 'test_ontology_age'
    svc._ensure_ontology_client = AsyncMock(return_value=True)
    svc.flavour = AgeFlavour()

    client = MagicMock()
    client.driver = _AgeOntologyDriver(nodes, edges)
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
    async def test_explore_unknown_class_answers_rather_than_fails(self):
        """A configured, reachable ontology that holds no such class has ANSWERED.

        This used to assert the opposite — `'error' in result` — and that filing
        made a consumer applying ADR-015 R4 count a not-found as a tool FAILURE
        (audit M11 addendum, 2026-08-15). `explore_entity` and `search` have
        always used `message` for the identical situation. Full taxonomy guard:
        `test_ontology_not_found_taxonomy.py`.
        """
        from graphiti_mcp_server import explore_ontology

        svc = make_service([widget_row()])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Nonexistent')

        assert not result.get('error')
        assert 'Nonexistent' in result['message']

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
    async def test_subclass_of_edges_excluded_from_relationships(self):
        """SUBCLASS_OF edges duplicate the inherits_from hierarchy — they must
        appear in NEITHER documentation relationship_classes NOR explore
        relationships/neighbors, while ES_-style edges still do."""
        from graphiti_mcp_server import explore_ontology, get_ontology_documentation

        rows = [persona_row(), detencion_row(), lugar_row(), base_row()]
        edges = [es_detenido_edge(), ocurre_en_edge(), subclass_of_edge()]

        svc = make_service(rows, edges)
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_documentation()

        assert 'error' not in result
        assert {r['name'] for r in result['relationship_classes']} == {
            'ES_DETENIDO',
            'OCURRE_EN',
        }

        svc = make_service(rows, edges)
        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Persona')

        assert 'error' not in result
        assert [r['name'] for r in result['relationships']['outgoing']] == ['ES_DETENIDO']
        assert result['relationships']['incoming'] == []
        # Base is only reachable via the SUBCLASS_OF edge — it must not
        # surface as a neighbor; the ES_-style chain still does.
        assert [n['name'] for n in result['neighbors']] == ['Detencion', 'Lugar']

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
# AGE arm — the ontology read path is flavour-owned
# ---------------------------------------------------------------------------

class TestAgeOntologyReadPath:
    """Regression for the Task 1 spike's DIVERGENT verdict.

    Shipped, the AGE arm returned `ontology_type=0 inherits_from=0 alt_labels=0
    identity=0 properties=0 examples=0` over 30 classes and `relationship_classes:
    []`, because both ontology queries were FalkorDB-shaped. The data was there the
    whole time — a corrected projection recovered every field.
    """

    @pytest.mark.asyncio
    async def test_documentation_recovers_the_structured_fields_on_age(self):
        from graphiti_mcp_server import get_ontology_documentation

        svc = make_age_service(
            [
                age_vehiculo_node(),
                age_physical_object_node(),
                age_ubicacion_node(),
                age_municion_node(),
            ],
            age_typed_edges(),
        )

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_documentation()

        assert 'error' not in result
        entities = {e['name']: e for e in result['entity_classes']}
        assert set(entities) == {'Vehiculo', 'PhysicalObject', 'Ubicacion', 'Municion'}

        vehiculo = entities['Vehiculo']
        # Every field that came back empty on the shipped AGE arm.
        assert vehiculo['ontology_type'] == 'class'
        assert vehiculo['inherits_from'] == ['PhysicalObject']
        assert vehiculo['alt_labels'] == ['Vehículo', 'Coche']
        assert vehiculo['examples'] == ['Seat León 1234-ABC']
        assert vehiculo['identity'] is True
        assert vehiculo['properties'] == AGE_VEHICULO_PROPERTIES
        # ...and the two that always worked still do.
        assert vehiculo['summary'] == VEHICULO_SUMMARY

        # The abstract supertype keeps its own ontology_type, not the leaf's.
        assert entities['PhysicalObject']['ontology_type'] == 'abstract_class'
        assert entities['PhysicalObject']['identity'] is False

        # The spike's recommended ship gate, in its own terms.
        assert sum(1 for c in result['entity_classes'] if c['inherits_from']) > 0

    @pytest.mark.asyncio
    async def test_documentation_derives_relationships_from_typed_edges_on_age(self):
        """AGE has ZERO RELATES_TO edges — the relation name is `type(r)`."""
        from graphiti_mcp_server import get_ontology_documentation

        svc = make_age_service(
            [
                age_vehiculo_node(),
                age_physical_object_node(),
                age_ubicacion_node(),
                age_municion_node(),
            ],
            age_typed_edges(),
        )

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_documentation()

        assert 'error' not in result
        relationships = {r['name']: r for r in result['relationship_classes']}
        # SUBCLASS_OF is in the AGE row set and is dropped by the SHARED filter,
        # exactly as on FalkorDB — the flavour never filters it. The shared drop
        # keys on this same `name` value, so reading it from `r.name` keeps that
        # filter working.
        assert set(relationships) == {'OCURRE_EN', INVOLUCRA_MUNICION_NAME}
        ocurre_en = relationships['OCURRE_EN']
        assert ocurre_en['source_entity'] == 'Vehiculo'
        assert ocurre_en['target_entity'] == 'Ubicacion'
        assert ocurre_en['summary'] == OCURRE_EN_PROSE  # prefix stripped

        # The accent case, and why `type(r)` is not good enough: this edge is
        # STORED under the RELATES_TO fallback label, so `type(r)` would name it
        # "RELATES_TO" — losing the relation outright AND breaking the prefix strip
        # in _edge_relationship_entry, which rebuilds the prefix from this name.
        municion = relationships[INVOLUCRA_MUNICION_NAME]
        assert municion['source_entity'] == 'Vehiculo'
        assert municion['target_entity'] == 'Municion'
        assert municion['summary'] == INVOLUCRA_MUNICION_PROSE

        # And the query the driver actually saw is an untyped edge match reading
        # the stored name — a token-for-token mirror of the base projection.
        rel_queries = [q for q in svc.ontology_client.driver.queries if 'OntologyClass)-[' in q]
        assert rel_queries, svc.ontology_client.driver.queries
        assert all('[r:RELATES_TO]' not in q for q in rel_queries), rel_queries
        assert all('-[r]->' in q for q in rel_queries), rel_queries
        assert all('r.name AS name' in q for q in rel_queries), rel_queries
        assert all('type(r)' not in q for q in rel_queries), rel_queries

    @pytest.mark.asyncio
    async def test_structure_recovers_the_structured_fields_on_age(self):
        """get_ontology_structure reads the same two dialect axes — and its FROZEN
        per-entry key set is unchanged by the fix."""
        from graphiti_mcp_server import get_ontology_structure

        svc = make_age_service(
            [age_vehiculo_node(), age_physical_object_node(), age_ubicacion_node()]
        )

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await get_ontology_structure()

        assert 'error' not in result
        entities = {e['name']: e for e in result['entity_classes']}
        assert entities['Vehiculo']['ontology_type'] == 'class'
        assert entities['Vehiculo']['inherits_from'] == ['PhysicalObject']
        assert entities['Vehiculo']['alt_labels'] == ['Vehículo', 'Coche']
        assert entities['Vehiculo']['examples'] == ['Seat León 1234-ABC']
        for entry in result['entity_classes']:
            assert set(entry.keys()) == FROZEN_ENTITY_KEYS
        # The map tier stays lightweight: no properties/identity crept in.
        issued = [q for q in svc.ontology_client.driver.queries if 'MATCH (n:OntologyClass)' in q]
        assert issued and all('n.uuid' not in q for q in issued), issued
        assert issued and all('AS identity' not in q for q in issued), issued

    @pytest.mark.asyncio
    async def test_explore_gets_full_context_on_age(self):
        from graphiti_mcp_server import explore_ontology

        svc = make_age_service(
            [
                age_vehiculo_node(),
                age_physical_object_node(),
                age_ubicacion_node(),
                age_municion_node(),
            ],
            age_typed_edges(),
        )

        with patch('graphiti_mcp_server.graphiti_service', svc):
            result = await explore_ontology(node_name='Vehiculo')

        assert 'error' not in result
        center = result['center']
        assert center['ontology_type'] == 'class'
        assert center['properties'] == AGE_VEHICULO_PROPERTIES
        assert center['identity'] is True

        assert result['relationships']['outgoing'] == [
            {'name': 'OCURRE_EN', 'target': 'Ubicacion', 'summary': OCURRE_EN_PROSE},
            {
                'name': INVOLUCRA_MUNICION_NAME,
                'target': 'Municion',
                'summary': INVOLUCRA_MUNICION_PROSE,
            },
        ]
        # The hierarchy travels via inherits_from, which was [] on the shipped arm.
        assert result['hierarchy']['parents'] == [
            {
                'name': 'PhysicalObject',
                'summary_line': 'An abstract physical thing.',
            }
        ]
        # ...and the SUBCLASS_OF edge does not double as a neighbour.
        assert [n['name'] for n in result['neighbors']] == ['Ubicacion', 'Municion']

    @pytest.mark.asyncio
    async def test_age_arm_never_issues_the_falkordb_shaped_class_query(self):
        """The loud form of the regression: a top-level projection returns nulls
        on AGE, so this pins the query text the driver actually receives."""
        from graphiti_mcp_server import get_ontology_documentation

        svc = make_age_service([age_vehiculo_node()])

        with patch('graphiti_mcp_server.graphiti_service', svc):
            await get_ontology_documentation()

        class_queries = [
            q for q in svc.ontology_client.driver.queries if 'MATCH (n:OntologyClass)' in q
        ]
        assert class_queries, svc.ontology_client.driver.queries
        for q in class_queries:
            assert 'n.attributes.ontology_type AS ontology_type' in q, q
            assert 'n.ontology_type AS ontology_type' not in q, q


# ---------------------------------------------------------------------------
# Registration — the new tool is registered next to get_ontology_structure
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_get_ontology_documentation_registered(self, monkeypatch):
        import graphiti_mcp_server as srv
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

        # An ontology graph is what makes these two servable at all (M11).
        monkeypatch.setattr(
            srv,
            'config',
            type(
                'C',
                (),
                {
                    'graphiti': type(
                        'G', (), {'ontology_graph': 'onto_v1', 'group_id': 'test_graph'}
                    )
                },
            ),
            raising=False,
        )
        register_dynamic_tools(profile)

        tools = mcp._tool_manager._tools
        assert 'get_ontology_documentation' in tools
        assert 'get_ontology_structure' in tools

    def test_the_ontology_tiers_are_not_registered_without_an_ontology_graph(
        self, monkeypatch
    ):
        """The other arm of the same fact (M11): the two bulk tiers exist to read
        a companion ontology graph, so a connector without one does not announce
        them."""
        import graphiti_mcp_server as srv
        from domain_profile import DomainProfile
        from graphiti_mcp_server import mcp, register_dynamic_tools

        monkeypatch.setattr(
            srv,
            'config',
            type(
                'C',
                (),
                {
                    'graphiti': type(
                        'G', (), {'ontology_graph': None, 'group_id': 'test_graph'}
                    )
                },
            ),
            raising=False,
        )
        register_dynamic_tools(DomainProfile(group_id='test_graph'))

        tools = mcp._tool_manager._tools
        assert 'get_ontology_documentation' not in tools
        assert 'get_ontology_structure' not in tools
