"""M3: author-owned ontology content must not be able to crash the tool.

A-D6 was about the TOP LEVEL publishing a degenerate `{"type":"object"}` schema.
The first fix over-reached and typed the NESTED, author-owned content too —
`properties` entries, `alt_labels`, `inherits_from`, `examples` — all of which are
whatever JSON the ontology author stored, returned verbatim by `_parse_properties`.

That created a failure mode worse than the one it fixed. pydantic validates the
payload inside `FuncMetadata.convert_result`, which the lowlevel server calls
AFTER the tool has returned — outside the tool's `try/except`. A validation error
therefore escapes as a PROTOCOL error, which is exactly the ADR-015 R4 breach the
branch exists to prevent, and it is invisible in the tool's own logging.

Each shape below was verified to RAISE before the fix. `inherited_from` was
verified to be silently dropped — and our own generator
(`aletheia/core/ontology/generic_loader.py`) emits it on every property.
"""

from __future__ import annotations

import jsonschema
import pytest
from mcp.server.fastmcp import FastMCP

import graphiti_mcp_server as srv

ONTOLOGY_TOOLS = ('get_ontology_documentation', 'get_ontology_structure', 'explore_ontology')


@pytest.fixture(scope='module')
def tools():
    m = FastMCP('ontology-variants')
    for name in ONTOLOGY_TOOLS:
        m.add_tool(getattr(srv, name))
    return m._tool_manager._tools


def _entry(**overrides) -> dict:
    entry = {
        'name': 'Widget',
        'ontology_type': 'class',
        'summary': 'A widget.',
        'alt_labels': [],
        'inherits_from': [],
        'examples': [],
    }
    entry.update(overrides)
    return entry


def _roundtrip(tool, payload):
    """Drive the real convert_result + schema validation the server performs."""
    meta = tool.fn_metadata
    converted = meta.convert_result(payload)
    structured = converted[1] if isinstance(converted, tuple) else converted
    jsonschema.validate(structured, meta.output_schema)
    return structured


# Shapes a real ontology can hold. Every one of these raised ValidationError
# before the fix — from OUTSIDE the tool's try/except.
HOSTILE_ENTRIES = {
    'properties_as_list_of_strings': _entry(properties=['serial', 'name']),
    'alt_labels_as_list_of_dicts': _entry(alt_labels=[{'lang': 'es', 'v': 'Trasto'}]),
    'required_is_not_a_bool': _entry(properties=[{'name': 'serial', 'required': 'sometimes'}]),
    'inherits_from_as_dicts': _entry(inherits_from=[{'class': 'Thing'}]),
    'examples_as_objects': _entry(examples=[{'value': 'W-1'}]),
    'properties_is_empty': _entry(properties=[]),
    'summary_is_absent': {'name': 'Widget', 'ontology_type': 'class'},
}


@pytest.mark.parametrize('shape', sorted(HOSTILE_ENTRIES))
def test_documentation_survives_author_written_shapes(tools, shape):
    _roundtrip(
        tools['get_ontology_documentation'],
        {
            'ontology_graph': 'onto_v1',
            'entity_classes': [HOSTILE_ENTRIES[shape]],
            'relationship_classes': [],
        },
    )


@pytest.mark.parametrize('shape', sorted(HOSTILE_ENTRIES))
def test_structure_survives_author_written_shapes(tools, shape):
    _roundtrip(
        tools['get_ontology_structure'],
        {
            'ontology_graph': 'onto_v1',
            'entity_classes': [HOSTILE_ENTRIES[shape]],
            'relationship_classes': [],
        },
    )


@pytest.mark.parametrize('shape', sorted(HOSTILE_ENTRIES))
def test_explore_ontology_survives_author_written_shapes(tools, shape):
    _roundtrip(
        tools['explore_ontology'],
        {
            'center': HOSTILE_ENTRIES[shape],
            'relationships': {'outgoing': [], 'incoming': []},
            'hierarchy': {'parents': [], 'children': [], 'siblings': []},
            'neighbors': [],
        },
    )


def test_property_keys_the_typing_does_not_know_are_not_dropped(tools):
    """`inherited_from` is emitted by aletheia's own ontology generator on every
    property; the first typing pass silently removed it from structuredContent,
    along with anything else an ontology author chose to record."""
    prop = {
        'name': 'serial',
        'label': 'Serial',
        'range': 'string',
        'comment': 'The serial number.',
        'required': True,
        'inherited_from': 'Thing',
        'cardinality': '1..*',
        'datatype': 'xsd:string',
    }
    structured = _roundtrip(
        tools['get_ontology_documentation'],
        {
            'ontology_graph': 'onto_v1',
            'entity_classes': [_entry(properties=[prop])],
            'relationship_classes': [],
        },
    )
    assert structured['entity_classes'][0]['properties'] == [prop], (
        'author-owned property keys must survive the round trip unchanged'
    )


def test_the_top_level_is_still_typed(tools):
    """The A-D6 goal is met without constraining author-owned nested content."""
    for name in ONTOLOGY_TOOLS:
        schema = tools[name].fn_metadata.output_schema
        props = set(schema.get('properties') or {})
        assert props, f'{name} publishes a degenerate schema again'
        assert 'error' in props, f'{name} lost its ADR-015 R4 error key'


def test_a_well_formed_payload_still_validates(tools):
    """Loosening must not turn into "accepts anything, means nothing"."""
    structured = _roundtrip(
        tools['get_ontology_documentation'],
        {
            'ontology_graph': 'onto_v1',
            'entity_classes': [_entry(identity=True, properties=[{'name': 'serial'}])],
            'relationship_classes': [
                _entry(
                    name='USES',
                    ontology_type='relationship_class',
                    source_entity='Widget',
                    target_entity='Gadget',
                )
            ],
        },
    )
    assert structured['entity_classes'][0]['identity'] is True
    assert structured['relationship_classes'][0]['source_entity'] == 'Widget'
