"""ADR-019 R2: every served tool publishes a MEANINGFUL, FLAT outputSchema.

Two audit findings meet here.

A-D6 — three tools returned `dict[str, Any]`, so their published `outputSchema`
was `{"type":"object","additionalProperties":true}`: a schema that constrains
nothing and, worse, hides the ADR-015 R4 error path from every consumer reading
the surface. `explore_ontology` had the same problem one level in.

A-D7 — the surface carried two envelope families: four tools returned a flat
typed payload while eleven returned `X | ErrorResponse`, which makes FastMCP nest
`structuredContent` under `result`. A consumer needed per-tool knowledge of which
bucket a tool was in, and the ADR-015 R4 client rule `if "error" in result` held
only for the flat four. The fork had already reasoned this out for `run_cypher`
(`response_types.py`, "a union return would nest it under `result`") and never
applied it to the rest.

Every case below runs the payload through FastMCP's real `convert_result` and
validates it against the tool's real `output_schema` — the exact check the
lowlevel server performs before sending. A nullability slip fails HERE rather
than in production, where it dies outside the tool's try/except and escapes the
ADR-015 envelope as a protocol error.
"""

from __future__ import annotations

import asyncio
import json

import jsonschema
import pytest
from mcp.server.fastmcp import FastMCP

import graphiti_mcp_server as srv
from tool_annotations import TOOL_ANNOTATIONS, annotations_for

# Every tool that must publish a flat, non-degenerate outputSchema, with a
# success payload and the ADR-015 R4 error payload for each.
FLAT_SURFACE: dict[str, dict] = {
    'get_ontology_structure': {
        'success': {
            'ontology_graph': 'onto_v1',
            'entity_classes': [
                {
                    'name': 'Widget',
                    'ontology_type': 'class',
                    'summary': 'A widget.',
                    'alt_labels': ['Gizmo'],
                    'inherits_from': ['Thing'],
                    'examples': ['W-1'],
                }
            ],
            'relationship_classes': [
                {
                    'name': 'USES',
                    'ontology_type': 'relationship_class',
                    'summary': 'A uses B.',
                    'alt_labels': [],
                    'inherits_from': [],
                    'examples': [],
                    'source_entity': 'Widget',
                    'target_entity': 'Gadget',
                }
            ],
        },
        'error': {'error': 'No ontology graph configured for this connector.'},
        'required_keys': ('ontology_graph', 'entity_classes', 'relationship_classes', 'error'),
    },
    'get_ontology_documentation': {
        'success': {
            'ontology_graph': 'onto_v1',
            'entity_classes': [
                {
                    'name': 'Widget',
                    'ontology_type': 'class',
                    'summary': 'A widget, at length.',
                    'alt_labels': [],
                    'inherits_from': [],
                    'examples': [],
                    'identity': True,
                    'properties': [
                        {
                            'name': 'serial',
                            'label': 'Serial',
                            'range': 'string',
                            'comment': 'The serial number.',
                            'required': True,
                        }
                    ],
                }
            ],
            'relationship_classes': [],
        },
        'error': {'error': 'Failed to retrieve ontology documentation: boom'},
        'required_keys': ('ontology_graph', 'entity_classes', 'relationship_classes', 'error'),
    },
    'profile_graph': {
        'success': {
            'entity_profiles': {'Widget': {'count': 3, 'properties': {}}},
            'relationship_profiles': {'USES': {'count': 2}},
            'language_summary': {'primary_languages': ['es'], 'multilingual_fields': []},
        },
        'error': {'error': 'Failed to profile graph: boom'},
        'required_keys': (
            'entity_profiles',
            'relationship_profiles',
            'language_summary',
            'error',
        ),
    },
    'explore_ontology': {
        'success': {
            'center': {
                'name': 'Widget',
                'ontology_type': 'class',
                'summary': 'A widget.',
                'alt_labels': [],
                'inherits_from': ['Thing'],
                'examples': [],
                'identity': False,
                'properties': [],
            },
            'relationships': {
                'outgoing': [{'name': 'USES', 'target': 'Gadget', 'summary': 'A uses B.'}],
                'incoming': [{'name': 'OWNS', 'source': 'Owner', 'summary': 'X owns Y.'}],
            },
            'hierarchy': {
                'parents': [{'name': 'Thing', 'summary_line': 'Anything.'}],
                'children': [],
                'siblings': [{'name': 'Gadget', 'summary_line': 'A gadget.'}],
            },
            'neighbors': [{'name': 'Gadget', 'summary_line': 'A gadget.', 'via': 'USES'}],
        },
        'error': {'error': 'Ontology explore error: boom'},
        'required_keys': ('center', 'relationships', 'hierarchy', 'neighbors', 'error'),
    },
}


@pytest.fixture(scope='module')
def isolated_tools():
    """A throwaway FastMCP carrying the whole surface — never the module global."""
    m = FastMCP('typed-output-probe')
    for name in sorted(TOOL_ANNOTATIONS):
        m.add_tool(getattr(srv, name), annotations=annotations_for(name))
    return m


@pytest.fixture(scope='module')
def listed(isolated_tools):
    return {t.name: t for t in asyncio.run(isolated_tools.list_tools())}


@pytest.mark.parametrize('tool_name', sorted(FLAT_SURFACE))
def test_previously_untyped_tools_publish_their_keys(listed, tool_name):
    schema = listed[tool_name].outputSchema
    assert schema is not None, f'{tool_name} publishes no outputSchema'
    props = schema.get('properties')
    assert props, (
        f'{tool_name} publishes a degenerate outputSchema '
        f'({schema}) — it constrains nothing (A-D6)'
    )
    for key in FLAT_SURFACE[tool_name]['required_keys']:
        assert key in props, f'{tool_name} outputSchema missing {key}'


@pytest.mark.parametrize('tool_name', sorted(TOOL_ANNOTATIONS))
def test_no_tool_nests_its_payload_under_result(listed, tool_name):
    """A-D7: one envelope family. `if "error" in result` must work for EVERY tool."""
    schema = listed[tool_name].outputSchema
    assert schema is not None, f'{tool_name} publishes no outputSchema'
    props = set(schema.get('properties') or {})
    assert props != {'result'}, (
        f'{tool_name} still nests structuredContent under "result" — a union return '
        f'(X | ErrorResponse) wraps the payload one level (A-D7)'
    )


# `get_status` is the one tool whose failure is not an ADR-015 R4 in-band error:
# "the database is unreachable" is its ANSWER, reported as `status: "error"` with a
# `message`, not as a failed call. Folding an `error` key in would give it two ways
# to say the same thing.
NO_ERROR_KEY = {'get_status'}


@pytest.mark.parametrize(
    'tool_name', sorted(set(TOOL_ANNOTATIONS) - NO_ERROR_KEY)
)
def test_every_tool_publishes_the_error_key(listed, tool_name):
    """ADR-015 R4 is part of the CONTRACT, so it belongs in the published schema."""
    props = set(listed[tool_name].outputSchema.get('properties') or {})
    assert 'error' in props, (
        f'{tool_name} does not publish its ADR-015 R4 error path in outputSchema'
    )


def test_get_status_reports_health_through_status_not_error(listed):
    """Pin the exemption above so it stays a decision, not an oversight."""
    props = set(listed['get_status'].outputSchema.get('properties') or {})
    assert props == {'status', 'message'}


def _validates(tool, payload):
    meta = tool.fn_metadata
    converted = meta.convert_result(payload)
    structured = converted[1] if isinstance(converted, tuple) else converted
    jsonschema.validate(structured, meta.output_schema)
    return structured


@pytest.mark.parametrize('tool_name', sorted(FLAT_SURFACE))
@pytest.mark.parametrize('kind', ['success', 'error'])
def test_payloads_survive_fastmcp_output_validation(isolated_tools, tool_name, kind):
    tool = isolated_tools._tool_manager._tools[tool_name]
    _validates(tool, FLAT_SURFACE[tool_name][kind])


@pytest.mark.parametrize('tool_name', sorted(FLAT_SURFACE))
def test_an_all_null_payload_survives_validation(isolated_tools, tool_name):
    """FastMCP injects None for every absent optional field and dumps without
    exclude_unset, so a non-nullable field type rejects its own injected None."""
    tool = isolated_tools._tool_manager._tools[tool_name]
    all_null = dict.fromkeys(FLAT_SURFACE[tool_name]['required_keys'])
    _validates(tool, all_null)


@pytest.mark.parametrize(
    ('tool_name', 'payload'),
    [
        ('search', {'message': '2 found', 'nodes': [], 'edges': [], 'communities': []}),
        ('search', {'error': 'boom'}),
        ('explore_node', {'message': 'no node', 'center_node': None, 'nodes': [],
                          'edges': [], 'communities': []}),
        ('clear_graph', {'message': 'cleared'}),
        ('add_memory', {'message': 'queued'}),
        ('get_episodes', {'message': '1 episode', 'episodes': [{'uuid': 'e1'}]}),
    ],
)
def test_the_unstructured_payload_is_byte_identical_after_flattening(
    isolated_tools, tool_name, payload
):
    """THE backward-compatibility guard for A-D7.

    Consumers parse the unstructured content and check `"error" in result`
    (aletheia's `per_call_client._parse_mcp_response`). Flattening changes only
    where `structuredContent` puts the payload — the text content must stay
    exactly the dict the tool returned, with no injected nulls and no new keys.
    """
    tool = isolated_tools._tool_manager._tools[tool_name]
    unstructured, structured = tool.fn_metadata.convert_result(payload)

    assert json.loads(unstructured[0].text) == payload, (
        'the text content a consumer parses must be unchanged by the typing'
    )
    # ...while structuredContent is now flat rather than nested under "result".
    assert 'result' not in structured or 'result' in payload
    for key, value in payload.items():
        assert structured[key] == value


@pytest.mark.parametrize('tool_name', sorted(FLAT_SURFACE))
def test_the_success_payload_keys_survive_the_round_trip(isolated_tools, tool_name):
    """Backward compatibility: the keys consumers already read must still be there
    with their values — the typing is additive, never a rename or a drop."""
    tool = isolated_tools._tool_manager._tools[tool_name]
    payload = FLAT_SURFACE[tool_name]['success']
    structured = _validates(tool, payload)
    for key, value in payload.items():
        assert key in structured, f'{tool_name}: {key} dropped from structuredContent'
        assert structured[key] == value, f'{tool_name}: {key} altered in structuredContent'
