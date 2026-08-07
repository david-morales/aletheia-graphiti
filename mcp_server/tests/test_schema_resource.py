"""A-D8 / ADR-019 R4: the schema is stable, cacheable context — serve it as a resource.

Three domain-profile resources existed; the two payloads the ADR names by example
(the schema, the ontology) were tool-only. The schema is the single thing every
consumer needs before it can do anything — labels, relationship patterns, and the
backend `dialect_reference` — and it is already cached server-side, so requiring a
tool round-trip to reach it is pure friction.

Read through `get_schema()` rather than off `_schema_cache` directly: that honours
the dirty flag, so the resource can never serve a schema the tool would not.
"""

from __future__ import annotations

import json

import pytest

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo

SCHEMA_URI = 'graphiti://schema'
PROFILE_URIS = (
    'graphiti://domain_summary',
    'graphiti://entity_catalog',
    'graphiti://relationship_types',
)

pytestmark = pytest.mark.asyncio


def _profile(group_id: str = 'resource_graph') -> DomainProfile:
    return DomainProfile(
        group_id=group_id,
        entity_types={'Widget': EntityTypeInfo('Widget', 3, 'A widget', ['W-1'])},
        edge_types={'USES': EdgeTypeInfo('USES', 1, 'uses', 'Widget -> Widget')},
        time_range=None,
    )


@pytest.fixture
def registered(monkeypatch):
    """Resources registered, with `get_schema` stubbed to a known payload."""
    payload = {
        'type': 'schema',
        'graph_name': 'resource_graph',
        'dialect': 'falkordb-cypher',
        'dialect_reference': 'no APOC; use toLower()',
        'node_labels': {'Widget': {'count': 3, 'attribute_keys': ['serial']}},
        'relationship_types': {'USES': {'count': 1, 'patterns': [['Widget', 'Widget']]}},
    }
    calls = []

    async def _stub_get_schema():
        calls.append(1)
        return payload

    monkeypatch.setattr(srv, 'get_schema', _stub_get_schema)
    srv.register_resources(_profile())
    return payload, calls


async def _read(uri: str) -> str:
    contents = await srv.mcp.read_resource(uri)
    return next(iter(contents)).content


async def test_the_schema_resource_is_listed(registered):
    uris = {str(r.uri) for r in await srv.mcp.list_resources()}
    assert SCHEMA_URI in uris


async def test_the_existing_profile_resources_are_still_listed(registered):
    uris = {str(r.uri) for r in await srv.mcp.list_resources()}
    assert set(PROFILE_URIS) <= uris


async def test_the_schema_resource_is_readable_and_is_the_schema(registered):
    payload, _ = registered
    assert json.loads(await _read(SCHEMA_URI)) == payload


async def test_the_schema_resource_is_announced_as_json(registered):
    resource = next(r for r in await srv.mcp.list_resources() if str(r.uri) == SCHEMA_URI)
    assert resource.mime_type == 'application/json'
    assert resource.description


async def test_reading_goes_through_the_tool_not_the_cache(registered):
    """Reading off `_schema_cache` directly would ignore the dirty flag and could
    serve a schema the tool itself would refuse."""
    _, calls = registered
    await _read(SCHEMA_URI)
    assert calls, 'the resource did not call get_schema()'


async def test_the_resource_reflects_a_changed_schema(registered):
    """It must not freeze the first payload it ever served."""
    await _read(SCHEMA_URI)

    async def _changed():
        return {'type': 'schema', 'graph_name': 'after_reingest', 'node_labels': {}}

    srv.mcp._resource_manager._resources[SCHEMA_URI].fn = _changed
    assert json.loads(await _read(SCHEMA_URI))['graph_name'] == 'after_reingest'


async def test_an_error_from_get_schema_is_served_as_the_error_payload(monkeypatch):
    """ADR-015 R4 all the way through: the resource does not invent a second
    failure mode for something the tool already reports in-band."""

    async def _failing():
        return {'error': 'Failed to retrieve schema: boom'}

    monkeypatch.setattr(srv, 'get_schema', _failing)
    srv.register_resources(_profile())
    assert json.loads(await _read(SCHEMA_URI)) == {'error': 'Failed to retrieve schema: boom'}


async def test_the_degraded_path_still_serves_the_schema_resource(monkeypatch):
    """A degraded connector is exactly when a consumer needs the live schema:
    every description it is reading is a static fallback."""
    payload = {'type': 'schema', 'graph_name': 'degraded_graph', 'node_labels': {}}

    async def _stub():
        return payload

    monkeypatch.setattr(srv, 'get_schema', _stub)
    srv.register_fallback_tools(reason='boom')

    uris = {str(r.uri) for r in await srv.mcp.list_resources()}
    assert SCHEMA_URI in uris
    assert json.loads(await _read(SCHEMA_URI)) == payload


async def test_re_registration_replaces_rather_than_keeps_the_stale_resource(monkeypatch):
    """MCPServer's `add_resource` KEEPS the existing entry on a duplicate URI, so a
    re-profile would otherwise go on serving the text rendered from the old one."""

    async def _stub():
        return {'type': 'schema'}

    monkeypatch.setattr(srv, 'get_schema', _stub)
    srv.register_resources(_profile('first_graph'))
    assert 'first_graph' in await _read('graphiti://domain_summary')

    srv.register_resources(_profile('second_graph'))
    summary = await _read('graphiti://domain_summary')
    assert 'second_graph' in summary
    assert 'first_graph' not in summary
