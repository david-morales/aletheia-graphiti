"""Over-the-wire tool-coverage matrix — every retrieval tool must return a
non-error envelope on EVERY flavour, exercised through a real MCP client.

This is the producer-side conformance gate (ADR-019/015): the server self-verifies
its whole announced retrieval surface, per flavour, over the wire — BEFORE any
consumer integrates it. It exists because the P1–P4 parity gates ran at the
*capability* level (calling the tool functions directly), which never hit the
MCPServer wire pipeline nor a live client. That blind spot let two classes of bug
ship green:
  * the MCPServer `total=False` TypedDict output-validation bug (fixed mcp-v1.1.1),
  * the AGE `search`(combined)/`search_ontology`/`explore_entity` failures
    (episode + community fanout raised; execute_query rejected params) —
    all of which pass every offline/structural test but fail the moment a real
    client calls the tool against a live backend.

Gated. Point each URL at a running connector serving a POPULATED graph
(nodes + edges + a companion ontology graph):

    TOOL_COVERAGE_LIVE=1 \
    TOOL_COVERAGE_FALKORDB_URL=http://localhost:8000/mcp/ \
    TOOL_COVERAGE_AGE_URL=http://localhost:8099/mcp/ \
    uv run python -m pytest tests/live/test_tool_coverage_matrix_live.py -v

A flavour whose URL is unset is skipped, so the FalkorDB and AGE arms can run
independently.
"""
import json
import os

import pytest

RUN = os.environ.get('TOOL_COVERAGE_LIVE') == '1'

FLAVOUR_URLS = {
    'falkordb': os.environ.get('TOOL_COVERAGE_FALKORDB_URL', ''),
    'age': os.environ.get('TOOL_COVERAGE_AGE_URL', ''),
}

# The full canonical retrieval surface. explore_entity / explore_ontology take an
# identifier, discovered per-graph at run time (see _discover). search is checked
# in its default (combined) mode AND explicitly, because combined is the mode that
# fans out the episode + community sub-searches that broke on AGE.
#
# search_mode='episodes' is here for the arm where it CANNOT match: on AGE
# `episode_fulltext_search` returns [] by construction, and this row asserts the
# mode still answers in band rather than erroring — an empty result is the honest
# answer, a failure is not. On FalkorDB the same row exercises the real leg.
NOARG_TOOLS = [
    ('get_schema', {}),
    ('graph_query', {'query': 'MATCH (n) RETURN count(n) AS c'}),
    ('search', {'query': 'test', 'limit': 3}),
    ('search', {'query': 'test', 'search_mode': 'combined', 'limit': 3}),
    ('search', {'query': 'test', 'search_mode': 'episodes', 'limit': 3}),
    ('search_ontology', {'query': 'test', 'limit': 3}),
    ('get_ontology_structure', {}),
    ('get_ontology_documentation', {}),
    ('profile_data', {'sample_size': 3}),
]


def _text(result) -> str:
    return ' '.join(
        c.text for c in getattr(result, 'content', []) if getattr(c, 'type', '') == 'text'
    )


def _is_error(result) -> bool:
    if getattr(result, 'is_error', False):
        return True
    try:
        payload = json.loads(_text(result))
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and bool(payload.get('error'))


async def _call(session, tool, args):
    result = await session.call_tool(tool, args)
    return (not _is_error(result)), _text(result)[:160]


async def _discover_node_name(session) -> str | None:
    result = await session.call_tool(
        'graph_query',
        {'query': 'MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name LIMIT 1'},
    )
    if _is_error(result):
        return None
    try:
        payload = json.loads(_text(result))
        recs = payload.get('result') or payload.get('rows') or []
        if recs:
            first = recs[0]
            return first.get('name') if isinstance(first, dict) else first[0]
    except Exception:
        pass
    return None


async def _discover_ontology_class(session) -> str | None:
    result = await session.call_tool('get_ontology_structure', {})
    try:
        payload = json.loads(_text(result))
        classes = payload.get('entity_classes') or []
        if classes:
            c = classes[0]
            return c.get('name') if isinstance(c, dict) else c
    except Exception:
        pass
    return None


@pytest.mark.skipif(not RUN, reason='set TOOL_COVERAGE_LIVE=1 to run the over-the-wire matrix')
@pytest.mark.asyncio
@pytest.mark.parametrize('flavour', list(FLAVOUR_URLS))
async def test_all_retrieval_tools_green_over_the_wire(flavour):
    url = FLAVOUR_URLS[flavour]
    if not url:
        pytest.skip(f'{flavour}: set TOOL_COVERAGE_{flavour.upper()}_URL')

    # SDK 2.x: one `Client` replaces the transport + ClientSession + initialize()
    # layering. Constructing a bare `ClientSession` and calling it now raises
    # (`send_raw_request called before run()`), so the layered form is not merely
    # deprecated here — it does not work.
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    failures: dict[str, str] = {}
    async with Client(streamable_http_client(url)) as session:
        announced = {t.name for t in (await session.list_tools()).tools}
        expected = {
            'get_schema', 'graph_query', 'search', 'search_ontology',
            'explore_entity', 'explore_ontology', 'get_ontology_structure',
            'get_ontology_documentation', 'profile_data',
        }
        missing = expected - announced
        assert not missing, f'{flavour}: connector does not announce {sorted(missing)}'

        calls = list(NOARG_TOOLS)
        node_name = await _discover_node_name(session)
        if node_name:
            calls.append(('explore_entity', {'node_name': node_name, 'depth': 1, 'limit': 3}))
        else:
            failures['explore_entity'] = 'could not discover a named node to explore'
        onto_class = await _discover_ontology_class(session)
        if onto_class:
            calls.append(('explore_ontology', {'node_name': onto_class, 'depth': 1, 'limit': 3}))
        else:
            failures['explore_ontology'] = 'could not discover an ontology class to explore'

        for tool, args in calls:
            label = f'{tool}({args.get("search_mode", "")})' if tool == 'search' else tool
            ok, detail = await _call(session, tool, args)
            if not ok:
                failures[label] = detail

    assert not failures, f'{flavour}: tools returned an error envelope over the wire:\n' + '\n'.join(
        f'  {name}: {detail}' for name, detail in failures.items()
    )
