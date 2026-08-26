"""M11 addendum: a not-found is an ANSWER, and answers go in `message`.

`explore_ontology` filed "no ontology class matches that name" under `error`,
while `explore_entity` and `search` file exactly the same situation — the graph
was queried, it answered, the answer is nothing — under `message`. Two tools, one
situation, two envelopes.

That is not a cosmetic inconsistency. ADR-015 R4 makes `error` the in-band
failure channel, so a consumer loop counting failures per tool counts an
under-`error` not-found as a tool FAILURE: it retries a call whose answer will
not change, and it downgrades a connector that is working. Ruled
loop-side-correct-by-contract (audit 2026-08-15) — the fix belongs here, in the
fork's taxonomy:

    not-found / no-results  ->  `message`      (the graph answered; it was empty)
    everything else         ->  `error`        (the call could not be answered)

The boundary matters as much as the move, so both sides are pinned. A bad
argument, an unready service, an unconfigured capability and a backend fault stay
under `error`: none of them is the graph answering. Only the resolution failing
against a live, configured ontology is.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import jsonschema
import pytest
from mcp.server.mcpserver import MCPServer

import graphiti_mcp_server as srv
from config.schema import GraphitiConfig
from flavours.falkordb import FalkorDbFlavour
from tool_annotations import annotations_for

# No database, no API key.
pytestmark = pytest.mark.contract


WIDGET_ROW = {
    'uuid': 'widget-uuid',
    'name': 'Widget',
    'ontology_type': 'class',
    'summary': 'A widget.',
    'alt_labels': '',
    'inherits_from': [],
    'examples': '',
    'identity': False,
    'properties': None,
}


def make_service(rows: list[dict], *, ontology_graph: str | None = 'test_ontology'):
    """A service whose ontology driver returns `rows` and resolves nothing fuzzily.

    Carries a REAL FalkorDbFlavour: the ontology query texts are flavour-owned,
    so a MagicMock flavour would hand the driver a MagicMock instead of Cypher.
    Convention borrowed from `test_ontology_tiers.py`.
    """
    svc = MagicMock()
    svc.config = GraphitiConfig()
    svc.config.graphiti.ontology_graph = ontology_graph
    svc._ensure_ontology_client = AsyncMock(return_value=ontology_graph is not None)
    svc.flavour = FalkorDbFlavour()

    async def dispatch_query(query, **kwargs):
        if 'RELATES_TO' in query:
            return ([], None, None)
        return (rows, None, None)

    client = MagicMock()
    client.driver.execute_query = AsyncMock(side_effect=dispatch_query)
    client.search_ = AsyncMock(return_value=MagicMock(nodes=[]))
    svc.ontology_client = client
    return svc


def _explore(service, **kwargs):
    with_service = srv.graphiti_service
    srv.graphiti_service = service
    try:
        return asyncio.run(srv.explore_ontology(**kwargs))
    finally:
        srv.graphiti_service = with_service


def _failed(result: dict) -> bool:
    """What a consumer applying ADR-015 R4 concludes: `if result.get('error')`."""
    return bool(result.get('error'))


# ---------------------------------------------------------------------------
# The move: a live, configured ontology that simply holds no such class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'kwargs',
    [
        {'node_name': 'NoSuchClass'},
        {'node_uuid': 'no-such-uuid'},
    ],
    ids=['by-name', 'by-uuid'],
)
def test_a_missing_class_is_not_a_failure(kwargs):
    result = _explore(make_service([WIDGET_ROW]), **kwargs)
    assert not _failed(result), (
        f'a not-found came back as a tool FAILURE: {result.get("error")!r}. The '
        f'ontology was configured, reachable and queried — it answered, and the '
        f'answer was nothing.'
    )


@pytest.mark.parametrize(
    'kwargs, needle',
    [
        ({'node_name': 'NoSuchClass'}, 'NoSuchClass'),
        ({'node_uuid': 'no-such-uuid'}, 'no-such-uuid'),
    ],
    ids=['by-name', 'by-uuid'],
)
def test_the_not_found_answer_still_says_what_was_not_found(kwargs, needle):
    """Moving the field must not cost the caller the information."""
    result = _explore(make_service([WIDGET_ROW]), **kwargs)
    assert needle in (result.get('message') or ''), result


def test_the_not_found_payload_is_shaped_like_a_found_one(kwargs=None):
    """An empty ANSWER, not an absent one — the `explore_entity` precedent, whose
    not-found returns `nodes=[], edges=[], communities=[]` around a null centre.
    A consumer can iterate the result without a None-check either way."""
    result = _explore(make_service([WIDGET_ROW]), node_name='NoSuchClass')
    assert result['center'] is None
    assert result['relationships'] == {'outgoing': [], 'incoming': []}
    assert result['hierarchy'] == {'parents': [], 'children': [], 'siblings': []}
    assert result['neighbors'] == []


def test_a_class_that_does_exist_is_unaffected():
    """The move must not turn a hit into a miss."""
    result = _explore(make_service([WIDGET_ROW]), node_name='Widget')
    assert not _failed(result)
    assert result['center']['name'] == 'Widget'


# ---------------------------------------------------------------------------
# The boundary: what stays under `error`
# ---------------------------------------------------------------------------


def test_an_unready_service_is_an_error():
    """Nothing was queried, so nothing answered."""
    result = _explore(None, node_name='Widget')
    assert _failed(result)


def test_an_unconfigured_ontology_is_an_error():
    """Post-M11 this tool is not even ANNOUNCED without an ontology graph, so
    reaching this guard means a caller invoked a tool the server never offered.
    That is a contract violation, not the graph answering."""
    result = _explore(make_service([], ontology_graph=None), node_name='Widget')
    assert _failed(result)


def test_a_missing_argument_is_an_error():
    """The caller named nothing to look for — no lookup happened."""
    result = _explore(make_service([WIDGET_ROW]))
    assert _failed(result)


def test_a_backend_fault_is_an_error():
    """A driver that raises must NOT be dressed up as 'no such class'. This is the
    failure the move could plausibly cause: both paths end with no centre row."""
    service = make_service([WIDGET_ROW])
    service.ontology_client.driver.execute_query = AsyncMock(
        side_effect=RuntimeError('connection reset')
    )
    result = _explore(service, node_name='Widget')
    assert _failed(result)
    assert 'connection reset' in result['error']


def test_a_backend_fault_does_not_masquerade_as_a_not_found():
    """Stated from the other side: the fault must not arrive under `message`."""
    service = make_service([WIDGET_ROW])
    service.ontology_client.driver.execute_query = AsyncMock(
        side_effect=RuntimeError('connection reset')
    )
    result = _explore(service, node_name='Widget')
    assert 'connection reset' not in (result.get('message') or '')


# ---------------------------------------------------------------------------
# The doctrine, stated across tools
# ---------------------------------------------------------------------------


def test_explore_ontology_and_explore_entity_file_a_not_found_the_same_way():
    """The inconsistency the ruling was about, pinned as one invariant.

    `explore_entity` is the reference implementation: it has always answered
    'No node found matching "X"' under `message`.
    """
    ontology = _explore(make_service([WIDGET_ROW]), node_name='NoSuchClass')

    service = MagicMock()
    client = MagicMock()
    client.search_ = AsyncMock(return_value=MagicMock(nodes=[]))
    service.get_client = AsyncMock(return_value=client)

    cfg = GraphitiConfig()
    cfg.graphiti.group_id = 'g'
    previous_service, previous_config = srv.graphiti_service, srv.__dict__.get('config')
    srv.graphiti_service = service
    srv.config = cfg
    try:
        entity = asyncio.run(srv.explore_entity(node_name='NoSuchEntity'))
    finally:
        srv.graphiti_service = previous_service
        if previous_config is None:
            srv.__dict__.pop('config', None)
        else:
            srv.config = previous_config

    assert _failed(entity) == _failed(ontology) is False
    assert (entity.get('message') or '') and (ontology.get('message') or '')


def test_search_ontology_keeps_reporting_an_empty_result_as_a_message():
    """Already correct, and pinned so it cannot drift back.

    Zero hits is the ontology answering, exactly as a zero-hit `search` is.
    """
    service = make_service([])
    service.ontology_client.search_ = AsyncMock(
        return_value=MagicMock(nodes=[], edges=[], communities=[])
    )
    previous_service, previous_config = srv.graphiti_service, srv.__dict__.get('config')
    cfg = GraphitiConfig()
    cfg.graphiti.ontology_graph = 'test_ontology'
    srv.graphiti_service = service
    srv.config = cfg
    try:
        result = asyncio.run(srv.search_ontology(query='nothing matches this'))
    finally:
        srv.graphiti_service = previous_service
        if previous_config is None:
            srv.__dict__.pop('config', None)
        else:
            srv.config = previous_config

    assert not _failed(result)
    assert result.get('message')


# ---------------------------------------------------------------------------
# The published contract
# ---------------------------------------------------------------------------


def test_the_not_found_payload_survives_the_published_output_schema():
    """`message` has to be DECLARED, not merely returned.

    MCPServer validates structured content against the tool's `outputSchema`
    before sending, so a field absent from the TypedDict is dropped or rejected
    outside the tool's try/except — a protocol error, not an ADR-015 envelope.
    """
    server = MCPServer('not-found-probe')
    server.add_tool(srv.explore_ontology, annotations=annotations_for('explore_ontology'))
    tool = server._tool_manager._tools['explore_ontology']

    payload = _explore(make_service([WIDGET_ROW]), node_name='NoSuchClass')
    meta = tool.fn_metadata
    structured = meta.convert_result(payload).structured_content
    jsonschema.validate(structured, meta.output_schema)

    assert 'message' in (meta.output_schema.get('properties') or {}), (
        'explore_ontology returns a `message` that its outputSchema never declares'
    )
    assert structured['message']
