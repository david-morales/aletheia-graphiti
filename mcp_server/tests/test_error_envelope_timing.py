"""BUG-33(a) — an error envelope must report the wall clock it actually spent.

`format_error` hardcoded `execution_ms: 0` on every path, so a consumer could not
tell a rejection the validator made before touching the database from a query the
database ran for four seconds and then failed. They cost wildly different amounts and
a client's retry/breaker policy needs to tell them apart: the workbench breaker reads
this envelope, and with everything reported as free the expensive class was classified
cheap (the BUG-33 chain).

Zero stays the right answer for the pre-flight paths — they really did cost nothing.
The point is that it is now MEASURED rather than asserted.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from flavours.age import AgeFlavour  # noqa: E402
from flavours.falkordb import FalkorDbFlavour  # noqa: E402
from utils.cypher import CypherError, format_error  # noqa: E402

_FLAVOUR = FalkorDbFlavour()

# Long enough that a rounded-to-0.1ms wall clock cannot land on zero by accident,
# short enough not to slow the suite.
_SLOW_FAILURE_SECONDS = 0.05
_SLOW_FAILURE_MS = _SLOW_FAILURE_SECONDS * 1000


def _err(stage: str = 'execution', reason: str = 'query_failed') -> CypherError:
    return CypherError(
        stage=stage,
        reason=reason,
        found='',
        explanation='It broke.',
        suggestion='Try something else.',
    )


# ---------------------------------------------------------------------------
# the formatter
# ---------------------------------------------------------------------------


def test_format_error_reports_the_execution_ms_it_is_given():
    envelope = format_error('MATCH (n) RETURN n', _err(), execution_ms=4210.7)
    assert envelope['execution_ms'] == 4210.7


def test_format_error_defaults_to_zero_for_paths_that_never_executed():
    """A validator rejection really did cost nothing — 0 is the honest value there."""
    envelope = format_error('CREATE (n)', _err(stage='security', reason='write_operation'))
    assert envelope['execution_ms'] == 0.0


def test_format_error_keeps_the_rest_of_the_envelope_unchanged():
    envelope = format_error('MATCH (n) RETURN n', _err(), execution_ms=12.5)
    assert envelope['type'] == 'error'
    assert isinstance(envelope['error'], str)
    assert envelope['error_detail']['stage'] == 'execution'
    assert envelope['cypher_quality']['outcome'] == 'error'


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------


def _mock_service(flavour, *, side_effect=None):
    """A graphiti_service whose backend raises `side_effect` after a real delay."""

    async def _slow_boom(*_args, **_kwargs):
        await asyncio.sleep(_SLOW_FAILURE_SECONDS)
        raise side_effect

    mock_graph = AsyncMock()
    mock_graph.ro_query = AsyncMock(side_effect=_slow_boom)

    mock_driver = MagicMock()
    mock_driver._get_graph = MagicMock(return_value=mock_graph)
    mock_driver._database = 'test_db'
    mock_driver.execute_query = AsyncMock(side_effect=_slow_boom)

    mock_client = MagicMock()
    mock_client.driver = mock_driver

    svc = AsyncMock()
    svc.flavour = flavour
    svc.get_client = AsyncMock(return_value=mock_client)
    svc.config = MagicMock()
    svc.config.graphiti.group_id = 'test_graph'
    return svc


@pytest.mark.asyncio
async def test_a_failure_after_real_execution_reports_nonzero_execution_ms():
    """THE regression. The backend burns 50ms and then fails; the envelope must say so."""
    from graphiti_mcp_server import run_cypher

    svc = _mock_service(_FLAVOUR, side_effect=Exception("Unknown function 'foo'"))
    with patch('graphiti_mcp_server.graphiti_service', svc):
        result = await run_cypher(query='MATCH (n) RETURN foo(n) LIMIT 5')

    assert result['type'] == 'error'
    assert result['execution_ms'] >= _SLOW_FAILURE_MS * 0.8, result['execution_ms']


@pytest.mark.asyncio
async def test_the_age_catch_all_query_failed_path_also_reports_its_wall_clock():
    """`reason='query_failed'` is the unclassified AGE fallback — the path BUG-33
    named specifically, because everything that is not a recognised pattern lands
    there and it was the one most likely to be expensive."""
    from graphiti_mcp_server import run_cypher

    svc = _mock_service(AgeFlavour(), side_effect=Exception('something nobody has a pattern for'))
    with patch('graphiti_mcp_server.graphiti_service', svc):
        result = await run_cypher(query='MATCH (n) RETURN n.name LIMIT 5')

    assert result['error_detail']['reason'] == 'query_failed'
    assert result['execution_ms'] >= _SLOW_FAILURE_MS * 0.8, result['execution_ms']


@pytest.mark.asyncio
async def test_a_preflight_rejection_still_reports_zero():
    """The other half of the contract: a consumer can only trust a nonzero reading if
    zero still means "nothing ran"."""
    from graphiti_mcp_server import run_cypher

    svc = AsyncMock()
    svc.flavour = _FLAVOUR
    svc.config = MagicMock()
    svc.config.graphiti.group_id = 'test_graph'

    with patch('graphiti_mcp_server.graphiti_service', svc):
        result = await run_cypher(query='CREATE (n:Test {name: "test"})')

    assert result['error_detail']['stage'] == 'security'
    assert result['execution_ms'] == 0.0


@pytest.mark.asyncio
async def test_an_uninitialised_service_reports_zero():
    from graphiti_mcp_server import run_cypher

    with patch('graphiti_mcp_server.graphiti_service', None):
        result = await run_cypher(query='MATCH (n) RETURN n LIMIT 1')

    assert result['error_detail']['reason'] == 'service_not_ready'
    assert result['execution_ms'] == 0.0


@pytest.mark.asyncio
async def test_a_failure_before_the_query_runs_is_still_measured_not_asserted():
    """`get_client()` can itself be slow (connection setup, a retrying pool). That cost
    is real and belongs in the reading — what must never happen is a hardcoded 0."""
    from graphiti_mcp_server import run_cypher

    async def _slow_connect():
        await asyncio.sleep(_SLOW_FAILURE_SECONDS)
        raise Exception('could not reach the database')

    svc = AsyncMock()
    svc.flavour = _FLAVOUR
    svc.get_client = AsyncMock(side_effect=_slow_connect)
    svc.config = MagicMock()
    svc.config.graphiti.group_id = 'test_graph'

    with patch('graphiti_mcp_server.graphiti_service', svc):
        result = await run_cypher(query='MATCH (n) RETURN n LIMIT 1')

    assert result['type'] == 'error'
    assert result['execution_ms'] >= _SLOW_FAILURE_MS * 0.8, result['execution_ms']


# ---------------------------------------------------------------------------
# the asymmetry between the two envelopes
# ---------------------------------------------------------------------------


def _mock_service_with_slow_acquisition(flavour, *, rows):
    """A service whose get_client() is slow but whose query is instant."""

    async def _slow_get_client():
        await asyncio.sleep(_SLOW_FAILURE_SECONDS)
        result = MagicMock()
        result.header = [('string', 'cnt')]
        result.result_set = rows
        graph = AsyncMock()
        graph.ro_query = AsyncMock(return_value=result)
        driver = MagicMock()
        driver._get_graph = MagicMock(return_value=graph)
        driver._database = 'test_db'
        client = MagicMock()
        client.driver = driver
        return client

    svc = AsyncMock()
    svc.flavour = flavour
    svc.get_client = AsyncMock(side_effect=_slow_get_client)
    svc.config = MagicMock()
    svc.config.graphiti.group_id = 'test_graph'
    return svc


@pytest.mark.asyncio
async def test_a_successful_call_measures_the_query_not_the_connection_setup():
    """The success envelope's `execution_ms` means "how long the query took", and the
    BUG-33a fix must not have redefined it.

    Moving one clock above the `try` would have folded `get_client()` into every
    successful reading too — so the first call after a restart would look like a slow
    query rather than a cold pool, silently changing a field consumers already read
    (it feeds `cypher_quality` and any latency view built on it). Acquisition is noise
    on this path and signal on the failure path, so the two paths use two clocks.
    """
    from graphiti_mcp_server import run_cypher

    svc = _mock_service_with_slow_acquisition(_FLAVOUR, rows=[[47]])
    with patch('graphiti_mcp_server.graphiti_service', svc):
        result = await run_cypher(query='MATCH (n) RETURN count(n) AS cnt')

    assert result['type'] == 'scalar', result
    assert result['result'] == 47
    # 50ms of connection setup happened; none of it belongs in this number.
    assert result['execution_ms'] < _SLOW_FAILURE_MS * 0.5, result['execution_ms']
