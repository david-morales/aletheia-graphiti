"""The fork's live CI gate: the served surface, over a real wire, against FalkorDB.

`.github/workflows/mcp-server-tests.yml` runs this module with `-m integration` and it
is the ONLY job in the fork that executes the connector end to end. Until wave 7 it
tested a surface that had not existed for several releases — it asserted `search_nodes`,
`summarize_saga`, `add_triplet` and `get_episode_entities`, none of which are served —
so it was red at every commit, and a job that is always red gates nothing.

Rewritten against the CURRENT 18-tool catalog (`tool_annotations.TOOL_ORDER`), on the
SDK-2 `Client`, over BOTH transports the connector ships: stdio (what an agent host
spawns) and streamable HTTP (what every deployed connector serves).

The server subprocess is launched with `sys.executable`, not `uv run`, so it is
guaranteed to be the same interpreter and the same checked-out tree the tests import
from. Under CI that is the `uv sync` venv, so the CI path is unchanged; locally it means
a change to `mcp_server/src` is exercised without an install step. Override with
`MCP_SERVER_PYTHON` if the server must run somewhere else.

Skipped unless an OpenAI key is available AND `FALKORDB_URI` is set AND that endpoint
answers. `FALKORDB_URI` has NO default: this suite ingests episodes and calls
clear_graph, so it must never discover a store nobody pointed it at.

    # local, against a THROWAWAY FalkorDB — never the reserved 6379
    docker run -d --name ax_falkor -p 127.0.0.1:16379:6379 \
        -e FALKORDB_ARGS="TIMEOUT 0" falkordb/falkordb:v4.18.0
    FALKORDB_URI=redis://localhost:16379 \
        python -m pytest tests/test_live_falkordb_int.py -m integration

    # CI: OPENAI_API_KEY from the GitHub environment, FALKORDB_URI from the workflow,
    # FalkorDB as a container.

`MODEL_NAME` picks the model (CI pins a light, broadly-available one).
"""

import asyncio
import json
import os
import socket
import sys
import time
from contextlib import AsyncExitStack, closing, suppress
from pathlib import Path
from typing import Any

import httpx
import pytest
from dotenv import load_dotenv
from mcp import Client, StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

# Load the project-root .env for local runs (CI provides the env directly).
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / '.env')

MCP_SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MCP_SERVER_DIR / 'src'))

from tool_annotations import TOOL_ORDER  # noqa: E402
from version import CONNECTOR_VERSION  # noqa: E402

# NO DEFAULT, deliberately. `redis://localhost:6379` used to be the fallback, from an
# era when this module was red at every commit and could not reach a server anyway.
# Now that it works, that default is a loaded gun: on a developer machine 6379 is a
# real, populated FalkorDB, and this suite INGESTS episodes and calls clear_graph. A
# forgotten env var must produce a skip, never a write to somebody's graph store.
FALKORDB_URI = os.environ.get('FALKORDB_URI', '')
SERVER_PYTHON = os.environ.get('MCP_SERVER_PYTHON', sys.executable)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_falkordb,
    pytest.mark.requires_openai,
]


def _falkordb_reachable(uri: str) -> bool:
    """Best-effort TCP check that a FalkorDB (Redis) endpoint is reachable."""
    host, port = 'localhost', 6379
    if '://' in uri:
        rest = uri.split('://', 1)[1].split('@')[-1]  # strip scheme + any credentials
        hostport = rest.split('/')[0]
        if ':' in hostport:
            host_part, port_part = hostport.rsplit(':', 1)
            host = host_part or host
            with suppress(ValueError):
                port = int(port_part)
        elif hostport:
            host = hostport
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


# Skip the whole module unless prerequisites are present, so the suite is a no-op
# locally without setup and on fork PRs (which have no secrets).
if not os.environ.get('OPENAI_API_KEY'):
    pytest.skip('OPENAI_API_KEY not set; skipping live MCP tests', allow_module_level=True)
if not FALKORDB_URI:
    pytest.skip(
        'FALKORDB_URI is not set; skipping live MCP tests. Point it at a THROWAWAY '
        'FalkorDB (this suite ingests episodes and calls clear_graph) — there is no '
        'default, so it can never find a real store by accident.',
        allow_module_level=True,
    )
if not _falkordb_reachable(FALKORDB_URI):
    pytest.skip(
        f'FalkorDB not reachable at {FALKORDB_URI}; skipping live MCP tests',
        allow_module_level=True,
    )


def _unique_group_id() -> str:
    # Alphanumeric only (no '_'/'-'): keeps the RediSearch fulltext group filter
    # valid without relying on group_id escaping, so the suite is backend-portable.
    return f'livetest{int(time.time())}{os.getpid()}'


def _free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def _server_env(group_id: str) -> dict[str, str]:
    return {
        **os.environ,
        'GRAPHITI_GROUP_ID': group_id,
        'FALKORDB_URI': FALKORDB_URI,
    }


def _payload(result: Any) -> Any:
    """The tool's JSON payload, or the raw text when it is not JSON."""
    content = getattr(result, 'content', None) or []
    text = getattr(content[0], 'text', None) if content else None
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _raise_on_error(tool: str, resp: Any) -> Any:
    """Surface a tool's error response immediately instead of masking it as empty.

    Otherwise a real server/LLM/schema error only shows up as a generic
    'not processed within the timeout' after the full poll, hiding the cause.
    """
    if isinstance(resp, dict) and resp.get('error'):
        raise AssertionError(f'{tool} returned an error: {resp["error"]}')
    return resp


class LiveMCPClient:
    """Drives the MCP server over one of its real transports.

    SDK 2.x: ONE `Client` wrapping a transport replaces the 1.x transport +
    `ClientSession` + `initialize()` layering. The layered form does not merely warn
    under 2.x — a bare `ClientSession` raises `send_raw_request called before run()` on
    the first call, which is exactly what six legacy modules in this directory were
    doing while looking green (BUG-60).
    """

    def __init__(self, group_id: str, transport: str = 'stdio'):
        self.group_id = group_id
        self.transport = transport
        self._stack = AsyncExitStack()
        self._process: asyncio.subprocess.Process | None = None
        self.session: Client | None = None

    async def __aenter__(self) -> 'LiveMCPClient':
        if self.transport == 'stdio':
            params = StdioServerParameters(
                command=SERVER_PYTHON,
                args=[str(MCP_SERVER_DIR / 'main.py'), '--transport', 'stdio'],
                env=_server_env(self.group_id),
                cwd=str(MCP_SERVER_DIR),
            )
            self.session = await self._stack.enter_async_context(Client(stdio_client(params)))
            return self

        if self.transport != 'http':
            raise ValueError(f'unsupported transport {self.transport!r}')

        port = _free_port()
        env = {
            **_server_env(self.group_id),
            'SERVER__HOST': '127.0.0.1',
            'SERVER__PORT': str(port),
        }
        self._process = await asyncio.create_subprocess_exec(
            SERVER_PYTHON,
            str(MCP_SERVER_DIR / 'main.py'),
            '--transport',
            'http',
            env=env,
            cwd=str(MCP_SERVER_DIR),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        url = f'http://127.0.0.1:{port}/mcp/'
        await self._await_http(url, port)
        self.session = await self._stack.enter_async_context(Client(streamable_http_client(url)))
        return self

    async def _await_http(self, url: str, port: int, timeout: float = 120.0) -> None:
        """Wait for the endpoint to answer at all.

        The server runs a full Graphiti init (indices, domain profile) BEFORE it binds,
        so a fixed sleep either flakes or wastes the whole budget. A dead subprocess is
        reported as itself rather than as a timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise AssertionError(
                    f'server exited with {self._process.returncode} before binding {port}'
                )
            try:
                async with httpx.AsyncClient(timeout=2.0) as http:
                    await http.get(url)
                return
            except httpx.HTTPError:
                await asyncio.sleep(1.0)
        raise AssertionError(f'server did not answer on {url} within {timeout}s')

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=10)
            if self._process.returncode is None:
                self._process.kill()
                await self._process.wait()

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        assert self.session is not None
        return _payload(await self.session.call_tool(tool, arguments))

    async def list_tool_names(self) -> list[str]:
        assert self.session is not None
        return [tool.name for tool in (await self.session.list_tools()).tools]

    async def wait_for_episodes(
        self, expected: int = 1, timeout: float = 180.0, poll: float = 3.0
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = _raise_on_error(
                'get_episodes',
                await self.call(
                    'get_episodes', {'group_ids': [self.group_id], 'max_episodes': 50}
                ),
            )
            episodes = resp.get('episodes', []) if isinstance(resp, dict) else []
            if len(episodes) >= expected:
                return episodes
            await asyncio.sleep(poll)
        return []

    async def search_until(
        self,
        tool: str,
        arguments: dict[str, Any],
        key: str,
        attempts: int = 6,
        poll: float = 3.0,
    ) -> list[Any]:
        """Call a search tool, retrying until ``resp[key]`` is non-empty (index lag)."""
        results: list[Any] = []
        for _ in range(attempts):
            resp = _raise_on_error(tool, await self.call(tool, arguments))
            results = resp.get(key, []) if isinstance(resp, dict) else []
            if results:
                return results
            await asyncio.sleep(poll)
        return results


# ---------------------------------------------------------------------------
# the announced surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('transport', ['stdio', 'http'])
async def test_the_server_announces_the_canonical_catalog_in_its_canonical_order(transport):
    """The catalog and its ORDER, on the wire, on both transports.

    Order is part of the contract (ADR-019, and the 2026-07-28 spec SHOULD that list
    endpoints must not vary per connection): the server instructions number the same
    capabilities, and a client's prompt cache is invalidated by a reshuffle. Asserting
    the list rather than a set is what makes `apply_canonical_tool_order` observable
    from outside the process.
    """
    async with LiveMCPClient(_unique_group_id(), transport=transport) as client:
        assert await client.list_tool_names() == list(TOOL_ORDER)


@pytest.mark.parametrize('transport', ['stdio', 'http'])
async def test_get_status_reports_a_healthy_backend_and_this_build(transport):
    """A-D11: the connector says which build it is, and the wire agrees with the package."""
    async with LiveMCPClient(_unique_group_id(), transport=transport) as client:
        status = await client.call('get_status', {})
        assert isinstance(status, dict), status
        assert status.get('status') == 'ok', status
        assert status.get('version') == CONNECTOR_VERSION, status


async def test_every_tool_declares_annotations_and_an_output_schema_on_the_wire():
    """ADR-019 R3/R2 as a CLIENT sees them, not as the registry holds them."""
    async with LiveMCPClient(_unique_group_id()) as client:
        assert client.session is not None
        tools = {t.name: t for t in (await client.session.list_tools()).tools}
        missing_annotations = sorted(n for n, t in tools.items() if t.annotations is None)
        assert not missing_annotations, missing_annotations
        missing_schemas = sorted(n for n, t in tools.items() if not t.output_schema)
        assert not missing_schemas, missing_schemas


# ---------------------------------------------------------------------------
# run_cypher: the envelope contract, over the wire
# ---------------------------------------------------------------------------


async def test_run_cypher_round_trips_a_scalar_envelope():
    async with LiveMCPClient(_unique_group_id()) as client:
        resp = _raise_on_error(
            'run_cypher',
            await client.call('run_cypher', {'query': 'MATCH (n) RETURN count(n) AS c'}),
        )
        assert resp['type'] == 'scalar', resp
        assert isinstance(resp['result'], int), resp
        assert resp['execution_ms'] > 0, resp


async def test_a_write_query_is_rejected_before_it_reaches_the_database():
    async with LiveMCPClient(_unique_group_id()) as client:
        resp = await client.call('run_cypher', {'query': 'CREATE (n:ShouldNeverExist)'})
        assert resp['type'] == 'error', resp
        assert resp['error_detail']['stage'] == 'security', resp
        assert isinstance(resp['error'], str), resp  # ADR-015 R4


async def test_an_execution_failure_reports_the_wall_clock_a_rejection_does_not():
    """BUG-33(a), asserted OVER THE WIRE rather than in-process.

    Both calls come back as error envelopes. The one the validator rejected never
    touched the database and reports 0; the one FalkorDB accepted, planned and then
    failed reports the round trip it actually cost. Before the fix both read 0 — which
    is why the workbench breaker classified every expensive failure as cheap.

    Stated as a RELATIVE claim on purpose: `executed > rejected == 0` is the semantic
    consumers need, and unlike an absolute threshold it cannot be satisfied by a clock
    that happens to be coarse.
    """
    async with LiveMCPClient(_unique_group_id()) as client:
        rejected = await client.call('run_cypher', {'query': 'CREATE (n:ShouldNeverExist)'})
        executed = await client.call(
            'run_cypher', {'query': 'MATCH (n) RETURN nosuchfunction(n) AS c LIMIT 5'}
        )

    assert rejected['error_detail']['stage'] == 'security', rejected
    assert rejected['execution_ms'] == 0, rejected

    assert executed['type'] == 'error', executed
    assert executed['error_detail']['stage'] == 'execution', executed
    assert executed['execution_ms'] > rejected['execution_ms'], (executed, rejected)


async def test_get_schema_answers_the_canonical_shape():
    """ADR-019 R5's canonical `get_schema` keys, read off the wire.

    `node_labels`/`relationship_types`, not `entity_types`/`edge_types` — the names are
    the contract every consumer's schema reader is written against, and the previous
    version of this module asserted a set of tools that had already been renamed once.
    """
    async with LiveMCPClient(_unique_group_id()) as client:
        schema = _raise_on_error('get_schema', await client.call('get_schema', {}))
        for key in (
            'type',
            'graph_name',
            'dialect',
            'dialect_reference',
            'node_labels',
            'relationship_types',
            'tool_capabilities',
        ):
            assert key in schema, f'get_schema is missing {key}: {sorted(schema)}'
        assert schema['type'] == 'schema', schema['type']


# ---------------------------------------------------------------------------
# the ingestion round trip (this is the part that needs the LLM)
# ---------------------------------------------------------------------------


async def test_end_to_end_add_search_context_delete_clear():
    group = _unique_group_id()
    async with LiveMCPClient(group) as client:
        try:
            add = _raise_on_error(
                'add_memory',
                await client.call(
                    'add_memory',
                    {
                        'name': 'Live Test Episode',
                        'episode_body': (
                            'Alice is a software engineer at Acme Corporation. '
                            'She works on the Graphiti project.'
                        ),
                        'source': 'text',
                        'source_description': 'live integration test',
                        'group_id': group,
                    },
                ),
            )
            assert isinstance(add, dict) and 'message' in add, f'add_memory: {add}'

            episodes = await client.wait_for_episodes(expected=1)
            assert episodes, 'episode was not processed within the timeout'
            assert any(e.get('group_id') == group for e in episodes)
            episode_uuid = episodes[0]['uuid']

            # `search` replaces the 1.x search_nodes/search_memory_facts pair: ONE tool,
            # `search_mode` selecting the projection.
            nodes = await client.search_until(
                'search',
                {'query': 'Alice Acme Graphiti', 'group_ids': [group], 'search_mode': 'nodes'},
                key='nodes',
            )
            assert nodes, 'expected at least one extracted entity node'

            # Facts, asserted from the GRAPH rather than from `search(mode='edges')`.
            #
            # Deliberate, and not a weaker assertion: on the FalkorDB flavour
            # `search(search_mode='edges')` cannot return anything at all. The FalkorDB
            # bulk writer MERGEs each edge under its relation NAME as the database
            # relationship type (`utils/bulk_utils.py`, the FALKORDB branch), while
            # `edge_fulltext_search` defaults its lookup to `['RELATES_TO']`
            # (`search/search_utils.py`) — a type that path never writes. Measured
            # 2026-08-08 against the live bench connectors: FalkorDB returns 0 edges
            # from a graph holding 1,532 typed ones, the AGE arm returns 5. Creating
            # the missing per-type fulltext indices does NOT change it; the query
            # itself only looks at RELATES_TO.
            #
            # Asserting `search edges` here would pin a broken behaviour as expected.
            # Asserting the graph pins what the episode actually produced, and stays
            # true once the search defect is fixed.
            facts = _raise_on_error(
                'run_cypher',
                await client.call(
                    'run_cypher',
                    {'query': 'MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c'},
                ),
            )
            assert facts['result'] > 0, f'episode extracted no entity-to-entity edges: {facts}'

            # `get_episode_context` replaces get_episode_entities: what did THIS episode
            # put in the graph?
            context = _raise_on_error(
                'get_episode_context',
                await client.call('get_episode_context', {'episode_uuids': [episode_uuid]}),
            )
            assert context.get('nodes'), f'episode extracted no nodes: {context}'

            # explore_node walks out from a name the search just returned.
            explored = _raise_on_error(
                'explore_node',
                await client.call(
                    'explore_node',
                    {
                        'node_name': nodes[0]['name'],
                        'group_ids': [group],
                        'depth': 1,
                        'limit': 5,
                    },
                ),
            )
            assert explored.get('center_node'), f'explore_node found no centre: {explored}'

            deleted = _raise_on_error(
                'delete_episode', await client.call('delete_episode', {'uuid': episode_uuid})
            )
            assert 'message' in deleted, f'delete_episode: {deleted}'

            cleared = _raise_on_error(
                'clear_graph', await client.call('clear_graph', {'group_ids': [group]})
            )
            assert 'message' in cleared, f'clear_graph: {cleared}'

            remaining = _raise_on_error(
                'get_episodes',
                await client.call('get_episodes', {'group_ids': [group], 'max_episodes': 50}),
            )
            assert not remaining.get('episodes'), f'clear_graph left episodes behind: {remaining}'
        finally:
            # Always remove this run's data, even if an assertion above failed, so a
            # long-lived local FalkorDB doesn't accumulate orphaned test groups.
            with suppress(Exception):
                await client.call('clear_graph', {'group_ids': [group]})
