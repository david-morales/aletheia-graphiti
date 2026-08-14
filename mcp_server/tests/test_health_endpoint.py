"""H-F1: `/health` must not stay green while the connector is degraded.

`/health` returned `{'status': 'healthy'}` unconditionally. The only real probe
lives in `get_status`, which is an MCP TOOL — reachable by an agent that has
already connected, and by nothing in the operational layer. So the container
HEALTHCHECK (`docker/Dockerfile.standalone`: `curl -f .../health`) reported a
connector serving static fallback descriptions as indistinguishable from a
healthy one, and the only recovery path there is (a restart) was never triggered.

That was already the visible half of BUG-50: "`get_schema` ... vanished from
`tools/list` while `/health` and `get_status` stayed green".
"""

from __future__ import annotations

import json

import pytest

import graphiti_mcp_server as srv
from flavours.falkordb import FalkorDbFlavour

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _restore_degraded_state():
    """`_degraded_reason` is process-global, like the registries it mirrors."""
    before = srv._degraded_reason
    try:
        yield
    finally:
        srv._degraded_reason = before


async def _probe() -> tuple[int, dict]:
    response = await srv.health_check(request=None)
    return response.status_code, json.loads(response.body)


async def test_a_healthy_connector_answers_200():
    srv._degraded_reason = None
    status, body = await _probe()
    assert status == 200
    assert body['status'] == 'healthy'
    assert body['service'] == 'graphiti-mcp'


async def test_a_degraded_connector_answers_503():
    """503, not 200: `curl -f` has to fail, or the HEALTHCHECK cannot see this.

    503 rather than 500 because the condition is "cannot serve this properly
    right now" and it is the status an orchestrator already treats as
    not-ready/unhealthy.
    """
    srv._degraded_reason = 'graph introspection exploded'
    status, body = await _probe()
    assert status == 503, body


async def test_the_degraded_body_is_machine_readable_and_names_the_cause():
    """An operator reading the probe output must not have to grep the logs."""
    srv._degraded_reason = 'graph introspection exploded'
    _, body = await _probe()
    assert body['status'] == 'degraded'
    assert body['service'] == 'graphiti-mcp'
    assert body['reason'] == 'graph introspection exploded'


async def test_the_real_degrade_path_sets_the_flag(monkeypatch):
    """Pinned against `register_fallback_tools` itself, not against the flag.

    A test that only sets `_degraded_reason` by hand proves the endpoint reads
    it, never that anything writes it — which is exactly how the endpoint came
    to be unconditional in the first place.
    """
    srv._degraded_reason = None
    monkeypatch.setattr(srv, 'config', srv.GraphitiConfig(), raising=False)
    monkeypatch.setattr(srv, 'graphiti_service', None)

    srv.register_fallback_tools(reason='graph introspection exploded')

    assert srv._degraded_reason == 'graph introspection exploded'
    status, body = await _probe()
    assert status == 503, body
    assert body['reason'] == 'graph introspection exploded'


async def test_a_healthy_startup_leaves_the_probe_green(monkeypatch):
    """The happy path must not inherit a degraded flag from a prior build."""
    srv._degraded_reason = 'stale'

    from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo

    class _StubClient:
        driver = object()

    class _StubService:
        flavour = FalkorDbFlavour()
        ontology_client = None
        domain_profile = None

        def __init__(self):
            self.config = srv.GraphitiConfig()

        async def get_client(self):
            return _StubClient()

    service = _StubService()
    monkeypatch.setattr(srv, 'graphiti_service', service)
    monkeypatch.setattr(srv, 'config', service.config, raising=False)

    async def _ok(*a, **k):
        return DomainProfile(
            group_id='healthy_graph',
            entity_types={'Widget': EntityTypeInfo('Widget', 4, 'A widget', ['W-1'])},
            edge_types={'USES': EdgeTypeInfo('USES', 2, 'uses', 'Widget -> Widget')},
            time_range=None,
        )

    monkeypatch.setattr(srv, 'build_domain_profile', _ok)
    await srv._build_and_register_domain_surface()

    assert srv._degraded_reason is None
    status, _ = await _probe()
    assert status == 200


async def test_a_successful_refresh_clears_a_degraded_probe(monkeypatch):
    """F5: recovery must be reachable, or 503 is a one-way door.

    `refresh_domain_surface` is the ONE path that rebuilds a served surface in
    a running process — it re-censuses, re-registers all nine dynamic tools and
    restores the three resources `register_fallback_tools` pruned. It rebuilt
    everything except the flag, so a connector that had recovered completely
    went on answering 503 for the life of the process. That turns the probe
    from a signal into a permanent verdict, and it is sharper than a stale
    field: a compose `service_healthy` dependency (F6, accepted) keeps
    dependents blocked on it.
    """
    from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo

    class _StubClient:
        driver = object()

    class _StubService:
        flavour = FalkorDbFlavour()
        ontology_client = None
        domain_profile = None

        def __init__(self):
            self.config = srv.GraphitiConfig()

        async def get_client(self):
            return _StubClient()

    service = _StubService()
    monkeypatch.setattr(srv, 'graphiti_service', service)
    monkeypatch.setattr(srv, 'config', service.config, raising=False)

    # Degrade first, through the real path, and prove the probe reports it.
    srv.register_fallback_tools(reason='graph introspection exploded')
    assert (await _probe())[0] == 503

    async def _ok(*a, **k):
        return DomainProfile(
            group_id='recovered_graph',
            entity_types={'Widget': EntityTypeInfo('Widget', 4, 'A widget', ['W-1'])},
            edge_types={'USES': EdgeTypeInfo('USES', 2, 'uses', 'Widget -> Widget')},
            time_range=None,
        )

    monkeypatch.setattr(srv, 'build_domain_profile', _ok)
    assert await srv.refresh_domain_surface(reason='test recovery') is True

    assert srv._degraded_reason is None
    status, body = await _probe()
    assert status == 200, body
    assert body['status'] == 'healthy'


async def test_a_FAILED_refresh_leaves_the_degraded_probe_standing(monkeypatch):
    """The other direction: a refresh that could not rebuild must not clear the
    flag. `refresh_domain_surface` restores the surface in force on failure and
    deliberately does not fall through to the fallback path — so the connector
    is still exactly as degraded as it was, and the probe must keep saying so.
    """
    class _StubClient:
        driver = object()

    class _StubService:
        flavour = FalkorDbFlavour()
        ontology_client = None
        domain_profile = None

        def __init__(self):
            self.config = srv.GraphitiConfig()

        async def get_client(self):
            return _StubClient()

    service = _StubService()
    monkeypatch.setattr(srv, 'graphiti_service', service)
    monkeypatch.setattr(srv, 'config', service.config, raising=False)

    srv.register_fallback_tools(reason='graph introspection exploded')

    async def _boom(*a, **k):
        raise RuntimeError('still unreachable')

    monkeypatch.setattr(srv, 'build_domain_profile', _boom)
    assert await srv.refresh_domain_surface(reason='test failed recovery') is False

    assert srv._degraded_reason == 'graph introspection exploded'
    assert (await _probe())[0] == 503
