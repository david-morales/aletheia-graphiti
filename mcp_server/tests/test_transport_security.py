"""The DNS rebinding policy must stay keyed to FASTMCP_HOST, not to the bind host.

WHY THIS MODULE EXISTS

SDK 1.x decided DNS rebinding protection ONCE, in the `FastMCP` constructor, from
the host it was constructed with — which this server read from `FASTMCP_HOST`.
The host it actually BOUND was set later and separately (`mcp.settings.host`, from
`config.server.host`, which defaults to `0.0.0.0`). Those are two different values
in every default deployment, and only the first one chose the policy.

SDK 2.x moved transport settings to `run_*_async()` and kept the localhost
auto-enable rule — but it now evaluates that rule against the host being bound.
Taking the new default would therefore have flipped the policy for every
deployment that binds `0.0.0.0` while leaving `FASTMCP_HOST` unset: protection ON
under 1.x, silently OFF under 2.x. Nothing in the suite would have noticed, and
the symptom in production is not an error — it is a security control quietly
absent.

WHAT `None` ACTUALLY MEANS (measured, not assumed)

`_transport_security()` returning `None` does NOT mean "protection off". It means
"no explicit policy", and the SDK then applies its own rule to the BIND host
(`mcp/server/mcpserver/server.py`: `if transport_security is None and host in
("127.0.0.1", "localhost", "::1")`). Driving a real served app with a Docker
service name in the Host header, across both bind shapes:

    _transport_security()      bind host    Host: <docker-service>:8000   result
    None      (FASTMCP_HOST=0.0.0.0)  0.0.0.0    accepted                 200
    None      (FASTMCP_HOST=0.0.0.0)  127.0.0.1  refused                  421
    SETTINGS  (FASTMCP_HOST loopback) 0.0.0.0    refused                  421
    SETTINGS  (FASTMCP_HOST loopback) 127.0.0.1  refused                  421

So the deployment shape is the pairing that matters, and the compose overlays set
BOTH halves of it (`FASTMCP_HOST=0.0.0.0` and `SERVER__HOST=0.0.0.0`). Row 2 is
the trap: setting only `FASTMCP_HOST` while binding loopback still refuses
inter-container traffic. That is the SDK's behaviour, not this server's, and it is
pinned here so it is discovered by a test rather than by a connector that cannot
talk to its own stack.

These tests therefore assert the SERVED policy — a real ASGI round trip through
the transport-security middleware — plus the wiring that carries the decision to
the transport at all. Asserting the config object alone would pass with the
`transport_security=` kwarg deleted from the `run_*` calls.
"""

from __future__ import annotations

import httpx
import pytest

import graphiti_mcp_server as srv
from config.schema import ServerConfig

pytestmark = pytest.mark.contract

# What inter-container traffic actually sends: a Docker service name, which is in
# no loopback allowlist. This is the exact header the compose network produces.
DOCKER_HOST_HEADER = 'aletheia-connector-pp-bench-falkor:8000'

_INITIALIZE = (
    b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":'
    b'{"protocolVersion":"2025-11-25","capabilities":{},'
    b'"clientInfo":{"name":"transport-security-probe","version":"1"}}}'
)


async def _serve(bind_host: str, host_header: str) -> int:
    """Round-trip `initialize` through the REAL served app and return the status.

    Built exactly as `run_mcp_server` builds it: the policy from
    `_transport_security()`, the bind host passed alongside it.
    """
    app = srv.mcp.streamable_http_app(
        transport_security=srv._transport_security(), host=bind_host
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url=f'http://{host_header}'
        ) as client:
            response = await client.post(
                '/mcp',
                content=_INITIALIZE,
                headers={
                    'content-type': 'application/json',
                    'accept': 'application/json, text/event-stream',
                },
            )
    return response.status_code


# ---------------------------------------------------------------------------
# The served policy
# ---------------------------------------------------------------------------

async def test_the_deployment_shape_serves_inter_container_traffic(monkeypatch):
    """THE case the Docker overlays depend on.

    `FASTMCP_HOST=0.0.0.0` (so no explicit policy) paired with a `0.0.0.0` bind is
    what the compose files configure, and a request carrying a Docker service name
    in the Host header must be served. If this fails the connector cannot answer
    the workbench sitting next to it on the same network.
    """
    monkeypatch.setattr(srv, '_init_host', '0.0.0.0')
    assert await _serve('0.0.0.0', DOCKER_HOST_HEADER) == 200


@pytest.mark.parametrize('host', ['127.0.0.1', 'localhost', '::1'])
async def test_a_loopback_env_refuses_a_foreign_host_header(monkeypatch, host):
    """The protected shape, asserted where it counts: on the wire, not on a config
    object. An explicit policy refuses the Docker header on ANY bind host."""
    monkeypatch.setattr(srv, '_init_host', host)
    assert await _serve('0.0.0.0', DOCKER_HOST_HEADER) == 421
    assert await _serve('127.0.0.1', DOCKER_HOST_HEADER) == 421


async def test_a_loopback_env_still_serves_loopback_callers(monkeypatch):
    """The control for the two above: the protected policy is not refusing
    everything. A caller that IS on the allowlist gets through."""
    monkeypatch.setattr(srv, '_init_host', '127.0.0.1')
    assert await _serve('127.0.0.1', '127.0.0.1:8000') == 200


async def test_no_explicit_policy_plus_a_loopback_BIND_still_refuses(monkeypatch):
    """The trap, pinned as SDK behaviour rather than as this server's intent.

    `FASTMCP_HOST=0.0.0.0` alone is NOT enough: with no explicit policy the SDK
    applies its own rule to the BIND host, so binding loopback re-enables
    protection and inter-container traffic is refused. Both halves of the
    deployment shape are load-bearing, which is why the compose overlays set
    `SERVER__HOST=0.0.0.0` as well as `FASTMCP_HOST=0.0.0.0`.
    """
    monkeypatch.setattr(srv, '_init_host', '0.0.0.0')
    assert srv._transport_security() is None, 'premise: no explicit policy here'
    assert await _serve('127.0.0.1', DOCKER_HOST_HEADER) == 421, (
        'the SDK stopped auto-enabling protection on a loopback bind — the '
        '`None` return from _transport_security() no longer defers to that rule, '
        'so re-check what an unset FASTMCP_HOST now means'
    )


# ---------------------------------------------------------------------------
# ...and the decision actually reaches the transport
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('transport', ['http', 'sse'])
async def test_the_run_call_carries_the_policy(monkeypatch, transport):
    """Deleting `transport_security=` from either `run_*` call must fail here.

    Without this, every assertion above could pass while the server shipped the
    SDK's bind-host default — the exact regression this module exists to prevent.
    """
    seen: dict[str, object] = {}

    async def _fake_run(**kwargs):
        seen.update(kwargs)

    target = 'run_streamable_http_async' if transport == 'http' else 'run_sse_async'
    monkeypatch.setattr(srv.mcp, target, _fake_run)
    monkeypatch.setattr(srv, 'configure_uvicorn_logging', lambda: None)
    monkeypatch.setattr(srv, '_init_host', '127.0.0.1')

    async def _cfg():
        return ServerConfig(transport=transport, host='0.0.0.0', port=8123)

    monkeypatch.setattr(srv, 'initialize_server', _cfg)
    await srv.run_mcp_server()

    assert 'transport_security' in seen, (
        f'{target} was called without transport_security — the server falls back '
        f'to the SDK bind-host rule and FASTMCP_HOST stops deciding anything'
    )
    policy = seen['transport_security']
    assert policy is not None and policy.enable_dns_rebinding_protection is True, (
        'a loopback FASTMCP_HOST must reach the transport as an explicit protective '
        'policy, even though the server binds 0.0.0.0'
    )


def test_the_bind_host_cannot_change_the_policy(monkeypatch):
    """The decision is a function of FASTMCP_HOST alone.

    `config.server.host` (default `0.0.0.0`) is what gets BOUND and is passed
    separately; it must not feed the policy. The two disagree in the default
    deployment, which is precisely the row the SDK 2.x default would have flipped.
    """
    assert ServerConfig().host == '0.0.0.0', (
        'the premise of this test is that the default BIND host is not loopback'
    )
    monkeypatch.setattr(srv, '_init_host', '127.0.0.1')
    assert srv._transport_security() is not None
    monkeypatch.setattr(srv, '_init_host', '0.0.0.0')
    assert srv._transport_security() is None


@pytest.mark.parametrize('host', ['127.0.0.1', 'localhost', '::1'])
def test_a_loopback_env_produces_the_documented_allowlists(monkeypatch, host):
    monkeypatch.setattr(srv, '_init_host', host)
    settings = srv._transport_security()
    assert settings is not None
    assert settings.enable_dns_rebinding_protection is True
    assert settings.allowed_hosts == ['127.0.0.1:*', 'localhost:*', '[::1]:*']
    assert settings.allowed_origins == [
        'http://127.0.0.1:*',
        'http://localhost:*',
        'http://[::1]:*',
    ]


def test_the_default_when_the_env_is_unset_is_protected(monkeypatch):
    """`_init_host` defaults to 127.0.0.1, so an unset env is the protected case."""
    monkeypatch.delenv('FASTMCP_HOST', raising=False)
    monkeypatch.setattr(srv, '_init_host', '127.0.0.1')
    assert srv._transport_security() is not None
