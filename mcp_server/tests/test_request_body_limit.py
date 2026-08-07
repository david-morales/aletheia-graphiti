"""SDK 2.x caps streamable-HTTP request bodies. 1.x did not. This restores 1.x.

WHY THIS MODULE EXISTS

SDK 1.x had no request-body limit of any kind — `max_request_body_size` does not
exist anywhere in the 1.26.0 source. SDK 2.x introduced one and defaults it to
4 MiB (`mcp/server/streamable_http_manager.py`, `DEFAULT_MAX_REQUEST_BODY_SIZE`),
enforced by `RequestBodyLimitMiddleware`: a POST whose `content-length` exceeds
the cap is answered `413 Request body too large` before the ASGI app is invoked —
so it never reaches the MCP layer, and the caller does not get an in-band
ADR-015 R4 error, it gets a transport failure.

That is a SECOND wire delta hiding inside the bump, and a silent one: nothing in
the tool surface changes, `tools/list` is identical, and every offline test stays
green. It only shows up on a large `add_memory` — exactly the bulk-ingest call
most likely to carry megabytes of episode text.

This is a migration, not a redesign, so the default restores 1.x behaviour. There
is no unlimited sentinel to ask for — the SDK types the parameter `int` and
rejects `<= 0` outright (`max_request_body_size must be a positive number of
bytes`) — so "no cap" is expressed as a deliberately large bound, and it is
surfaced as a real knob rather than a constant so an operator who WANTS the 4 MiB
protection can set it back without editing source.

The tests below drive the real ASGI stack the server serves, so the middleware is
genuinely in the path; the control case proves the cap is real by watching the
SDK default reject the same body this server accepts.
"""

from __future__ import annotations

import httpx
import pytest

import graphiti_mcp_server as srv
from config.schema import DEFAULT_MAX_REQUEST_BODY_SIZE, ServerConfig

pytestmark = pytest.mark.contract

SDK_DEFAULT_CAP = 4 * 1024 * 1024
BODY_BYTES = 5 * 1024 * 1024  # comfortably over the SDK cap, under ours


def _oversized_payload() -> bytes:
    """A well-formed JSON-RPC request whose body exceeds the SDK cap.

    `initialize` rather than `add_memory`: the claim under test is that a >4 MiB
    POST reaches the MCP layer AND is answered, and `initialize` is the request
    that can be answered without a live graph. A tool call would come back as an
    in-band ADR-015 R4 error here, which cannot distinguish "the transport let it
    through" from "the transport ate it".
    """
    return (
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":'
        '{"protocolVersion":"2025-11-25","capabilities":{},'
        f'"clientInfo":{{"name":"body-limit-probe","version":"1"}},"_pad":"{"a" * BODY_BYTES}"}}}}'
    ).encode()


async def _post(app, body: bytes) -> tuple[int, str]:
    """POST straight at the served ASGI app; return (status, body text).

    The app's lifespan is entered explicitly: `streamable_http_app()` starts its
    session manager there, and ASGITransport does not run lifespan events. Without
    it the request still passes the body-limit middleware — which is what this
    module measures — but then dies on an uninitialised task group, which would
    make a PASS indistinguishable from a crash.
    """
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url='http://testserver'
        ) as client:
            response = await client.post(
                '/mcp',
                content=body,
                headers={
                    'content-type': 'application/json',
                    'accept': 'application/json, text/event-stream',
                },
            )
    return response.status_code, response.text


def _app(max_body: int):
    """The app in its DEPLOYED shape.

    `host='0.0.0.0'` matches what the compose overlays run (`FASTMCP_HOST=0.0.0.0`,
    `SERVER__HOST=0.0.0.0`) and is load-bearing for this module: on a loopback host
    the SDK auto-enables DNS rebinding protection, and the probe's `Host:
    testserver` would be refused `421 Invalid Host header` BEFORE the body limit
    could be observed. Measuring the cap requires a request that is otherwise
    allowed to succeed.
    """
    return srv.mcp.streamable_http_app(max_request_body_size=max_body, host='0.0.0.0')


async def test_the_sdk_default_would_reject_a_bulk_body():
    """The control. Without this, the test below could pass on an SDK that never
    capped anything, and the guard would be measuring nothing."""
    status, text = await _post(_app(SDK_DEFAULT_CAP), _oversized_payload())
    assert status == 413, (
        f'the SDK 4 MiB cap is no longer enforced (HTTP {status}) — if the SDK '
        f"dropped it, this server's large default is no longer load-bearing"
    )
    assert 'too large' in text


async def test_our_configured_limit_accepts_a_bulk_body():
    """THE regression: the same body the SDK default rejects must be ANSWERED.

    Not merely "not 413" — a 200 carrying a JSON-RPC result, which is the only
    evidence that the request travelled the whole way rather than dying one layer
    further in.
    """
    status, text = await _post(_app(ServerConfig().max_request_body_size), _oversized_payload())
    assert status == 200, (
        f'a {BODY_BYTES // 1024 // 1024} MiB request body was refused (HTTP {status}) — '
        f'SDK 1.x answered it and this migration must not change that:\n{text[:200]}'
    )
    assert '"result"' in text, f'reached the server but was not answered:\n{text[:200]}'


def test_the_default_restores_pre_2x_headroom():
    assert DEFAULT_MAX_REQUEST_BODY_SIZE > SDK_DEFAULT_CAP
    assert ServerConfig().max_request_body_size == DEFAULT_MAX_REQUEST_BODY_SIZE


def test_the_limit_is_operator_configurable(monkeypatch):
    """Wired through `ServerConfig`, so it reaches the server the same way host and
    port do: `SERVER__MAX_REQUEST_BODY_SIZE` in the environment, or `server:` in the
    YAML. An operator who wants the SDK's 4 MiB protection can ask for it."""
    assert ServerConfig(max_request_body_size=SDK_DEFAULT_CAP).max_request_body_size == (
        SDK_DEFAULT_CAP
    )


def test_a_non_positive_limit_is_rejected_at_config_time():
    """The SDK raises on `<= 0` deep inside `run()`, long after startup logging has
    claimed success. Failing at config parse turns that into a startup error that
    names the field."""
    with pytest.raises(ValueError):
        ServerConfig(max_request_body_size=0)


async def test_the_transport_actually_receives_the_configured_limit(monkeypatch):
    """The wiring, not the value. Deleting the kwarg from the `run_*` call must fail
    a test — otherwise the knob above is decorative."""
    seen: dict[str, object] = {}

    async def _fake_run(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(srv.mcp, 'run_streamable_http_async', _fake_run)
    monkeypatch.setattr(
        srv, 'initialize_server', _returning(ServerConfig(transport='http', port=8123))
    )
    monkeypatch.setattr(srv, 'configure_uvicorn_logging', lambda: None)

    await srv.run_mcp_server()

    assert seen.get('max_request_body_size') == DEFAULT_MAX_REQUEST_BODY_SIZE, (
        'run_streamable_http_async was called without the configured body limit, so '
        'the server silently falls back to the SDK 4 MiB cap'
    )


def _returning(value):
    async def _fn():
        return value

    return _fn
