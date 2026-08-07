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

So `_transport_security()` reimplements the 1.x rule against `FASTMCP_HOST` and
passes the result explicitly. These tests pin BOTH halves of that: the policy the
env asks for, and the fact that the bind host cannot influence it.

The Docker case is the one that matters operationally: the compose overlays set
`FASTMCP_HOST=0.0.0.0` precisely so inter-container requests carrying Docker
hostnames in the Host header are accepted. If that stopped working the connector
would reject its own stack.
"""

from __future__ import annotations

import pytest

import graphiti_mcp_server as srv

pytestmark = pytest.mark.contract


@pytest.mark.parametrize('host', ['127.0.0.1', 'localhost', '::1'])
def test_a_loopback_host_keeps_dns_rebinding_protection(monkeypatch, host):
    monkeypatch.setattr(srv, '_init_host', host)
    settings = srv._transport_security()
    assert settings is not None, f'{host} must keep DNS rebinding protection'
    assert settings.enable_dns_rebinding_protection is True
    assert settings.allowed_hosts == ['127.0.0.1:*', 'localhost:*', '[::1]:*']
    assert settings.allowed_origins == [
        'http://127.0.0.1:*',
        'http://localhost:*',
        'http://[::1]:*',
    ]


def test_the_default_when_the_env_is_unset_is_protected(monkeypatch):
    """`_init_host` defaults to 127.0.0.1, so an unset env is the protected case.

    This is the row the 2.x default would have silently flipped: unset env plus the
    `0.0.0.0` bind default meant "protected" under 1.x and would mean "open" if the
    bind host chose.
    """
    monkeypatch.delenv('FASTMCP_HOST', raising=False)
    monkeypatch.setattr(srv, '_init_host', '127.0.0.1')
    assert srv._transport_security() is not None


@pytest.mark.parametrize('host', ['0.0.0.0', '::', '10.1.2.3'])
def test_a_non_loopback_host_disables_protection(monkeypatch, host):
    """What Docker asks for. `FASTMCP_HOST=0.0.0.0` is set by the compose overlays
    so requests arriving with a Docker service name in the Host header are served
    rather than rejected."""
    monkeypatch.setattr(srv, '_init_host', host)
    assert srv._transport_security() is None, (
        f'{host} must NOT enable DNS rebinding protection — inter-container '
        f'requests carry Docker hostnames that the allowlist would reject'
    )


def test_the_bind_host_cannot_change_the_policy(monkeypatch):
    """THE regression. Under 2.x the SDK would decide from the bind host; this
    server decides from FASTMCP_HOST. The two disagree in the default deployment
    (env unset -> loopback, config binds 0.0.0.0), and the env must win."""
    monkeypatch.setattr(srv, '_init_host', '127.0.0.1')
    protected = srv._transport_security()

    # `run_mcp_server` binds `config.server.host`, whose default is 0.0.0.0. That
    # value is passed as `host=` and must not feed the policy decision.
    from config.schema import ServerConfig

    assert ServerConfig().host == '0.0.0.0', (
        'the premise of this test is that the default BIND host is not loopback'
    )
    assert protected is not None, (
        'a 0.0.0.0 bind with a loopback FASTMCP_HOST must still be protected — '
        'this is exactly the row the SDK 2.x default would have flipped'
    )
