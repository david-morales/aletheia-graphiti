"""BUG-50 / A-D2: a domain-profile failure must not silently shrink the surface.

Before this guard, `initialize_server`'s except branch re-registered only four
tools (`search`, `explore_node`, `search_ontology`, `explore_ontology`), so
`get_schema` — the canonical ADR-019 R5 payload every consumer discovers the
connector through — plus `run_cypher`, `profile_graph` and both ontology-bulk
tools vanished from `tools/list` while `/health` and `get_status` stayed green.
The failure was a `logger.warning`, and the hand-written seed instructions (a
10-tool catalog naming `clear_graph`) became the served announcement.

Composed with the consumer's capture-once prefetch, that shipped destructive-tool
guidance to an analyst-facing agent until the workbench restarted. So the
degraded state must be LOUD (ERROR), COMPLETE (18 tools) and SELF-DECLARING (a
marker in the served instructions).
"""

from __future__ import annotations

import logging

import pytest

import graphiti_mcp_server as srv
from flavours.falkordb import FalkorDbFlavour
from tool_annotations import TOOL_ANNOTATIONS

pytestmark = pytest.mark.asyncio


class _StubClient:
    driver = object()


class _StubService:
    """Enough of GraphitiService for the domain-surface build to reach its except."""

    def __init__(self):
        self.flavour = FalkorDbFlavour()
        self.ontology_client = None
        self.domain_profile = None
        self.config = srv.GraphitiConfig()

    async def get_client(self):
        return _StubClient()


@pytest.fixture
def degraded(monkeypatch):
    """Run the real domain-surface build with introspection guaranteed to fail."""
    service = _StubService()
    monkeypatch.setattr(srv, 'graphiti_service', service)
    monkeypatch.setattr(srv, 'config', service.config, raising=False)

    async def _boom(*args, **kwargs):
        raise RuntimeError('graph introspection exploded')

    monkeypatch.setattr(srv, 'build_domain_profile', _boom)
    return service


async def test_the_degraded_surface_still_serves_all_eighteen_tools(degraded):
    await srv._build_and_register_domain_surface()

    served = set(srv.mcp._tool_manager._tools)
    assert served == set(TOOL_ANNOTATIONS), (
        'a profile failure must not shrink the served surface; '
        f'missing={sorted(set(TOOL_ANNOTATIONS) - served)}'
    )


@pytest.mark.parametrize(
    'tool_name',
    ['get_schema', 'run_cypher', 'profile_graph', 'get_ontology_structure',
     'get_ontology_documentation'],
)
async def test_the_tools_the_old_fallback_dropped_are_served(degraded, tool_name):
    await srv._build_and_register_domain_surface()
    assert tool_name in srv.mcp._tool_manager._tools


async def test_the_degraded_surface_keeps_its_annotations(degraded):
    await srv._build_and_register_domain_surface()
    for name, tool in srv.mcp._tool_manager._tools.items():
        assert tool.annotations is not None, name


async def test_the_degraded_tools_carry_their_static_docstrings(degraded):
    await srv._build_and_register_domain_surface()
    desc = srv.mcp._tool_manager._tools['get_schema'].description or ''
    assert 'schema' in desc.lower()


async def test_the_served_instructions_declare_the_degradation(degraded):
    await srv._build_and_register_domain_surface()

    instructions = srv.mcp._mcp_server.instructions or ''
    assert instructions.startswith(srv.DEGRADED_INSTRUCTIONS_MARKER), (
        'the degraded marker must LEAD the announcement, not be buried in it'
    )
    assert 'DEGRADED' in instructions
    assert 'domain profile unavailable' in instructions


async def test_the_degraded_announcement_does_not_claim_an_empty_graph(degraded):
    """We could not introspect the graph — that is not the same as it being empty."""
    await srv._build_and_register_domain_surface()
    assert 'no entities yet' not in (srv.mcp._mcp_server.instructions or '')


async def test_the_degraded_announcement_still_carries_the_backend_dialect(degraded):
    """The flavour is known even when introspection fails — ADR-019 R6 still applies."""
    await srv._build_and_register_domain_surface()
    assert 'Cypher dialect:' in (srv.mcp._mcp_server.instructions or '')


async def test_the_degradation_is_logged_at_error(degraded, caplog):
    with caplog.at_level(logging.ERROR, logger=srv.logger.name):
        await srv._build_and_register_domain_surface()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, 'a collapsed tool surface is an ERROR, not a warning'
    assert any('graph introspection exploded' in r.getMessage() for r in errors), (
        'the underlying cause must reach the log'
    )


async def test_a_healthy_profile_leaves_no_degraded_marker(monkeypatch):
    """The happy path must not inherit the degraded announcement from a prior run."""
    service = _StubService()
    monkeypatch.setattr(srv, 'graphiti_service', service)
    monkeypatch.setattr(srv, 'config', service.config, raising=False)

    from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo

    profile = DomainProfile(
        group_id='healthy_graph',
        entity_types={'Widget': EntityTypeInfo('Widget', 4, 'A widget', ['W-1'])},
        edge_types={'USES': EdgeTypeInfo('USES', 2, 'uses', 'Widget -> Widget')},
        time_range=None,
    )

    async def _ok(*args, **kwargs):
        return profile

    monkeypatch.setattr(srv, 'build_domain_profile', _ok)
    await srv._build_and_register_domain_surface()

    instructions = srv.mcp._mcp_server.instructions or ''
    assert srv.DEGRADED_INSTRUCTIONS_MARKER not in instructions
    assert 'Widget' in instructions
    assert set(srv.mcp._tool_manager._tools) == set(TOOL_ANNOTATIONS)
