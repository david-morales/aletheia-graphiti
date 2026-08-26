"""BUG-50 / A-D2: a domain-profile failure must not silently shrink the surface.

Before this guard, `initialize_server`'s except branch re-registered only four
tools (`search`, `explore_entity`, `search_ontology`, `explore_ontology`), so
`get_schema` — the canonical ADR-019 R5 payload every consumer discovers the
connector through — plus `graph_query`, `profile_data` and both ontology-bulk
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
from tool_annotations import ONTOLOGY_TOOLS, TOOL_ANNOTATIONS

pytestmark = pytest.mark.asyncio

NON_ONTOLOGY_TOOLS = frozenset(TOOL_ANNOTATIONS) - ONTOLOGY_TOOLS


def _expected(service) -> set[str]:
    """What COMPLETE means for this arm.

    Losing the domain profile costs the DESCRIPTIONS, never the TOOLS — and it
    does not conjure a companion ontology graph either. On a connector that
    configures none, the four ontology tools are not part of the served surface
    in ANY state (M11), so a degraded fallback that registered them would be
    re-opening the announcement defect rather than preserving completeness.
    """
    if service.config.graphiti.ontology_graph:
        return set(TOOL_ANNOTATIONS)
    return set(NON_ONTOLOGY_TOOLS)


@pytest.fixture(autouse=True)
def _restore_last_served_bodies():
    """Every test in this module may drive `_build_and_register_domain_surface`,
    whose last act records the P4 content baseline into process-global state no
    `monkeypatch` tracks. Autouse — a per-fixture save/restore is bypassable,
    and the healthy-arm test below takes `monkeypatch`, not `degraded`, so it
    would leak all four fingerprints into the freshness suites, where a
    hand-registered resource then reads as a content change.
    """
    served = dict(srv._last_served_resource_bodies)
    try:
        yield
    finally:
        srv._last_served_resource_bodies.clear()
        srv._last_served_resource_bodies.update(served)


class _StubClient:
    driver = object()


class _StubService:
    """Enough of GraphitiService for the domain-surface build to reach its except."""

    def __init__(self, ontology_graph: str | None = None):
        self.flavour = FalkorDbFlavour()
        self.ontology_client = None
        self.domain_profile = None
        self.config = srv.GraphitiConfig()
        self.config.graphiti.ontology_graph = ontology_graph

    async def get_client(self):
        return _StubClient()


@pytest.fixture(params=[None, 'onto_v1'], ids=['without-ontology', 'with-ontology'])
def degraded(request, monkeypatch):
    """Run the real domain-surface build with introspection guaranteed to fail.

    Both ontology configurations, because "COMPLETE" is per-arm since M11 and a
    guard pinned to one of them would not notice the fallback re-announcing four
    tools the other arm cannot serve.

    The P4 content-baseline leak this module used to cause is closed at source
    by `_restore_last_served_bodies` above — autouse, so no test can bypass it
    the way a per-fixture save/restore could be (and the healthy-arm test did).
    The freshness suites keep their own clear-on-entry as defence in depth.
    """
    service = _StubService(ontology_graph=request.param)
    monkeypatch.setattr(srv, 'graphiti_service', service)
    monkeypatch.setattr(srv, 'config', service.config, raising=False)

    async def _boom(*args, **kwargs):
        raise RuntimeError('graph introspection exploded')

    monkeypatch.setattr(srv, 'build_domain_profile', _boom)
    yield service


async def test_the_degraded_surface_still_serves_every_tool_this_arm_has(degraded):
    await srv._build_and_register_domain_surface()

    served = set(srv.mcp._tool_manager._tools)
    expected = _expected(degraded)
    assert served == expected, (
        'a profile failure must not shrink the served surface; '
        f'missing={sorted(expected - served)}, extra={sorted(served - expected)}'
    )


async def test_the_degraded_surface_is_all_eighteen_where_an_ontology_is_configured(
    degraded,
):
    """The BUG-50 number, stated as a number, on the arm where 18 is the answer."""
    if not degraded.config.graphiti.ontology_graph:
        pytest.skip('this arm serves 14 — see the guard above')
    await srv._build_and_register_domain_surface()
    assert len(srv.mcp._tool_manager._tools) == 18


@pytest.mark.parametrize(
    'tool_name',
    ['get_schema', 'graph_query', 'profile_data', 'get_ontology_structure',
     'get_ontology_documentation'],
)
async def test_the_tools_the_old_fallback_dropped_are_served(degraded, tool_name):
    if tool_name not in _expected(degraded):
        pytest.skip(f'{tool_name} is not served on this arm')
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

    instructions = srv.mcp._lowlevel_server.instructions or ''
    assert instructions.startswith(srv.DEGRADED_INSTRUCTIONS_MARKER), (
        'the degraded marker must LEAD the announcement, not be buried in it'
    )
    assert 'DEGRADED' in instructions
    assert 'domain profile unavailable' in instructions


async def test_the_degraded_announcement_does_not_claim_an_empty_graph(degraded):
    """We could not introspect the graph — that is not the same as it being empty."""
    await srv._build_and_register_domain_surface()
    assert 'no entities yet' not in (srv.mcp._lowlevel_server.instructions or '')


async def test_the_degraded_announcement_still_carries_the_backend_dialect(degraded):
    """The flavour is known even when introspection fails — ADR-019 R6 still applies."""
    await srv._build_and_register_domain_surface()
    assert 'Cypher dialect:' in (srv.mcp._lowlevel_server.instructions or '')


async def test_the_degradation_is_logged_at_error(degraded, caplog):
    with caplog.at_level(logging.ERROR, logger=srv.logger.name):
        await srv._build_and_register_domain_surface()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, 'a collapsed tool surface is an ERROR, not a warning'
    assert any('graph introspection exploded' in r.getMessage() for r in errors), (
        'the underlying cause must reach the log'
    )


@pytest.mark.parametrize('ontology_graph', [None, 'onto_v1'])
async def test_a_healthy_profile_leaves_no_degraded_marker(monkeypatch, ontology_graph):
    """The happy path must not inherit the degraded announcement from a prior run."""
    service = _StubService(ontology_graph=ontology_graph)
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

    instructions = srv.mcp._lowlevel_server.instructions or ''
    assert srv.DEGRADED_INSTRUCTIONS_MARKER not in instructions
    assert 'Widget' in instructions
    assert set(srv.mcp._tool_manager._tools) == _expected(service)


async def test_the_degraded_resource_set_is_pruned_and_declared(degraded):
    """L5: `resources/list` shrinks under degradation — say so, don't let a client
    infer it from an absence, and never serve profile-rendered text after the
    profile that rendered it is gone."""
    srv.register_resources(
        __import__('domain_profile').DomainProfile(group_id='before_failure')
    )
    await srv._build_and_register_domain_surface()

    uris = {str(r.uri) for r in await srv.mcp.list_resources()}
    assert uris == {'graphiti://schema'}, uris

    instructions = srv.mcp._lowlevel_server.instructions or ''
    assert 'graphiti://schema' in instructions
    assert 'cannot be built without one' in instructions


async def test_the_pre_startup_announcement_is_itself_a_degraded_one():
    """L6: the string MCPServer is constructed with used to be a hand-written 10-tool
    catalog naming `clear_graph` without marking it destructive — the artifact that
    made BUG-50 dangerous. It is unreachable on the wire today, and a dead
    announcement contradicting the live one is a trap for whoever changes that.
    """
    seed = srv.GRAPHITI_MCP_INSTRUCTIONS
    assert seed.startswith(srv.DEGRADED_INSTRUCTIONS_MARKER)
    # Built at IMPORT time, before any config is loaded, so it has no positive
    # evidence of a companion ontology graph and does not claim one (M11) —
    # the same reading `_ontology_is_configured` gives an absent config.
    for name in NON_ONTOLOGY_TOOLS:
        assert name in seed, f'{name} missing from the pre-startup catalog'
    for name in ONTOLOGY_TOOLS:
        assert name not in seed, (
            f'{name} is claimed by the pre-startup catalog, which cannot know '
            f'whether this connector configures an ontology graph'
        )
    entry = next(ln for ln in seed.split('\n') if 'clear_graph' in ln)
    assert 'DESTRUCTIVE' in entry.upper()
