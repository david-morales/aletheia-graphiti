"""The served surface uses ADR-019's canonical tool names — and only those.

ADR-019 R1 names the connector's capabilities after what they DO, not after the
mechanism that happens to implement them: `graph_query`, not `run_cypher`;
`explore_entity`, not `explore_node`; `profile_data`, not `profile_graph`.
`get_status` landed earlier; these three were the remainder.

Why this matters beyond taste: aletheia's capability/alias layer lists candidates
canonical-first, so while the fork announced legacy names EVERY lookup resolved to
its *second* candidate. The ordering expressed an intention the wire contradicted.

**No legacy aliases.** The rename deletes the old shape rather than gating it —
the same call ADR-025 made for the protocol era, and the standing house rule for
subsystem redesigns. A legacy name is not deprecated here; it is absent. The
direct callers (the UI's `nodes` and `profile` routes) move in the same wave.

These assertions are deliberately about the ANNOUNCED surface, because that is
what a consumer binds to. A test that only checked `TOOL_ORDER` would pass on a
table the registration path never reads.
"""

from __future__ import annotations

import asyncio

import pytest
from mcp.server.mcpserver import MCPServer

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from tool_annotations import TOOL_ANNOTATIONS, TOOL_ORDER, annotations_for

# ADR-019 R1. Left = what the connector announced before this change.
# NOTE: the legacy names below are string literals ON PURPOSE. They are the only
# place in the tree that may still spell them, because their absence everywhere
# else is exactly what this module asserts. A bulk rename that rewrites these
# turns every assertion into a tautology — which is how this file was almost
# lost during the very sweep it was written to guard.
RENAMED = {
    'run_cypher': 'graph_query',
    'explore_node': 'explore_entity',
    'profile_graph': 'profile_data',
}
LEGACY = frozenset(RENAMED)
CANONICAL = frozenset(RENAMED.values())


def _profile(group_id: str = 'canonical_graph') -> DomainProfile:
    return DomainProfile(
        group_id=group_id,
        entity_types={'Widget': EntityTypeInfo('Widget', 3, 'A widget', ['W-1'])},
        edge_types={'USES': EdgeTypeInfo('USES', 1, 'uses', 'Widget -> Widget')},
        time_range=None,
    )


def _announced(mcp_server: MCPServer) -> set[str]:
    return {t.name for t in asyncio.run(mcp_server.list_tools())}


@pytest.mark.parametrize('canonical', sorted(CANONICAL))
def test_the_contract_tables_carry_the_canonical_name(canonical):
    assert canonical in TOOL_ANNOTATIONS
    assert canonical in TOOL_ORDER


@pytest.mark.parametrize('legacy', sorted(LEGACY))
def test_no_contract_table_still_carries_a_legacy_name(legacy):
    assert legacy not in TOOL_ANNOTATIONS
    assert legacy not in TOOL_ORDER


@pytest.mark.parametrize('canonical', sorted(CANONICAL))
def test_the_module_exposes_the_tool_under_its_canonical_name(canonical):
    """`TOOL_ORDER` is resolved with `getattr(srv, name)` by both the registration
    path and the order tests, so the function name IS part of the contract."""
    assert callable(getattr(srv, canonical, None))


@pytest.mark.parametrize('legacy', sorted(LEGACY))
def test_the_module_no_longer_exposes_the_legacy_name(legacy):
    assert getattr(srv, legacy, None) is None


def test_the_dynamic_registration_path_announces_canonical_names_only():
    srv.register_dynamic_tools(_profile())
    announced = set(srv.mcp._tool_manager._tools)
    assert announced >= CANONICAL
    assert not (LEGACY & announced)


def test_the_degraded_fallback_announces_canonical_names_only():
    """A profile failure must not resurrect the old vocabulary."""
    srv.register_fallback_tools(reason='boom')
    announced = set(srv.mcp._tool_manager._tools)
    assert announced >= CANONICAL
    assert not (LEGACY & announced)


def test_the_rename_did_not_change_the_size_of_the_surface():
    """Renames, not additions: the ruling was a clean swap with no aliases."""
    server = MCPServer('sized')
    for name in TOOL_ORDER:
        server.add_tool(getattr(srv, name), annotations=annotations_for(name))
    assert len(_announced(server)) == len(TOOL_ORDER) == 18
