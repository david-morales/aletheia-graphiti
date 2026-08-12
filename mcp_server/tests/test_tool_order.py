"""`tools/list` must return the same order every time (2026-07-28 spec SHOULD).

The revision asks that list endpoints not vary per connection, and a stable order
is also what lets a client's prompt cache hit: the tool block is usually the first
thing in the context window, so a reshuffle invalidates everything after it.

Registration order alone is not a guarantee — it is an emergent property of two
code paths (`register_dynamic_tools` and the degraded fallback) appending in
whatever sequence they happen to use, and of nine decorators firing in source
order. `TOOL_ORDER` makes it a declared contract, and both paths apply it.
"""

from __future__ import annotations

import asyncio

import pytest
from mcp.server.mcpserver import MCPServer

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from tool_annotations import (
    TOOL_ANNOTATIONS,
    TOOL_ORDER,
    annotations_for,
    apply_canonical_tool_order,
)


def _profile(group_id: str = 'order_graph') -> DomainProfile:
    return DomainProfile(
        group_id=group_id,
        entity_types={'Widget': EntityTypeInfo('Widget', 3, 'A widget', ['W-1'])},
        edge_types={'USES': EdgeTypeInfo('USES', 1, 'uses', 'Widget -> Widget')},
        time_range=None,
    )


def _names(mcp_server: MCPServer) -> list[str]:
    return [t.name for t in asyncio.run(mcp_server.list_tools())]


def test_the_order_contract_covers_exactly_the_served_surface():
    assert list(TOOL_ORDER) == sorted(TOOL_ORDER, key=list(TOOL_ORDER).index)  # no dupes
    assert len(TOOL_ORDER) == len(set(TOOL_ORDER))
    assert set(TOOL_ORDER) == set(TOOL_ANNOTATIONS)


def test_two_independent_constructions_list_the_same_order():
    """The spec's requirement, stated directly."""
    first, second = MCPServer('a'), MCPServer('b')
    # Register in DIFFERENT sequences: reversed on the second server. A guarantee
    # that only holds when both paths happen to append identically is no guarantee.
    for name in TOOL_ORDER:
        first.add_tool(getattr(srv, name), annotations=annotations_for(name))
    for name in reversed(TOOL_ORDER):
        second.add_tool(getattr(srv, name), annotations=annotations_for(name))

    apply_canonical_tool_order(first._tool_manager._tools)
    apply_canonical_tool_order(second._tool_manager._tools)

    assert _names(first) == _names(second) == list(TOOL_ORDER)


def test_listing_twice_returns_the_same_order():
    server = MCPServer('stable')
    for name in reversed(TOOL_ORDER):
        server.add_tool(getattr(srv, name), annotations=annotations_for(name))
    apply_canonical_tool_order(server._tool_manager._tools)
    assert _names(server) == _names(server)


def test_the_dynamic_registration_path_emits_the_canonical_order():
    srv.register_dynamic_tools(_profile())
    assert list(srv.mcp._tool_manager._tools) == list(TOOL_ORDER)


def test_re_registration_does_not_reshuffle():
    """`register_dynamic_tools` deletes and re-adds nine tools; without the
    canonical pass they would migrate to the end of the dict every time."""
    srv.register_dynamic_tools(_profile())
    before = list(srv.mcp._tool_manager._tools)
    srv.register_dynamic_tools(_profile('other_graph'))
    assert list(srv.mcp._tool_manager._tools) == before == list(TOOL_ORDER)


def test_the_degraded_path_emits_the_same_order_as_the_healthy_one():
    """A profile failure must not reorder the catalog on top of everything else."""
    srv.register_dynamic_tools(_profile())
    healthy = list(srv.mcp._tool_manager._tools)
    srv.register_fallback_tools(reason='boom')
    assert list(srv.mcp._tool_manager._tools) == healthy


def test_an_unknown_tool_is_appended_rather_than_dropped():
    """Reordering must never lose a tool the table does not know about."""
    server = MCPServer('extra')
    for name in TOOL_ORDER:
        server.add_tool(getattr(srv, name), annotations=annotations_for(name))

    async def bonus_tool() -> dict:
        """A tool the order table has never heard of."""
        return {}

    server.add_tool(bonus_tool)
    apply_canonical_tool_order(server._tool_manager._tools)
    assert _names(server) == [*TOOL_ORDER, 'bonus_tool']


@pytest.mark.parametrize('missing', ['graph_query', 'clear_graph'])
def test_a_partial_surface_keeps_relative_order(missing):
    server = MCPServer('partial')
    for name in TOOL_ORDER:
        if name != missing:
            server.add_tool(getattr(srv, name), annotations=annotations_for(name))
    apply_canonical_tool_order(server._tool_manager._tools)
    assert _names(server) == [n for n in TOOL_ORDER if n != missing]
