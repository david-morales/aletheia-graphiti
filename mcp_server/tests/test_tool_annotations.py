"""ADR-019 R3: every served tool declares truthful `annotations`.

Regression guard for audit finding A-D1 (annotations 0/18 on the wire, both
flavours). `clear_graph`/`delete_*`/`add_memory` were machine-indistinguishable
from `get_status`: a client gating writes on `readOnlyHint`, or a
human-in-the-loop UI auto-approving read-only calls, had nothing to gate on.

The assertions cover BOTH registration paths — the statically decorated
`@mcp.tool()` tools and the nine registered by `register_dynamic_tools()` — by
iterating the tool manager after the dynamic pass, which is exactly the surface
`tools/list` serves.
"""

from __future__ import annotations

import pytest

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from tool_annotations import TOOL_ANNOTATIONS, annotations_for

# The 13 tools that only ever read. `run_cypher` belongs here: it is read-only by
# construction (writes are rejected by the validator and, on FalkorDB, by ro_query).
READ_ONLY_TOOLS = frozenset(
    {
        'explore_node',
        'explore_ontology',
        'get_episode_context',
        'get_episodes',
        'get_ontology_documentation',
        'get_ontology_structure',
        'get_schema',
        'get_status',
        'profile_graph',
        'run_cypher',
        'sample_subgraph',
        'search',
        'search_ontology',
    }
)

# Mutating but NOT destructive: they add to the graph, they never remove.
MUTATING_TOOLS = frozenset({'add_memory', 'build_communities'})

# Destructive: they remove data that cannot be recovered from the connector.
DESTRUCTIVE_TOOLS = frozenset({'clear_graph', 'delete_entity_edge', 'delete_episode'})

ALL_TOOLS = READ_ONLY_TOOLS | MUTATING_TOOLS | DESTRUCTIVE_TOOLS


def _profile() -> DomainProfile:
    return DomainProfile(
        group_id='annotations_test',
        entity_types={
            'Widget': EntityTypeInfo('Widget', 7, 'A widget', ['W-1']),
            'Gadget': EntityTypeInfo('Gadget', 3, 'A gadget', ['G-1']),
        },
        edge_types={'USES': EdgeTypeInfo('USES', 5, 'Widget uses Gadget', 'Widget -> Gadget')},
        time_range=None,
    )


@pytest.fixture
def served_tools():
    """The full 18-tool surface, exactly as `tools/list` would serve it."""
    srv.register_dynamic_tools(_profile())
    return srv.mcp._tool_manager._tools


def test_the_served_surface_is_the_eighteen_known_tools(served_tools):
    assert set(served_tools) == ALL_TOOLS
    assert len(ALL_TOOLS) == 18


def test_every_served_tool_declares_annotations(served_tools):
    missing = sorted(name for name, tool in served_tools.items() if tool.annotations is None)
    assert missing == [], f'ADR-019 R3: tools served without annotations: {missing}'


def test_every_served_tool_declares_read_only_hint(served_tools):
    unset = sorted(
        name for name, tool in served_tools.items() if tool.annotations.readOnlyHint is None
    )
    assert unset == [], f'ADR-019 R3: readOnlyHint unset on: {unset}'


def test_read_only_tools_are_flagged_read_only(served_tools):
    for name in sorted(READ_ONLY_TOOLS):
        assert served_tools[name].annotations.readOnlyHint is True, name


def test_writing_tools_are_not_flagged_read_only(served_tools):
    for name in sorted(MUTATING_TOOLS | DESTRUCTIVE_TOOLS):
        assert served_tools[name].annotations.readOnlyHint is False, name


def test_destructive_tools_declare_destructive_hint(served_tools):
    for name in sorted(DESTRUCTIVE_TOOLS):
        assert served_tools[name].annotations.destructiveHint is True, name


def test_additive_writers_declare_destructive_hint_false(served_tools):
    """`add_memory`/`build_communities` mutate but never destroy — say so explicitly."""
    for name in sorted(MUTATING_TOOLS):
        assert served_tools[name].annotations.destructiveHint is False, name


def test_read_only_tools_do_not_claim_destructiveness(served_tools):
    """destructiveHint is only meaningful when readOnlyHint is false."""
    for name in sorted(READ_ONLY_TOOLS):
        assert served_tools[name].annotations.destructiveHint in (None, False), name


def test_annotations_survive_a_dynamic_re_registration(served_tools):
    """register_dynamic_tools() deletes and re-adds nine tools — annotations must persist."""
    srv.register_dynamic_tools(_profile())
    tools = srv.mcp._tool_manager._tools
    assert set(tools) == ALL_TOOLS
    assert all(t.annotations is not None for t in tools.values())


def test_annotation_table_covers_exactly_the_served_surface():
    """One table, no per-call-site drift: the source of truth matches the surface."""
    assert set(TOOL_ANNOTATIONS) == ALL_TOOLS


def test_annotations_for_rejects_an_unknown_tool():
    """A typo at a registration site must fail loudly, not silently serve nothing."""
    with pytest.raises(KeyError):
        annotations_for('no_such_tool')
