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

# Selected by the CI `contract` job (.github/workflows/mcp-server-tests.yml):
# these guards need no database and no API key, so they gate every change.
pytestmark = pytest.mark.contract

# The 13 tools that only ever read. `graph_query` belongs here: it is read-only by
# construction (writes are rejected by the validator and, on FalkorDB, by ro_query).
READ_ONLY_TOOLS = frozenset(
    {
        'explore_entity',
        'explore_ontology',
        'get_episode_context',
        'get_episodes',
        'get_ontology_documentation',
        'get_ontology_structure',
        'get_schema',
        'get_status',
        'profile_data',
        'graph_query',
        'sample_subgraph',
        'search',
        'search_ontology',
    }
)

# Mutating but NOT destructive: they add to the graph, they never remove.
MUTATING_TOOLS = frozenset({'add_memory'})

# Destructive: they remove data that cannot be recovered from the connector.
# `build_communities` belongs here despite its name — see
# test_build_communities_is_destructive_because_it_wipes_every_community.
DESTRUCTIVE_TOOLS = frozenset(
    {'build_communities', 'clear_graph', 'delete_entity_edge', 'delete_episode'}
)

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
def served_tools(monkeypatch):
    """The full 18-tool surface, exactly as `tools/list` would serve it.

    An ontology graph is configured on purpose: the four ontology tools are
    served only where one is (M11), and this module's subject is that EVERY
    served tool declares truthful annotations — which needs the arm that serves
    all eighteen, not the one that serves fourteen.
    """
    monkeypatch.setattr(
        srv,
        'config',
        type(
            'C',
            (),
            {
                'graphiti': type(
                    'G', (), {'ontology_graph': 'onto_v1', 'group_id': 'annotations_test'}
                )
            },
        ),
        raising=False,
    )
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
        name for name, tool in served_tools.items() if tool.annotations.read_only_hint is None
    )
    assert unset == [], f'ADR-019 R3: readOnlyHint unset on: {unset}'


def test_read_only_tools_are_flagged_read_only(served_tools):
    for name in sorted(READ_ONLY_TOOLS):
        assert served_tools[name].annotations.read_only_hint is True, name


def test_writing_tools_are_not_flagged_read_only(served_tools):
    for name in sorted(MUTATING_TOOLS | DESTRUCTIVE_TOOLS):
        assert served_tools[name].annotations.read_only_hint is False, name


def test_destructive_tools_declare_destructive_hint(served_tools):
    for name in sorted(DESTRUCTIVE_TOOLS):
        assert served_tools[name].annotations.destructive_hint is True, name


def test_additive_writers_declare_destructive_hint_false(served_tools):
    """`add_memory` mutates but never destroys — say so explicitly."""
    for name in sorted(MUTATING_TOOLS):
        assert served_tools[name].annotations.destructive_hint is False, name


def test_build_communities_is_destructive_because_it_wipes_every_community(served_tools):
    """Pinned with its justification so nobody "corrects" it back by its name.

    `build_communities(group_ids=["a"])` reads as additive and is not.
    `graphiti_core.Graphiti.build_communities` clears before it rebuilds: every
    existing Community node in the requested partitions, and every HAS_MEMBER edge
    hanging off one, is DETACH DELETEd first.

    The blast radius used to be wider still. The clear ran unscoped —
    `MATCH (c:Community) DETACH DELETE c`, no group filter, in both
    `utils/maintenance/community_operations.remove_communities` and the per-driver
    overrides — so rebuilding one partition destroyed every OTHER group_id's
    community layer and never restored it. BUG-57 scoped the wipe to the caller's
    group_ids; the tests that pin that live in the graphiti_core suite
    (`tests/test_remove_communities_group_scope.py`,
    `tests/driver/test_age_community_group_scope.py`). The annotation does not move
    with the fix: the tool still destroys data in the partitions it was handed.

    A false `readOnlyHint`/`destructiveHint` is worse than none — a HITL client
    auto-approves on it, and with no annotation at all the client would have
    fallen back to asking.
    """
    annotations = served_tools['build_communities'].annotations
    assert annotations.read_only_hint is False
    assert annotations.destructive_hint is True
    # NOT idempotent: each run mints fresh uuid4 Community nodes with freshly
    # generated LLM summaries, so a repeat invalidates every uuid a caller holds
    # and pays for another summarization pass. A client auto-retrying on this hint
    # would re-wipe and re-pay.
    assert annotations.idempotent_hint is False


def test_read_only_tools_do_not_claim_destructiveness(served_tools):
    """destructiveHint is only meaningful when readOnlyHint is false."""
    for name in sorted(READ_ONLY_TOOLS):
        assert served_tools[name].annotations.destructive_hint in (None, False), name


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
