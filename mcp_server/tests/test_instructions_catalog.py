"""ADR-019 R1 / A-D9: the announced catalog must cover what the server serves.

The served instructions named 7 of 18 tools. `add_memory` — the connector's only
ingestion path, and the one downstream services are told to use — appeared
nowhere on either arm, and neither did `profile_data`, `get_ontology_structure`
or `get_ontology_documentation`. An agent reading the announcement had no way to
learn that half the surface exists.

The three destructive tools were the mirror problem: served, unmentioned, and
therefore undescribed at the one place a client reads before deciding what is
safe to call.
"""

from __future__ import annotations

import pytest

from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from flavours.age import AgeFlavour
from tool_annotations import TOOL_ANNOTATIONS
from tool_descriptions import build_degraded_instructions, build_instructions

# Selected by the CI `contract` job (.github/workflows/mcp-server-tests.yml):
# these guards need no database and no API key, so they gate every change.
pytestmark = pytest.mark.contract

DESTRUCTIVE = ('build_communities', 'clear_graph', 'delete_entity_edge', 'delete_episode')


def _profile() -> DomainProfile:
    return DomainProfile(
        group_id='catalog_graph',
        entity_types={'Widget': EntityTypeInfo('Widget', 5, 'A widget', ['W-1'])},
        edge_types={'USES': EdgeTypeInfo('USES', 2, 'uses', 'Widget -> Widget')},
        time_range=None,
    )


@pytest.fixture(
    params=['healthy', 'healthy-empty', 'degraded'],
)
def instructions(request) -> str:
    """Every state in which a client can read the announcement."""
    if request.param == 'healthy':
        return build_instructions(_profile(), AgeFlavour())
    if request.param == 'healthy-empty':
        return build_instructions(DomainProfile(group_id='empty'), AgeFlavour())
    return build_degraded_instructions(
        group_id='catalog_graph', flavour=AgeFlavour(), reason='boom', marker='!! DEGRADED !!'
    )


@pytest.mark.parametrize('tool_name', sorted(TOOL_ANNOTATIONS))
def test_every_served_tool_appears_in_the_catalog(instructions, tool_name):
    assert tool_name in instructions, (
        f'ADR-019 R1: {tool_name} is served but never announced'
    )


def _catalog_entry(instructions: str, tool_name: str) -> str:
    """The tool's OWN catalog item: its line plus the indented continuation lines,
    stopping at the blank line that ends the item.

    Scoped deliberately. A fixed character window bleeds into the NEXT item — and
    into the destructive section's own heading — which made this guard pass for a
    tool announced under "Writing to the graph" with no warning of its own.
    """
    lines = instructions.split('\n')
    start = next(i for i, ln in enumerate(lines) if tool_name in ln)
    entry = [lines[start]]
    for ln in lines[start + 1:]:
        if not ln.strip():
            break
        entry.append(ln)
    return '\n'.join(entry)


@pytest.mark.parametrize('tool_name', DESTRUCTIVE)
def test_destructive_tools_are_announced_as_destructive(instructions, tool_name):
    """Naming a destructive tool without saying so is worse than not naming it."""
    entry = _catalog_entry(instructions, tool_name)
    assert 'DESTRUCTIVE' in entry.upper(), (
        f'{tool_name} is announced without a destructive warning of its own: {entry!r}'
    )


def test_clear_graph_says_it_is_irreversible(instructions):
    entry = _catalog_entry(instructions, 'clear_graph')
    assert 'cannot be undone' in entry.lower() or 'irreversible' in entry.lower()


def test_build_communities_announces_the_delete_inside_the_requested_groups(instructions):
    """Its name says "build"; it DELETES the requested partitions' communities first.

    This assertion is deliberately two-sided, because the served text has been wrong
    in both directions. It first announced an additive-sounding rebuild; wave 1 made it
    announce a whole-graph wipe (true then); BUG-57 scoped the delete, which made the
    whole-graph wording false. A wire description that overstates the blast radius is
    not the safe error — an agent that believes `build_communities` nukes the store
    will refuse a call it should make, and an operator who later discovers the claim is
    false stops trusting the destructive announcements that ARE true.
    """
    entry = _catalog_entry(instructions, 'build_communities').lower()
    assert 'destructive' in entry
    # The delete must still be announced — this is not a downgrade to "additive".
    assert 'delet' in entry, entry
    # ...and it must be announced as SCOPED to what the caller asked for.
    assert 'group_ids' in entry, entry
    # The pre-BUG-57 wording is now a factual error on the wire. Reject it by name.
    for stale in ('every community', 'all communities', 'did not ask'):
        assert stale not in entry, (
            f'the catalog still announces the pre-BUG-57 blast radius ({stale!r}): the '
            f'wipe is scoped to the requested group_ids now. Entry: {entry!r}'
        )


def test_the_catalog_does_not_announce_a_tool_that_is_not_served(instructions):
    """A phantom in the announcement costs a wasted call and a confused agent."""
    phantoms = {'add_triplet', 'search_nodes', 'search_memory_facts', 'get_entity_edge'}
    named = {p for p in phantoms if p in instructions}
    assert not named, f'announced but not served: {sorted(named)}'


def test_read_only_and_writing_tools_are_not_presented_as_interchangeable(instructions):
    """The catalog must separate reading from writing, not list all 18 flat."""
    assert 'Writing to the graph' in instructions
    assert 'Destructive' in instructions
