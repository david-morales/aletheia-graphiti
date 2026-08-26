"""M11: the ANNOUNCED surface must be honest about the ontology tools.

The four ontology tools (`search_ontology`, `explore_ontology`,
`get_ontology_structure`, `get_ontology_documentation`) were registered
unconditionally, with the `ontology_graph` check living INSIDE each handler. On a
connector configured without a companion ontology graph that made 4 of 18
announced tools answerable only by 'No ontology graph configured' — a quarter of
the catalogue that exists solely to fail.

Announcement is a promise (ADR-019 R1). A consumer resolves its capabilities
against `tools/list` and treats an unserved capability as the real answer
(`ToolNotAvailable`), so a SMALLER honest surface is the correct contract and a
larger dishonest one is the defect. The fix is therefore at REGISTRATION, not in
the handlers' error text.

Both layers are guarded here, because a fix to one alone re-creates the defect in
the other:

  * the TOOLS — `tools/list` must not name a tool this arm cannot serve;
  * the ANNOUNCEMENT — `instructions` is the catalogue a consumer reads at the
    handshake, and telling an agent to call `search_ontology` on an arm that does
    not announce it is the same dishonesty one layer up. Same for every served
    tool DESCRIPTION that cross-references one.

Both registration paths are covered. The degraded fallback exists to keep the
surface COMPLETE when the domain profile cannot be built (BUG-50 / A-D2), and
completeness there means every tool this configuration serves — not every tool
that exists. Losing the profile costs the descriptions; it does not conjure an
ontology graph.
"""

from __future__ import annotations

import asyncio
import re

import pytest
from mcp.server.mcpserver import MCPServer

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from flavours.age import AgeFlavour
from flavours.falkordb import FalkorDbFlavour
from tool_annotations import ONTOLOGY_TOOLS, TOOL_ANNOTATIONS, TOOL_ORDER
from tool_descriptions import build_degraded_instructions, build_instructions

# No database, no API key — this gates every change.
pytestmark = pytest.mark.contract

NON_ONTOLOGY_TOOLS = frozenset(TOOL_ANNOTATIONS) - ONTOLOGY_TOOLS


@pytest.fixture(autouse=True)
def _restore_server_globals():
    """Put the module-global server back exactly as we found it.

    `register_dynamic_tools` / `register_fallback_tools` mutate `srv.mcp` in
    place, and a leaked registration satisfies another module's
    `served == set(TOOL_ANNOTATIONS)` even when the path under test registered
    nothing — the failure no count-based gate can see. Copied from
    `test_canonical_tool_names.py`, which learned it the hard way.
    """
    mcp = srv.mcp
    tools = dict(mcp._tool_manager._tools)
    resources = dict(mcp._resource_manager._resources)
    instructions = mcp._lowlevel_server.instructions
    degraded_reason = srv._degraded_reason
    try:
        yield
    finally:
        mcp._tool_manager._tools.clear()
        mcp._tool_manager._tools.update(tools)
        mcp._resource_manager._resources.clear()
        mcp._resource_manager._resources.update(resources)
        mcp._lowlevel_server.instructions = instructions
        srv._degraded_reason = degraded_reason


def _profile(group_id: str = 'gate_graph') -> DomainProfile:
    return DomainProfile(
        group_id=group_id,
        entity_types={'Widget': EntityTypeInfo('Widget', 3, 'A widget', ['W-1'])},
        edge_types={'USES': EdgeTypeInfo('USES', 1, 'uses', 'Widget -> Widget')},
        time_range=None,
    )


def _config(ontology_graph: str | None):
    """The fields the registration paths read, shaped as the real config nests them."""
    return type(
        'C',
        (),
        {
            'graphiti': type(
                'G', (), {'ontology_graph': ontology_graph, 'group_id': 'gate_graph'}
            )
        },
    )


@pytest.fixture(params=[None, 'onto_v1'], ids=['without-ontology', 'with-ontology'])
def arm(request, monkeypatch):
    """Both configurations, on every guard in this module.

    Parametrised rather than written twice: the two arms differ in exactly one
    config field, and a guard that only ever runs on one of them is how the
    unconditional registration survived this long.
    """
    monkeypatch.setattr(srv, 'config', _config(request.param), raising=False)
    monkeypatch.setattr(srv, 'graphiti_service', None)
    return request.param


def _expected(ontology_graph: str | None) -> frozenset[str]:
    return frozenset(TOOL_ANNOTATIONS) if ontology_graph else NON_ONTOLOGY_TOOLS


def _announced() -> set[str]:
    return {t.name for t in asyncio.run(srv.mcp.list_tools())}


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


def test_the_dynamic_path_announces_only_what_this_arm_can_serve(arm):
    srv.register_dynamic_tools(_profile())
    assert _announced() == set(_expected(arm))


def test_the_degraded_path_announces_only_what_this_arm_can_serve(arm):
    """Completeness under degradation means every tool this CONFIGURATION serves.

    Losing the domain profile costs the DESCRIPTIONS, never the TOOLS — and it
    does not conjure an ontology graph either.
    """
    srv.register_fallback_tools(reason='boom')
    assert _announced() == set(_expected(arm))


def test_the_two_paths_announce_the_same_surface(arm):
    srv.register_dynamic_tools(_profile())
    healthy = _announced()
    srv.register_fallback_tools(reason='boom')
    assert _announced() == healthy


def test_the_ontology_tools_never_appear_without_an_ontology_graph(monkeypatch):
    """Stated directly, so the guard survives a change to `_expected`."""
    monkeypatch.setattr(srv, 'config', _config(None), raising=False)
    monkeypatch.setattr(srv, 'graphiti_service', None)
    srv.register_dynamic_tools(_profile())
    assert not (ONTOLOGY_TOOLS & _announced())


def test_configuring_an_ontology_restores_all_eighteen(monkeypatch):
    """The other direction, equally directly: the gate must not be a one-way door."""
    monkeypatch.setattr(srv, 'config', _config('onto_v1'), raising=False)
    monkeypatch.setattr(srv, 'graphiti_service', None)
    srv.register_dynamic_tools(_profile())
    assert _announced() == set(TOOL_ANNOTATIONS)
    assert len(TOOL_ORDER) == 18


def test_a_re_registration_follows_the_config_it_finds(monkeypatch):
    """`refresh_domain_surface` re-registers into a LIVE process.

    The delete-then-add pass must therefore be able to move the surface in both
    directions, or a connector whose ontology config changed under a reload keeps
    announcing the previous answer.
    """
    monkeypatch.setattr(srv, 'graphiti_service', None)
    monkeypatch.setattr(srv, 'config', _config('onto_v1'), raising=False)
    srv.register_dynamic_tools(_profile())
    assert ONTOLOGY_TOOLS.issubset(_announced())

    monkeypatch.setattr(srv, 'config', _config(None), raising=False)
    srv.register_dynamic_tools(_profile())
    assert not (ONTOLOGY_TOOLS & _announced())

    monkeypatch.setattr(srv, 'config', _config('onto_v1'), raising=False)
    srv.register_dynamic_tools(_profile())
    assert ONTOLOGY_TOOLS.issubset(_announced())


def test_the_non_ontology_surface_is_never_touched(arm):
    """The gate removes four tools and only those four."""
    srv.register_dynamic_tools(_profile())
    assert NON_ONTOLOGY_TOOLS.issubset(_announced())


def test_the_gated_surface_keeps_the_canonical_order(arm):
    srv.register_dynamic_tools(_profile())
    served = list(srv.mcp._tool_manager._tools)
    assert served == [n for n in TOOL_ORDER if n in _expected(arm)]


def test_the_gated_surface_keeps_its_annotations(arm):
    srv.register_dynamic_tools(_profile())
    for name, tool in srv.mcp._tool_manager._tools.items():
        assert tool.annotations is not None, name


# ---------------------------------------------------------------------------
# The announcement — the same honesty, one layer up
# ---------------------------------------------------------------------------

_CATALOG_ITEM = re.compile(r'^(\d+)\. (\w+) --')


def _catalogued(instructions: str) -> list[tuple[int, str]]:
    """The numbered catalogue items, as (number, tool name)."""
    return [
        (int(m.group(1)), m.group(2))
        for m in (_CATALOG_ITEM.match(ln) for ln in instructions.split('\n'))
        if m
    ]


@pytest.fixture(params=['age', 'falkordb'])
def flavour(request):
    """Both backends. `_search_catalog_entry` branches on the flavour, so the
    catalogue's first entry exists twice in the source — a duplicated block needs
    a guard per copy or only one copy is guarded."""
    return AgeFlavour() if request.param == 'age' else FalkorDbFlavour()


@pytest.fixture(params=[False, True], ids=['without-ontology', 'with-ontology'])
def has_ontology(request):
    return request.param


@pytest.fixture(params=['healthy', 'healthy-empty', 'degraded'])
def announcement(request, flavour, has_ontology) -> str:
    """Every state in which a client can read the announcement, on both backends
    and both ontology configurations."""
    if request.param == 'healthy':
        return build_instructions(_profile(), flavour, has_ontology=has_ontology)
    if request.param == 'healthy-empty':
        return build_instructions(
            DomainProfile(group_id='empty'), flavour, has_ontology=has_ontology
        )
    return build_degraded_instructions(
        group_id='gate_graph',
        flavour=flavour,
        reason='boom',
        marker='!! DEGRADED !!',
        has_ontology=has_ontology,
    )


def test_the_catalogue_names_exactly_the_tools_this_arm_serves(
    announcement, has_ontology
):
    expected = set(TOOL_ANNOTATIONS) if has_ontology else set(NON_ONTOLOGY_TOOLS)
    assert {name for _, name in _catalogued(announcement)} == expected


def test_the_announcement_never_mentions_an_unannounced_tool(
    announcement, has_ontology
):
    """Not just the catalogue items — anywhere in the text.

    A cross-reference is a call instruction: 'Cheaper than search_ontology when
    you want the whole list' sends the agent at a tool this arm does not serve
    just as surely as a catalogue entry would.
    """
    if has_ontology:
        pytest.skip('every tool is announced on this arm')
    named = sorted(t for t in ONTOLOGY_TOOLS if t in announcement)
    assert not named, f'the announcement points at unserved tools: {named}'


def test_the_catalogue_is_numbered_contiguously_from_one(announcement):
    """A gap reads as a truncated document. Removing entries 3, 4, 9 and 10 from
    an 18-item list must renumber it, not punch holes in it."""
    numbers = [n for n, _ in _catalogued(announcement)]
    assert numbers == list(range(1, len(numbers) + 1)), numbers


def test_the_catalogue_length_matches_the_arm(announcement, has_ontology):
    assert len(_catalogued(announcement)) == (18 if has_ontology else 14)


def test_the_sections_survive_the_gate(announcement):
    """Four of the five gated-out entries live under two different headings; the
    remaining sections must still be there and still separate reading from
    writing."""
    for heading in (
        'Key tools:',
        'Schema and structure:',
        'Episodes (the ingested source documents):',
        'Writing to the graph:',
        'Health:',
    ):
        assert heading in announcement, heading
    assert 'Destructive' in announcement


# ---------------------------------------------------------------------------
# The served DESCRIPTIONS — the third copy of the same promise
# ---------------------------------------------------------------------------


def test_no_served_description_points_at_an_unserved_tool(monkeypatch):
    """Mechanical over the whole registered surface, not a list of known sites.

    `build_search_description` routed schema questions to `search_ontology` from
    inside the `search` description — reached by every agent on every arm,
    including the ones that do not announce it.
    """
    monkeypatch.setattr(srv, 'config', _config(None), raising=False)
    monkeypatch.setattr(srv, 'graphiti_service', None)
    srv.register_dynamic_tools(_profile())

    offenders = {
        name: sorted(t for t in ONTOLOGY_TOOLS if t in (tool.description or ''))
        for name, tool in srv.mcp._tool_manager._tools.items()
    }
    offenders = {name: hits for name, hits in offenders.items() if hits}
    assert not offenders, f'served descriptions point at unserved tools: {offenders}'


def test_the_ontology_cross_reference_survives_where_it_is_true(monkeypatch):
    """The gate must remove the pointer, not delete it from the codebase."""
    monkeypatch.setattr(srv, 'config', _config('onto_v1'), raising=False)
    monkeypatch.setattr(srv, 'graphiti_service', None)
    srv.register_dynamic_tools(_profile())

    desc = srv.mcp._tool_manager._tools['search'].description or ''
    assert 'search_ontology' in desc


def test_an_isolated_server_built_from_the_table_still_carries_all_eighteen():
    """The ROSTER is unchanged: this is a per-configuration announcement gate,
    not a retirement. `TOOL_ANNOTATIONS` / `TOOL_ORDER` still cover 18 tools and
    every one of them is still importable and registrable."""
    from tool_annotations import annotations_for

    server = MCPServer('roster')
    for name in TOOL_ORDER:
        server.add_tool(getattr(srv, name), annotations=annotations_for(name))
    listed = {t.name for t in asyncio.run(server.list_tools())}
    assert listed == set(TOOL_ANNOTATIONS)
    assert len(listed) == 18
