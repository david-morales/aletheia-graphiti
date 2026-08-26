"""M10: `search` must not announce a mode x reranker pair it cannot run.

`search_mode` and `reranker` are two independent `Literal`s, so the inputSchema
a client reads announces their full CROSS-PRODUCT — 5 x 5 = 25 pairs. Only the
pairs registered in `SEARCH_RECIPES` resolve; the rest raise `ValueError` inside
`resolve_search_config`. Measured on this tree before the fix: **17 resolve, 8
raise**, and the 8 are not obscure corners — `combined` is the DEFAULT mode and
`node_distance` is the reranker the `neighborhood` intent advertises.

Of the eight, FIVE are inexpressible and THREE are decisions:

  * inexpressible — `CommunityReranker` has only rrf / mmr / cross_encoder and
    `EpisodeReranker` only rrf / cross_encoder, so no recipe exists for
    `communities` x {node_distance, episode_mentions} or `episodes` x {mmr,
    node_distance, episode_mentions}.
  * decisions — `episodes` x `cross_encoder` is expressible and withheld pending
    measurement (see `EPISODE_SEARCH_RRF`). `combined` x {node_distance,
    episode_mentions} are ALSO expressible: a `combined` SearchConfig holds four
    independent per-leg rerankers, so the pair composes mechanically (edge and
    node legs take it; episode and community legs keep rrf because their enums
    hold nothing else). They are refused BECAUSE of that: `reranker` is one
    parameter, so the answer would honour it for two legs of four and rank the
    other two by rrf silently — a caller asking for proximity ranking getting
    rrf for half the payload. A pair that half-works quietly is worse than one
    that errors loudly, and `intent='neighborhood'` plus `explore_entity` are
    the honest routes to a proximity-ranked answer.

Either way the move is the M11 doctrine: do not widen the surface, CONSTRAIN the
announcement. A pair that will not run must be visible as such BEFORE the call,
in the text the client actually reads.

Which text that is has a trap in it, and it is the reason this defect survived a
docstring that already carried half the answer. `register_dynamic_tools` calls
`mcp.add_tool(search, description=build_search_description(...))`, and an
explicit `description=` REPLACES `search.__doc__`. So the docstring's
`search_mode="episodes" accepts only "rrf"` — the one constraint anyone had
written down — is served on the DEGRADED path (which passes no description and
falls back to the docstring) and is invisible on the normal one.

There are FOUR such channels, not one, and a fix that lands on some of them is
the same defect with a smaller blast radius. All four are pinned below:

  1. the served tool description (normal path),
  2. `search.__doc__` (degraded path),
  3. `get_schema` `tool_capabilities.search` — read as DATA by a planner, which
     had a flat five-reranker list saying nothing about which mode takes which,
     plus the whole `INTENT_STRATEGIES` dict advertising episode-shaped intents
     on arms that serve none,
  4. the in-band rejection — `search` returns the `ValueError` as
     `SearchResult(error=...)`, so the failure text is an announcement too, and
     an ungated render offered `episodes` as an alternative on a graph with none.

The mapping is DERIVED from `SEARCH_RECIPES` rather than written out per channel,
so a recipe added or removed cannot leave any of them lying: the mapping test
below fails the moment the derivation and the recipes disagree.

Every channel is checked on BOTH arms, because the episode gate cuts across all
four and the first cut of this fix got the direction wrong on two of them.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import typing

import pytest
from graphiti_core.search.search_config import (
    CommunityReranker,
    EdgeReranker,
    EpisodeReranker,
    NodeReranker,
)

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from flavours.age import AgeFlavour
from flavours.falkordb import FalkorDbFlavour
from graphiti_mcp_server import (
    SEARCH_RECIPES,
    describe_valid_search_combinations,
    resolve_search_config,
    search,
    valid_search_combinations,
)
from tool_annotations import annotations_for
from tool_descriptions import build_search_description

# No database, no API key.
pytestmark = pytest.mark.contract


def _profile(group_id: str = 'combo_graph') -> DomainProfile:
    return DomainProfile(
        group_id=group_id,
        entity_types={'Widget': EntityTypeInfo('Widget', 3, 'A widget', ['W-1'])},
        edge_types={'USES': EdgeTypeInfo('USES', 1, 'uses', 'Widget -> Widget')},
        time_range=None,
    )


def _literal_values(field: str) -> list[str]:
    """The values `search`'s signature announces for one parameter."""
    ann = typing.get_type_hints(search)[field]
    if typing.get_origin(ann) is typing.Literal:
        return list(typing.get_args(ann))
    for arg in typing.get_args(ann):  # Optional[Literal[...]]
        if typing.get_origin(arg) is typing.Literal:
            return list(typing.get_args(arg))
    raise AssertionError(f'{field} is not a Literal: {ann!r}')


ANNOUNCED_MODES = _literal_values('search_mode')
ANNOUNCED_RERANKERS = _literal_values('reranker')


def _unrunnable_pairs() -> list[tuple[str, str]]:
    bad = []
    for mode, reranker in itertools.product(ANNOUNCED_MODES, ANNOUNCED_RERANKERS):
        try:
            resolve_search_config(mode, reranker, 10)
        except ValueError:
            bad.append((mode, reranker))
    return bad


def _mode_line(rendered: str, mode: str) -> str:
    """The rendered line for `mode`, with a real failure if it is absent.

    N2: `next(genexpr)` raised a bare StopIteration here, which pytest reports as
    an error with no statement of what was expected — the wrong failure for a
    guard whose whole job is to say which mode went missing.
    """
    matches = [ln for ln in rendered.splitlines() if ln.strip().startswith(f'- {mode}:')]
    assert len(matches) == 1, (
        f'expected exactly one rendered line for mode {mode!r}, found '
        f'{len(matches)} in:\n{rendered}'
    )
    return matches[0]


def _rerankers_on(line: str) -> set[str]:
    """The rerankers a rendered line lists, by EXACT token.

    N2: substring matching would let a line reading `episode_mentions` satisfy a
    check for a reranker whose name is contained in another — the family already
    has `node_distance` next to `episode_mentions`, and one added tomorrow that
    is a prefix of a sibling would make every membership assertion here silently
    meaningless. Split on the separator the renderer actually uses.
    """
    _, _, listed = line.partition(':')
    return {token.strip() for token in listed.split(',') if token.strip()}


# ---------------------------------------------------------------------------
# The gap is real, and it is what the announcement has to cover
# ---------------------------------------------------------------------------


_MODE_RERANKER_ENUMS = {
    'nodes': (NodeReranker,),
    'edges': (EdgeReranker,),
    'episodes': (EpisodeReranker,),
    'communities': (CommunityReranker,),
    # A `combined` SearchConfig carries four INDEPENDENT per-leg rerankers, so the
    # pair is expressible when the legs that can vary accept it; the episode and
    # community legs keep rrf because their enums hold nothing else.
    'combined': (EdgeReranker, NodeReranker),
}


def _is_expressible(mode: str, reranker: str) -> bool:
    return all(reranker in enum.__members__ for enum in _MODE_RERANKER_ENUMS[mode])


def test_the_recorded_rationale_is_measured_not_asserted():
    """The rationale itself, pinned to the enums it claims to read.

    This test exists because the first version of this fix asserted "seven
    inexpressible, one decision" in three places and was wrong in two of them:
    `combined` was treated as if its community leg constrained the whole config,
    when a combined config carries four independent per-leg rerankers. A prose
    rationale nothing measured is exactly how that survived review-free.

    So the split is derived here. If a reranker is ever added to
    `CommunityReranker` or `EpisodeReranker`, these counts move and the three
    places that state them have to be revisited — which is the point.
    """
    unrunnable = _unrunnable_pairs()
    inexpressible = {p for p in unrunnable if not _is_expressible(*p)}
    decisions = {p for p in unrunnable if _is_expressible(*p)}

    assert inexpressible == {
        ('communities', 'node_distance'),
        ('communities', 'episode_mentions'),
        ('episodes', 'mmr'),
        ('episodes', 'node_distance'),
        ('episodes', 'episode_mentions'),
    }
    assert decisions == {
        ('episodes', 'cross_encoder'),
        ('combined', 'node_distance'),
        ('combined', 'episode_mentions'),
    }
    assert (len(inexpressible), len(decisions)) == (5, 3)


def test_the_announced_cross_product_exceeds_what_can_run():
    """Guards the premise. If someone ever makes every pair runnable this fails,
    and the honest-announcement machinery below becomes dead weight to delete."""
    announced = len(ANNOUNCED_MODES) * len(ANNOUNCED_RERANKERS)
    assert len(SEARCH_RECIPES) < announced
    assert _unrunnable_pairs(), 'no unrunnable pair — the premise has changed'


def test_every_unrunnable_pair_is_named_by_the_rendered_constraint():
    """The point of the whole exercise: a pair that raises must be visible as
    un-runnable in the text, per mode, before anyone spends a call on it."""
    rendered = describe_valid_search_combinations(episode_leg_is_live=True)
    for mode, reranker in _unrunnable_pairs():
        assert mode in rendered, f'mode {mode!r} absent from the constraint text'
        assert reranker not in _rerankers_on(_mode_line(rendered, mode)), (
            f'{mode!r} x {reranker!r} raises, yet the constraint text lists '
            f'{reranker!r} as accepted for {mode!r}'
        )


def test_the_rendered_constraint_lists_exactly_the_registered_recipes():
    """Derived, not restated — so a recipe change cannot leave the text stale."""
    rendered = describe_valid_search_combinations(episode_leg_is_live=True)
    for mode in ANNOUNCED_MODES:
        registered = {r for m, r in SEARCH_RECIPES if m == mode}
        listed = _rerankers_on(_mode_line(rendered, mode))
        assert listed == registered, (
            f'{mode!r}: text lists {sorted(listed)} but SEARCH_RECIPES registers '
            f'{sorted(registered)}'
        )


def test_the_data_and_the_prose_are_the_same_mapping():
    """`get_schema` serves the map and the description serves the prose. One
    source, so a planner and an agent cannot be told different things."""
    for live in (True, False):
        data = valid_search_combinations(live)
        rendered = describe_valid_search_combinations(live)
        assert set(data) == {
            ln.strip()[2:].split(':')[0]
            for ln in rendered.splitlines()
            if ln.strip().startswith('- ')
        }
        for mode, rerankers in data.items():
            assert _rerankers_on(_mode_line(rendered, mode)) == set(rerankers)


def test_the_liveness_argument_is_required():
    """No default in either direction. A default is how the review found the
    rejection text naming `episodes` on a graph that has none."""
    with pytest.raises(TypeError):
        valid_search_combinations()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        describe_valid_search_combinations()  # type: ignore[call-arg]


def test_the_header_does_not_claim_a_rejection_that_is_false_on_this_arm():
    """B4. On a gated arm three UNLISTED pairs are still accepted and answer
    (episodes x rrf, nodes/edges x episode_mentions), so a blanket "any other
    pair is rejected" is a falsehood that would stop a client calling a working
    pair. The gated header has to claim less than the full one."""
    full = describe_valid_search_combinations(episode_leg_is_live=True)
    gated = describe_valid_search_combinations(episode_leg_is_live=False)

    unlisted_but_accepted = [
        (mode, reranker)
        for mode, reranker in SEARCH_RECIPES
        if reranker not in valid_search_combinations(False).get(mode, ())
    ]
    assert unlisted_but_accepted, 'premise: the gate must hide some working pair'
    for mode, reranker in unlisted_but_accepted:
        resolve_search_config(mode, reranker, 10)  # accepted, and answers

    assert 'any other pair is rejected' in full
    assert 'any other pair is rejected' not in gated
    # both remain the rejection text, whose phrase has its own contract
    assert 'Valid combinations' in full and 'Valid combinations' in gated


def test_an_arm_without_the_episode_leg_names_nothing_episode_shaped():
    """The flavour gate this text has to live inside.

    Where episode content is not indexed, `search_mode='episodes'` returns
    nothing by construction AND `reranker='episode_mentions'` degenerates to rrf
    (its fallback Cypher anchors on `(n:Entity {uuid: ...})`, which AGE's
    single-label storage cannot satisfy for an ontology-classed entity —
    BUG-104). Announcing either would be the M11 defect wearing this fix as a
    disguise, so the whole episode shape drops out. Both stay callable.
    """
    gated = describe_valid_search_combinations(episode_leg_is_live=False)

    assert 'episode' not in gated.lower()
    assert not any(ln.strip().startswith('- episodes') for ln in gated.splitlines())
    # and it does not take the rest of the mapping down with it
    for mode in ('nodes', 'edges', 'communities', 'combined'):
        assert f'- {mode}: ' in gated
    assert 'rrf' in gated and 'node_distance' in gated


# ---------------------------------------------------------------------------
# Both channels that actually reach a client
# ---------------------------------------------------------------------------


def test_the_served_description_carries_the_constraint_on_a_full_backend():
    """The NORMAL path. `description=` overrides the docstring, so a constraint
    that lives only in the docstring is not served here at all."""
    desc = build_search_description(_profile(), FalkorDbFlavour(), False)
    for mode, reranker in _unrunnable_pairs():
        assert mode in desc, f'served description never mentions mode {mode!r}'
        assert reranker in desc or 'reranker' in desc, (
            f'served description says nothing that would stop a client trying '
            f'{mode!r} x {reranker!r}'
        )
    assert describe_valid_search_combinations(episode_leg_is_live=True) in desc


def test_the_served_description_carries_the_gated_constraint_on_age():
    """Same promise, minus the episode shape — and the AGE silence gate intact."""
    desc = build_search_description(_profile(), AgeFlavour(), False)

    assert describe_valid_search_combinations(episode_leg_is_live=False) in desc
    assert 'episode' not in desc.lower()
    assert 'node_distance' in desc, 'the gate must not swallow the whole mapping'


def test_the_docstring_carries_the_constraint_for_the_degraded_path():
    """The DEGRADED path passes no `description=`, so `search.__doc__` is what a
    client reads. It must not be the only honest channel, nor a silent one."""
    doc = search.__doc__ or ''
    for mode, _reranker in _unrunnable_pairs():
        assert mode in doc, f'docstring never mentions mode {mode!r}'
    assert 'node_distance' in doc and 'episode_mentions' in doc


def test_the_registered_tool_is_what_carries_it(monkeypatch):
    """End to end through the real registration, on the real tool manager: the
    description a client receives from tools/list is the constrained one."""
    mcp = srv.mcp
    tools = dict(mcp._tool_manager._tools)
    monkeypatch.setattr(
        srv,
        'config',
        type('C', (), {'graphiti': type('G', (), {'ontology_graph': None, 'group_id': 'g'})}),
        raising=False,
    )
    monkeypatch.setattr(srv, 'graphiti_service', None)
    try:
        srv.register_dynamic_tools(_profile())
        served = {t.name: t for t in asyncio.run(mcp.list_tools())}['search']
        # `graphiti_service` is None here, so the flavour is unknown and the
        # episode shape is gated out — the under-promising side, by design.
        assert describe_valid_search_combinations(episode_leg_is_live=False) in (
            served.description or ''
        )
    finally:
        mcp._tool_manager._tools.clear()
        mcp._tool_manager._tools.update(tools)


@pytest.mark.parametrize(
    ('flavour', 'leg_live'),
    [(FalkorDbFlavour(), True), (AgeFlavour(), False), (None, False)],
    ids=['falkordb', 'age', 'unknown'],
)
def test_the_error_a_client_gets_anyway_still_names_the_alternatives(
    monkeypatch, flavour, leg_live
):
    """Belt and braces: the announcement is advice, not enforcement — the schema
    cross-product still lets a pair through. The raise must stay actionable.

    And it is an ANNOUNCEMENT: `search` returns it in-band as
    `SearchResult(error=...)`, so it is gated like every other one. An ungated
    render offered `episodes` and `episode_mentions` as the alternatives to try
    on a graph that indexes no episode content — the fix's own defect, found by
    review, pinned here on all three arms.
    """
    monkeypatch.setattr(
        srv,
        'graphiti_service',
        None if flavour is None else type('S', (), {'flavour': flavour})(),
    )

    with pytest.raises(ValueError) as exc:
        resolve_search_config('combined', 'node_distance', 10)
    message = str(exc.value)

    assert 'combined' in message and 'node_distance' in message
    assert 'rrf' in message, 'the rejection must name a reranker that works'
    if leg_live:
        assert 'episodes' in message
    else:
        assert 'episode' not in message.lower(), (
            'the rejection offered an episode-shaped alternative on an arm that '
            'serves none'
        )


# ---------------------------------------------------------------------------
# Channel 3: get_schema, which a planner reads as DATA
# ---------------------------------------------------------------------------


async def _search_caps_for(flavour, monkeypatch):
    from tests.test_get_schema_canonical import _StubService

    monkeypatch.setattr(srv, 'graphiti_service', _StubService(flavour))
    schema = await srv.get_schema()
    return schema['tool_capabilities']['search']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('flavour', 'leg_live'),
    [(FalkorDbFlavour(), True), (AgeFlavour(), False)],
    ids=['falkordb', 'age'],
)
async def test_get_schema_serves_the_per_mode_mapping_not_a_flat_list(
    monkeypatch, flavour, leg_live
):
    """It was a flat list of five rerankers with no modes attached, so a planner
    reading it would compose `combined` x `node_distance` and get an error. The
    value has to be the per-mode map, and it has to be the SAME one the prose
    renders."""
    caps = await _search_caps_for(flavour, monkeypatch)

    assert caps['rerankers'] == valid_search_combinations(leg_live)
    assert isinstance(caps['rerankers'], dict), 'a flat list cannot express the constraint'
    for mode, rerankers in caps['rerankers'].items():
        for reranker in rerankers:
            resolve_search_config(mode, reranker, 10)  # every advertised pair runs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('flavour', 'leg_live'),
    [(FalkorDbFlavour(), True), (AgeFlavour(), False)],
    ids=['falkordb', 'age'],
)
async def test_get_schema_only_advertises_intents_this_arm_can_serve(
    monkeypatch, flavour, leg_live
):
    """`strategies` was the whole `INTENT_STRATEGIES` dict, so `narrative`
    (episodes) and `importance` (nodes x episode_mentions) were advertised on
    arms that serve neither."""
    caps = await _search_caps_for(flavour, monkeypatch)
    offered = valid_search_combinations(leg_live)

    for name, strategy in caps['strategies'].items():
        assert strategy['reranker'] in offered.get(strategy['search_mode'], ()), (
            f'intent {name!r} advertises {strategy} which this arm does not offer'
        )
    if leg_live:
        assert 'narrative' in caps['strategies']
        assert 'importance' in caps['strategies']
    else:
        assert 'narrative' not in caps['strategies']
        assert 'importance' not in caps['strategies']
        assert 'exhaustive' in caps['strategies'], 'the gate must not empty the set'


@pytest.mark.asyncio
async def test_get_schema_names_nothing_episode_shaped_on_a_gated_arm(monkeypatch):
    """The same silence the description keeps, in the data channel."""
    caps = await _search_caps_for(AgeFlavour(), monkeypatch)

    assert 'episode' not in json.dumps(
        {'rerankers': caps['rerankers'], 'strategies': caps['strategies']}
    ).lower()


# ---------------------------------------------------------------------------
# Nothing here may weaken what already worked
# ---------------------------------------------------------------------------


def test_every_registered_recipe_still_resolves():
    for mode, reranker in SEARCH_RECIPES:
        config = resolve_search_config(mode, reranker, 7)
        assert config.limit == 7


def test_every_intent_still_maps_to_a_registered_recipe():
    for name, strategy in srv.INTENT_STRATEGIES.items():
        key = (strategy['search_mode'], strategy['reranker'])
        assert key in SEARCH_RECIPES, f'intent {name!r} names unregistered {key}'


def test_the_search_tool_annotations_are_untouched():
    assert annotations_for('search') is not None
