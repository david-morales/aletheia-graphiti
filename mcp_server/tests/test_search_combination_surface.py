"""M10: `search` must not announce a mode x reranker pair it cannot run.

`search_mode` and `reranker` are two independent `Literal`s, so the inputSchema
a client reads announces their full CROSS-PRODUCT — 5 x 5 = 25 pairs. Only the
pairs registered in `SEARCH_RECIPES` resolve; the rest raise `ValueError` inside
`resolve_search_config`. Measured on this tree before the fix: **17 resolve, 8
raise**, and the 8 are not obscure corners — `combined` is the DEFAULT mode and
`node_distance` is the reranker the `neighborhood` intent advertises.

The gap is not implementable away. It is a limit of graphiti-core's own reranker
enums, not an omission here:

  * `CommunityReranker` has only rrf / mmr / cross_encoder — no `node_distance`,
    no `episode_mentions`. So `communities` cannot take those two, and neither
    can `combined`, whose community leg would have nothing to be reranked by.
  * `EpisodeReranker` has only rrf / cross_encoder, and cross_encoder is
    deliberately withheld pending measurement (see the `EPISODE_SEARCH_RRF`
    comment). So `episodes` takes rrf and nothing else.

Seven of the eight are therefore not expressible at all and the eighth is a
recorded decision. That leaves the M11 doctrine as the only honest move: do not
implement the announcement, CONSTRAIN it. A pair that cannot run must be visible
as un-runnable BEFORE the call, in the text the client actually reads.

Which text that is has a trap in it, and it is the reason this defect survived a
docstring that already carried half the answer. `register_dynamic_tools` calls
`mcp.add_tool(search, description=build_search_description(...))`, and an
explicit `description=` REPLACES `search.__doc__`. So the docstring's
`search_mode="episodes" accepts only "rrf"` — the one constraint anyone had
written down — is served on the DEGRADED path (which passes no description and
falls back to the docstring) and is invisible on the normal one. Both channels
carry the same promise and both are pinned here.

The rendered constraint is DERIVED from `SEARCH_RECIPES` rather than written out
a second time, so a recipe added or removed later cannot leave the announcement
lying: the mapping test below fails the moment the two disagree.
"""

from __future__ import annotations

import asyncio
import itertools
import typing

import pytest

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from flavours.age import AgeFlavour
from flavours.falkordb import FalkorDbFlavour
from graphiti_mcp_server import (
    SEARCH_RECIPES,
    describe_valid_search_combinations,
    resolve_search_config,
    search,
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


# ---------------------------------------------------------------------------
# The gap is real, and it is what the announcement has to cover
# ---------------------------------------------------------------------------


def test_the_announced_cross_product_exceeds_what_can_run():
    """Guards the premise. If someone ever makes every pair runnable this fails,
    and the honest-announcement machinery below becomes dead weight to delete."""
    announced = len(ANNOUNCED_MODES) * len(ANNOUNCED_RERANKERS)
    assert len(SEARCH_RECIPES) < announced
    assert _unrunnable_pairs(), 'no unrunnable pair — the premise has changed'


def test_every_unrunnable_pair_is_named_by_the_rendered_constraint():
    """The point of the whole exercise: a pair that raises must be visible as
    un-runnable in the text, per mode, before anyone spends a call on it."""
    rendered = describe_valid_search_combinations()
    for mode, reranker in _unrunnable_pairs():
        assert mode in rendered, f'mode {mode!r} absent from the constraint text'
        line = next(ln for ln in rendered.splitlines() if ln.strip().startswith(f'- {mode}'))
        assert reranker not in line, (
            f'{mode!r} x {reranker!r} raises, yet the constraint text lists '
            f'{reranker!r} as accepted for {mode!r}: {line!r}'
        )


def test_the_rendered_constraint_lists_exactly_the_registered_recipes():
    """Derived, not restated — so a recipe change cannot leave the text stale."""
    rendered = describe_valid_search_combinations()
    for mode in ANNOUNCED_MODES:
        registered = sorted(r for m, r in SEARCH_RECIPES if m == mode)
        line = next(ln for ln in rendered.splitlines() if ln.strip().startswith(f'- {mode}'))
        listed = sorted(r for r in ANNOUNCED_RERANKERS if r in line)
        assert listed == registered, (
            f'{mode!r}: text lists {listed} but SEARCH_RECIPES registers {registered}'
        )


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


def test_the_error_a_client_gets_anyway_still_names_the_alternatives():
    """Belt and braces: the announcement is advice, not enforcement — the schema
    cross-product still lets a pair through. The raise must stay actionable."""
    with pytest.raises(ValueError) as exc:
        resolve_search_config('combined', 'node_distance', 10)
    message = str(exc.value)
    assert 'combined' in message and 'node_distance' in message
    assert 'rrf' in message


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
