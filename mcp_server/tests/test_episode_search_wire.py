"""E1: the episode search leg, on the wire.

Every `combined` recipe in graphiti-core already carries
`episode_config=EpisodeSearchConfig(search_methods=[bm25])`, so the episode
full-text leg RUNS on every default `search` call. The connector then threw the
answer away: the result formatter built nodes, edges and communities and never
read `results.episodes`.

That is a capability-honesty defect, not a missing feature. An agent asking a
question whose answer lives in the free-text source narrative — never lifted into
an entity or a fact by extraction — got an empty-handed reply from a server that
had already computed the hit.

Whether the leg finds anything is a BACKEND question. FalkorDB builds the
`episode_content` fulltext index at ingest and ranks against it; AGE mirrors only
Entity nodes and edges into its tsvector shadow tables, so its
`episode_fulltext_search` returns [] by construction and the leg is a permanent
no-op there. The mode stays callable on both — an empty answer in-band beats a
hard error — but nothing ANNOUNCES it where it cannot work, which is what the
flavour-gating tests below pin.

These tests cover the payload, the explicit `episodes` search mode, the
`narrative` intent that routes to it, the flavour gate on every announcement,
and the truncation remedy actually being reachable.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.search.search_config import EpisodeReranker, EpisodeSearchMethod

from domain_profile import DomainProfile, EntityTypeInfo
from flavours.age import AgeFlavour
from flavours.base import BaseFlavour
from flavours.falkordb import FalkorDbFlavour
from graphiti_mcp_server import (
    INTENT_STRATEGIES,
    SEARCH_RECIPES,
    get_episode_context,
    resolve_search_config,
    search,
)
from tool_descriptions import build_instructions, build_search_description
from utils.formatting import COMBINED_EPISODE_LIMIT, EPISODE_CONTENT_CAP
from version import CONNECTOR_VERSION

# ---------------------------------------------------------------------------
# Mock factories — same idiom as test_tools.py
# ---------------------------------------------------------------------------

def make_mock_episode(
    uuid: str = 'ep-uuid-1',
    name: str = 'source-document-1',
    content: str = 'The narrative body of the ingested source document.',
    source: str = 'text',
    source_description: str = 'ingested batch',
    group_id: str = 'test-group',
    created_at: datetime | None = None,
    valid_at: datetime | None = None,
):
    """Create a mock EpisodicNode-like object."""
    episode = MagicMock()
    episode.uuid = uuid
    episode.name = name
    episode.content = content
    episode.source = MagicMock()
    episode.source.value = source
    episode.source_description = source_description
    episode.group_id = group_id
    episode.created_at = created_at or datetime(2025, 1, 1, tzinfo=timezone.utc)
    episode.valid_at = valid_at or datetime(2024, 6, 1, tzinfo=timezone.utc)
    return episode


def make_mock_search_results(
    nodes: list | None = None,
    edges: list | None = None,
    communities: list | None = None,
    episodes: list | None = None,
):
    """Create a mock SearchResults-like object, episode leg included."""
    results = MagicMock()
    results.nodes = nodes if nodes is not None else []
    results.edges = edges if edges is not None else []
    results.communities = communities if communities is not None else []
    results.episodes = episodes if episodes is not None else []
    return results


def make_mock_services(group_id: str = 'test-group'):
    """Mock graphiti_service / queue_service / config, plus the client."""
    mock_client = AsyncMock()
    mock_graphiti_service = AsyncMock()
    mock_graphiti_service.get_client = AsyncMock(return_value=mock_client)
    mock_queue_service = AsyncMock()

    mock_config = MagicMock()
    mock_config.graphiti.group_id = group_id

    return mock_graphiti_service, mock_queue_service, mock_config, mock_client


def _patched(svc, queue, cfg):
    return (
        patch('graphiti_mcp_server.graphiti_service', svc),
        patch('graphiti_mcp_server.queue_service', queue),
        patch('graphiti_mcp_server.config', cfg, create=True),
    )


async def _search_with(episodes: list, **kwargs):
    """Drive `search` over a results object carrying these episodes."""
    svc, queue, cfg, client = make_mock_services()
    client.search_ = AsyncMock(return_value=make_mock_search_results(episodes=episodes))
    a, b, c = _patched(svc, queue, cfg)
    with a, b, c:
        return await search(query='test query', **kwargs), client


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------

class TestEpisodesAreFormattedIntoTheResult:
    """The leg already ran; the formatter has to stop discarding it."""

    @pytest.mark.asyncio
    async def test_the_result_carries_an_episodes_key(self):
        result, _ = await _search_with([make_mock_episode()])

        assert 'error' not in result
        assert 'episodes' in result, (
            'the episode leg runs on every default search — its hits must reach the wire'
        )
        assert len(result['episodes']) == 1

    @pytest.mark.asyncio
    async def test_every_announced_field_is_present(self):
        result, _ = await _search_with([make_mock_episode()])
        episode = result['episodes'][0]

        assert episode['uuid'] == 'ep-uuid-1'
        assert episode['name'] == 'source-document-1'
        assert episode['content'] == 'The narrative body of the ingested source document.'
        assert episode['source'] == 'text'
        assert episode['source_description'] == 'ingested batch'
        assert episode['group_id'] == 'test-group'
        assert episode['created_at'] == '2025-01-01T00:00:00+00:00'
        assert episode['valid_at'] == '2024-06-01T00:00:00+00:00'

    @pytest.mark.asyncio
    async def test_null_timestamps_pass_through_rather_than_raising(self):
        episode = make_mock_episode()
        episode.created_at = None
        episode.valid_at = None
        result, _ = await _search_with([episode])

        assert result['episodes'][0]['created_at'] is None
        assert result['episodes'][0]['valid_at'] is None

    @pytest.mark.asyncio
    async def test_a_plain_string_source_survives_without_a_value_attribute(self):
        """`source` is an enum in core but a bare string in some fixtures."""
        episode = make_mock_episode()
        episode.source = 'json'
        result, _ = await _search_with([episode])

        assert result['episodes'][0]['source'] == 'json'

    @pytest.mark.asyncio
    async def test_no_embedding_key_reaches_the_wire(self):
        result, _ = await _search_with([make_mock_episode()])

        assert not [k for k in result['episodes'][0] if 'embedding' in k.lower()]

    @pytest.mark.asyncio
    async def test_the_search_ranking_order_is_preserved(self):
        """The formatter must not reorder what the reranker ranked.

        A count assertion cannot see this: reverse the list and every
        length-based test still passes. Rank is the only thing an episode result
        carries beyond its content — an agent reading the first hit is trusting
        it — so the order is asserted directly.
        """
        ranked = ['ep-first', 'ep-second', 'ep-third']
        result, _ = await _search_with([make_mock_episode(uuid=u) for u in ranked])

        assert [e['uuid'] for e in result['episodes']] == ranked

    @pytest.mark.asyncio
    async def test_the_message_counts_the_episodes(self):
        result, _ = await _search_with([make_mock_episode(), make_mock_episode(uuid='ep-2')])

        assert '2 episodes' in result['message'], result['message']

    @pytest.mark.asyncio
    async def test_no_episodes_is_an_empty_list_not_a_missing_key(self):
        result, _ = await _search_with([])

        assert result['episodes'] == []
        assert '0 episodes' in result['message']


class TestContentTruncation:
    """Content IS the payload here, so the cap is generous and always declared."""

    @pytest.mark.asyncio
    async def test_content_at_the_cap_is_not_truncated(self):
        result, _ = await _search_with([make_mock_episode(content='x' * EPISODE_CONTENT_CAP)])
        episode = result['episodes'][0]

        assert len(episode['content']) == EPISODE_CONTENT_CAP
        assert episode['content_truncated'] is False

    @pytest.mark.asyncio
    async def test_one_char_past_the_cap_truncates_and_says_so(self):
        result, _ = await _search_with(
            [make_mock_episode(content='x' * (EPISODE_CONTENT_CAP + 1))]
        )
        episode = result['episodes'][0]

        assert len(episode['content']) == EPISODE_CONTENT_CAP
        assert episode['content_truncated'] is True, (
            'a caller must be able to tell it is holding a prefix, so it knows to '
            'follow up with get_episode_context'
        )

    @pytest.mark.asyncio
    async def test_short_content_declares_itself_untruncated(self):
        result, _ = await _search_with([make_mock_episode(content='brief')])

        assert result['episodes'][0]['content_truncated'] is False

    @pytest.mark.asyncio
    async def test_empty_content_does_not_raise(self):
        episode = make_mock_episode()
        episode.content = None
        result, _ = await _search_with([episode])

        assert result['episodes'][0]['content'] == ''
        assert result['episodes'][0]['content_truncated'] is False


# ---------------------------------------------------------------------------
# The explicit mode
# ---------------------------------------------------------------------------

class TestEpisodesSearchMode:
    """`combined` runs the leg incidentally; `episodes` asks for it on purpose."""

    @pytest.mark.parametrize('reranker', ['rrf'])
    def test_the_mode_resolves_to_an_episode_only_config(self, reranker):
        config = resolve_search_config('episodes', reranker, 7)

        assert config.episode_config is not None
        assert config.node_config is None
        assert config.edge_config is None
        assert config.community_config is None
        assert config.limit == 7

    def test_the_leg_is_the_bm25_one_core_implements(self):
        config = resolve_search_config('episodes', 'rrf', 10)

        assert config.episode_config.search_methods == [EpisodeSearchMethod.bm25]

    def test_the_registered_reranker_reaches_the_config(self):
        config = resolve_search_config('episodes', 'rrf', 10)

        assert config.episode_config.reranker is EpisodeReranker.rrf

    def test_the_mode_is_case_insensitive_like_every_other(self):
        assert resolve_search_config('EPISODES', 'RRF', 10).episode_config is not None

    @pytest.mark.parametrize(
        'reranker', ['mmr', 'node_distance', 'episode_mentions', 'cross_encoder']
    )
    def test_an_unregistered_reranker_is_rejected(self, reranker):
        """Three of these `EpisodeReranker` cannot express at all. The fourth,
        cross_encoder, it CAN — and is withheld until someone measures it:
        reranking episodes means one completion per full passage, the reranker
        keys its map by content so duplicate passages collapse, and it zips
        scores against passages strictly. All four keep the same ValueError
        contract every other invalid combination gets."""
        with pytest.raises(ValueError, match=f"Invalid search_mode='episodes' \\+ reranker='{reranker}'"):
            resolve_search_config('episodes', reranker, 10)

    @pytest.mark.parametrize(
        'reranker', ['mmr', 'node_distance', 'episode_mentions', 'cross_encoder']
    )
    def test_the_rejection_still_lists_the_valid_combinations(self, reranker):
        with pytest.raises(ValueError, match='Valid combinations'):
            resolve_search_config('episodes', reranker, 10)

    def test_exactly_one_pair_is_registered(self):
        registered = {r for mode, r in SEARCH_RECIPES if mode == 'episodes'}
        assert registered == {'rrf'}, 'episodes ships rrf-only pending measurement'

    @pytest.mark.asyncio
    async def test_the_tool_accepts_the_mode_end_to_end(self):
        result, client = await _search_with(
            [make_mock_episode()], search_mode='episodes', reranker='rrf'
        )

        assert 'error' not in result
        assert len(result['episodes']) == 1
        config = client.search_.call_args.kwargs['config']
        assert config.episode_config is not None
        assert config.node_config is None


# ---------------------------------------------------------------------------
# The intent
# ---------------------------------------------------------------------------

class TestNarrativeIntent:
    """The intent layer is how an agent asks without knowing the mechanics."""

    def test_narrative_is_a_registered_intent(self):
        assert 'narrative' in INTENT_STRATEGIES

    def test_narrative_routes_to_the_episode_leg(self):
        strategy = INTENT_STRATEGIES['narrative']

        assert strategy['search_mode'] == 'episodes'
        assert strategy['reranker'] == 'rrf'
        assert strategy['limit'] == 5

    def test_every_intent_resolves_to_a_registered_recipe(self):
        """A strategy naming a pair `SEARCH_RECIPES` does not hold is a runtime
        error the type checker cannot see."""
        for name, strategy in INTENT_STRATEGIES.items():
            key = (strategy['search_mode'], strategy['reranker'])
            assert key in SEARCH_RECIPES, f"intent '{name}' names an unregistered recipe {key}"

    @pytest.mark.asyncio
    async def test_the_intent_drives_an_episode_only_config(self):
        _, client = await _search_with([make_mock_episode()], intent='narrative')

        config = client.search_.call_args.kwargs['config']
        assert config.episode_config is not None
        assert config.node_config is None
        assert config.edge_config is None

    @pytest.mark.asyncio
    async def test_the_intent_supplies_its_own_default_limit(self):
        _, client = await _search_with([make_mock_episode()], intent='narrative')

        assert client.search_.call_args.kwargs['config'].limit == 5

    @pytest.mark.asyncio
    async def test_an_explicit_limit_still_wins_over_the_intent(self):
        _, client = await _search_with([make_mock_episode()], intent='narrative', limit=25)

        assert client.search_.call_args.kwargs['config'].limit == 25


# ---------------------------------------------------------------------------
# Announcement + identity
# ---------------------------------------------------------------------------

def _profile() -> DomainProfile:
    return DomainProfile(
        group_id='wire_graph',
        entity_types={'Widget': EntityTypeInfo('Widget', 5, 'A widget', ['W-1'])},
        edge_types={},
        time_range=None,
    )


def _catalog_search_entry(flavour) -> str:
    from tool_descriptions import _key_tools_lines

    catalog = '\n'.join(_key_tools_lines(flavour))
    return catalog.split('1. search')[1].split('2. explore_entity')[0]


class TestTheAnnouncementIsGatedOnTheBackend:
    """M2. Announcement is a promise, and on AGE the leg cannot keep it.

    `age_search.episode_fulltext_search` returns [] by construction — episodic
    nodes are not mirrored into a tsvector shadow table. Telling an AGE-backed
    agent to fall back to `episodes` when nodes and edges look thin would spend
    a call and manufacture a false negative, so nothing announces the leg there.
    The mode stays CALLABLE on both: an honest empty answer in band beats a hard
    error the caller has to special-case.
    """

    def test_the_catalog_announces_the_leg_on_a_backend_that_implements_it(self):
        assert 'episode' in _catalog_search_entry(FalkorDbFlavour()).lower(), (
            'an agent that cannot learn the narratives are searchable will not search them'
        )

    def test_the_catalog_stays_silent_where_the_leg_is_a_permanent_no_op(self):
        assert 'episode' not in _catalog_search_entry(AgeFlavour()).lower()

    def test_an_unknown_backend_gets_no_claim(self):
        """Under-promising costs a capability; over-promising costs a wrong answer."""
        assert 'episode' not in _catalog_search_entry(BaseFlavour()).lower()
        assert 'episode' not in _catalog_search_entry(None).lower()

    def test_the_tool_description_announces_the_leg_on_falkordb(self):
        description = build_search_description(_profile(), FalkorDbFlavour())

        assert 'episodes' in description.lower()
        assert 'narrative' in description.lower()

    def test_the_tool_description_stays_silent_on_age(self):
        description = build_search_description(_profile(), AgeFlavour())

        assert 'episode' not in description.lower()
        assert 'narrative' not in description.lower()

    def test_the_served_instructions_follow_the_same_gate(self):
        assert 'episode narrative' in build_instructions(
            _profile(), FalkorDbFlavour()
        ).lower()
        age = build_instructions(_profile(), AgeFlavour()).lower()
        assert 'read `episodes`' not in age

    def test_the_docstring_makes_no_unconditional_episode_promise(self):
        """A docstring cannot be flavour-gated, so it must not claim the leg.

        It documents the PARAMETERS — which exist on every backend — and points
        at the two surfaces that do know: the served description and get_schema.
        """
        doc = (search.__doc__ or '').lower()

        assert 'episodes' in doc, 'the parameter still has to be documented'
        assert 'depends on the mode' in doc
        assert 'get_schema' in doc


class TestTheAnnouncedNumbersComeFromTheCode:
    """S4. An announcement carrying a hand-typed number drifts from the code
    the first time the constant moves, and nothing catches it."""

    def test_the_content_cap_is_announced_as_the_constant(self):
        assert str(EPISODE_CONTENT_CAP) in _catalog_search_entry(FalkorDbFlavour())
        assert str(EPISODE_CONTENT_CAP) in build_search_description(
            _profile(), FalkorDbFlavour()
        )

    def test_the_combined_episode_limit_is_announced_as_the_constant(self):
        assert str(COMBINED_EPISODE_LIMIT) in _catalog_search_entry(FalkorDbFlavour())
        assert str(COMBINED_EPISODE_LIMIT) in build_search_description(
            _profile(), FalkorDbFlavour()
        )

    def test_the_announced_remedy_names_the_real_signature(self):
        """M1. The remedy read `get_episode_context(uuid)`; the tool takes
        `episode_uuids: list[str]`. An agent following the announcement
        literally would have called it wrong."""
        for text in (
            _catalog_search_entry(FalkorDbFlavour()),
            build_search_description(_profile(), FalkorDbFlavour()),
        ):
            assert 'get_episode_context(episode_uuids=' in text
            assert 'get_episode_context(uuid)' not in text


class TestTheCombinedModeCapsTheEpisodeList:
    """M3. `combined` fans out over four legs and each episode hit carries a
    narrative, so an uncapped list drowns a result the caller asked for
    entities in. Asking for episodes explicitly IS the caller saying that is
    what they want, and keeps the full limit."""

    @pytest.mark.asyncio
    async def test_a_combined_search_returns_at_most_the_sample(self):
        many = [make_mock_episode(uuid=f'ep-{i}') for i in range(COMBINED_EPISODE_LIMIT + 4)]
        result, _ = await _search_with(many, limit=20)

        assert len(result['episodes']) == COMBINED_EPISODE_LIMIT

    @pytest.mark.asyncio
    async def test_the_sample_is_the_top_of_the_ranking_not_an_arbitrary_slice(self):
        many = [make_mock_episode(uuid=f'ep-{i}') for i in range(COMBINED_EPISODE_LIMIT + 4)]
        result, _ = await _search_with(many, limit=20)

        assert [e['uuid'] for e in result['episodes']] == [
            f'ep-{i}' for i in range(COMBINED_EPISODE_LIMIT)
        ]

    @pytest.mark.asyncio
    async def test_the_message_counts_what_was_actually_returned(self):
        many = [make_mock_episode(uuid=f'ep-{i}') for i in range(COMBINED_EPISODE_LIMIT + 4)]
        result, _ = await _search_with(many, limit=20)

        assert f'{COMBINED_EPISODE_LIMIT} episodes' in result['message']

    @pytest.mark.asyncio
    async def test_the_explicit_mode_is_not_capped(self):
        many = [make_mock_episode(uuid=f'ep-{i}') for i in range(COMBINED_EPISODE_LIMIT + 4)]
        result, _ = await _search_with(many, search_mode='episodes', reranker='rrf', limit=20)

        assert len(result['episodes']) == COMBINED_EPISODE_LIMIT + 4

    @pytest.mark.asyncio
    async def test_the_narrative_intent_is_not_capped_either(self):
        many = [make_mock_episode(uuid=f'ep-{i}') for i in range(COMBINED_EPISODE_LIMIT + 4)]
        result, _ = await _search_with(many, intent='narrative')

        assert len(result['episodes']) == COMBINED_EPISODE_LIMIT + 4

    @pytest.mark.asyncio
    async def test_a_short_list_is_untouched_by_the_cap(self):
        result, _ = await _search_with([make_mock_episode()])

        assert len(result['episodes']) == 1


class TestTheTruncationRemedyIsReal:
    """M1. `search` caps episode content and names `get_episode_context` as
    where the rest lives. That tool returned only the derived nodes and edges —
    no content at all — so the announced follow-up could not deliver what it
    was announced for."""

    @staticmethod
    async def _context_with(episodes, nodes=None, edges=None):
        svc, queue, cfg, client = make_mock_services()
        client.get_nodes_and_edges_by_episode = AsyncMock(
            return_value=make_mock_search_results(nodes=nodes or [], edges=edges or [])
        )
        a, b, c = _patched(svc, queue, cfg)
        with a, b, c, patch(
            'graphiti_mcp_server.EpisodicNode.get_by_uuids',
            AsyncMock(return_value=episodes),
        ):
            return await get_episode_context(episode_uuids=[e.uuid for e in episodes])

    @pytest.mark.asyncio
    async def test_the_requested_episodes_come_back(self):
        result = await self._context_with([make_mock_episode()])

        assert 'error' not in result
        assert [e['uuid'] for e in result['episodes']] == ['ep-uuid-1']

    @pytest.mark.asyncio
    async def test_content_is_served_in_full_past_the_search_cap(self):
        body = 'x' * (EPISODE_CONTENT_CAP * 2)
        result = await self._context_with([make_mock_episode(content=body)])

        assert result['episodes'][0]['content'] == body, (
            'capping here too would make the announced remedy a dead end'
        )

    @pytest.mark.asyncio
    async def test_nothing_is_ever_flagged_truncated_here(self):
        result = await self._context_with(
            [make_mock_episode(content='x' * (EPISODE_CONTENT_CAP * 2))]
        )

        assert result['episodes'][0]['content_truncated'] is False

    @pytest.mark.asyncio
    async def test_the_extraction_it_always_returned_is_still_there(self):
        result = await self._context_with([make_mock_episode()])

        assert 'nodes' in result
        assert 'edges' in result

    @pytest.mark.asyncio
    async def test_a_uuid_that_resolves_to_nothing_is_not_an_error(self):
        svc, queue, cfg, client = make_mock_services()
        client.get_nodes_and_edges_by_episode = AsyncMock(
            return_value=make_mock_search_results()
        )
        a, b, c = _patched(svc, queue, cfg)
        with a, b, c, patch(
            'graphiti_mcp_server.EpisodicNode.get_by_uuids', AsyncMock(return_value=[])
        ):
            result = await get_episode_context(episode_uuids=['ghost'])

        assert 'error' not in result
        assert result['episodes'] == []
        assert '0 of 1' in result['message']


class TestTheCapabilityBlockAnnouncesTheLeg:
    """M4. `get_schema.tool_capabilities` is what a planner READS to decide what
    is worth asking for — the primary consumer of this whole change. It is
    gated on the same capability as the prose, for the same reason."""

    @staticmethod
    async def _schema_for(flavour, monkeypatch):
        import graphiti_mcp_server as srv
        from tests.test_get_schema_canonical import _StubService

        monkeypatch.setattr(srv, 'graphiti_service', _StubService(flavour))
        return await srv.get_schema()

    @staticmethod
    def _search_caps(schema):
        return schema['tool_capabilities']['search']

    @pytest.mark.asyncio
    async def test_falkordb_announces_the_episode_content_index(self, monkeypatch):
        caps = self._search_caps(await self._schema_for(FalkorDbFlavour(), monkeypatch))
        indexes = [m.get('index') for m in caps['search_methods']]

        assert 'episode_content' in indexes

    @pytest.mark.asyncio
    async def test_falkordb_announces_it_as_a_bm25_leg(self, monkeypatch):
        caps = self._search_caps(await self._schema_for(FalkorDbFlavour(), monkeypatch))
        entry = next(m for m in caps['search_methods'] if m.get('index') == 'episode_content')

        assert entry['type'] == 'bm25_fulltext'

    @pytest.mark.asyncio
    async def test_falkordb_covers_episodes(self, monkeypatch):
        caps = self._search_caps(await self._schema_for(FalkorDbFlavour(), monkeypatch))

        assert caps['covers'].get('episodes') is True

    @pytest.mark.asyncio
    async def test_age_announces_neither(self, monkeypatch):
        caps = self._search_caps(await self._schema_for(AgeFlavour(), monkeypatch))
        indexes = [m.get('index') for m in caps['search_methods']]

        assert 'episode_content' not in indexes
        assert 'episodes' not in caps['covers']

    @pytest.mark.asyncio
    async def test_the_methods_a_planner_already_relied_on_are_untouched(self, monkeypatch):
        """Additive, on both arms — this must not narrow an existing promise."""
        for flavour in (FalkorDbFlavour(), AgeFlavour()):
            caps = self._search_caps(await self._schema_for(flavour, monkeypatch))
            indexes = {m.get('index') for m in caps['search_methods']}

            assert {'name_embedding', 'summary_embedding', 'summary', 'fact_embedding'} <= indexes
            assert caps['covers']['communities'] is True


class TestTheFlavourCapabilityIsTruthful:
    """The gate is only as good as what it reads."""

    def test_falkordb_says_it_indexes_episode_content(self):
        assert FalkorDbFlavour().searches_episode_content() is True

    def test_age_says_it_does_not(self):
        assert AgeFlavour().searches_episode_content() is False

    def test_the_generic_backend_defaults_to_no(self):
        assert BaseFlavour().searches_episode_content() is False


def _parts(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.split(r'[.\-+]', version)[:3] if p.isdigit())


def test_the_episode_leg_ships_no_earlier_than_the_release_that_introduced_it():
    """A FLOOR, not a pin.

    The suite deliberately refuses to hand-maintain a copy of the version
    (`test_the_version_matches_the_packaging_metadata`), so pinning `== 2.6.0`
    here would go red at the next bump and teach everyone to edit the guard
    instead of reading it. The honest invariant is one-directional: the wire now
    carries the episode leg, and a build announcing itself as older than the
    release that added it is announcing a surface it does not have.
    """
    assert _parts(CONNECTOR_VERSION) >= (2, 6, 0), (
        f'this build announces {CONNECTOR_VERSION}; the episode search leg is new wire '
        f'functionality and ships as a MINOR — bump to 2.6.0'
    )
