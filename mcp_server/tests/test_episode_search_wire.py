"""E1: the episode search leg, on the wire.

Every `combined` recipe in graphiti-core already carries
`episode_config=EpisodeSearchConfig(search_methods=[bm25])`, so the episode
full-text leg RUNS on every default `search` call and both flavours implement it
(FalkorDB `episode_fulltext_search` over the `episode_content` index; AGE
`age_search`). The connector then threw the answer away: the result formatter
built nodes, edges and communities and never read `results.episodes`.

That is a capability-honesty defect, not a missing feature. An agent asking a
question whose answer lives in the free-text source narrative — never lifted into
an entity or a fact by extraction — got an empty-handed reply from a server that
had already computed the hit.

These tests pin the three halves of putting it back on the wire: the payload, the
explicit `episodes` search mode, and the `narrative` intent that routes to it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.search.search_config import EpisodeReranker, EpisodeSearchMethod

from version import CONNECTOR_VERSION

from graphiti_mcp_server import (
    EPISODE_CONTENT_CAP,
    INTENT_STRATEGIES,
    SEARCH_RECIPES,
    resolve_search_config,
    search,
)


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
    async def test_the_cap_is_generous_enough_to_carry_a_narrative(self):
        assert EPISODE_CONTENT_CAP >= 6000

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

    @pytest.mark.parametrize('reranker', ['rrf', 'cross_encoder'])
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

    @pytest.mark.parametrize(
        ('reranker', 'expected'),
        [('rrf', EpisodeReranker.rrf), ('cross_encoder', EpisodeReranker.cross_encoder)],
    )
    def test_each_reranker_reaches_the_config(self, reranker, expected):
        config = resolve_search_config('episodes', reranker, 10)

        assert config.episode_config.reranker is expected

    def test_the_mode_is_case_insensitive_like_every_other(self):
        assert resolve_search_config('EPISODES', 'RRF', 10).episode_config is not None

    @pytest.mark.parametrize('reranker', ['mmr', 'node_distance', 'episode_mentions'])
    def test_a_reranker_episodereranker_does_not_have_is_rejected(self, reranker):
        """`EpisodeReranker` has exactly two members. Offering a third on the wire
        would be a promise the config cannot keep, so it keeps the ValueError
        contract every other invalid combination gets."""
        with pytest.raises(ValueError, match=f"Invalid search_mode='episodes' \\+ reranker='{reranker}'"):
            resolve_search_config('episodes', reranker, 10)

    @pytest.mark.parametrize('reranker', ['mmr', 'node_distance', 'episode_mentions'])
    def test_the_rejection_still_lists_the_valid_combinations(self, reranker):
        with pytest.raises(ValueError, match='Valid combinations'):
            resolve_search_config('episodes', reranker, 10)

    def test_both_valid_pairs_are_registered(self):
        assert ('episodes', 'rrf') in SEARCH_RECIPES
        assert ('episodes', 'cross_encoder') in SEARCH_RECIPES

    def test_no_invalid_pair_is_registered(self):
        registered = {r for mode, r in SEARCH_RECIPES if mode == 'episodes'}
        assert registered == {'rrf', 'cross_encoder'}

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

class TestTheCapabilityIsAnnounced:
    """ADR-019/020: an agent learns the surface from the server, not from us."""

    def test_the_tool_catalog_says_the_source_narratives_are_searchable(self):
        from tool_descriptions import _key_tools_lines

        catalog = '\n'.join(_key_tools_lines()).lower()
        search_entry = catalog.split('1. search')[1].split('2. explore_entity')[0]

        assert 'episode' in search_entry, (
            'an agent that cannot learn the narratives are searchable will not search them'
        )

    def test_the_search_docstring_documents_the_mode_and_the_intent(self):
        doc = search.__doc__ or ''

        assert 'episodes' in doc
        assert 'narrative' in doc


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
