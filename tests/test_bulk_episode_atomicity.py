"""
Tests for atomicity of add_episode_bulk: episodes must NOT be written
to the database before LLM extraction completes.

If extraction raises, no call to add_nodes_and_edges_bulk should occur.
Previously, an early save at line 1118 wrote episodes before extraction,
leaving orphaned episode nodes in FalkorDB when extraction failed.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode

pytest_plugins = ('pytest_asyncio',)


def _make_raw_episode(name: str = 'test episode') -> RawEpisode:
    return RawEpisode(
        name=name,
        content='Alice knows Bob.',
        source_description='test',
        source=EpisodeType.message,
        reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _make_graphiti(mock_driver: MagicMock, mock_llm: MagicMock, mock_embedder: MagicMock):
    """Build a Graphiti instance with mocked driver, LLM, and embedder."""
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.graphiti import Graphiti

    mock_cross_encoder = MagicMock(spec=CrossEncoderClient)

    graphiti = Graphiti(
        graph_driver=mock_driver,
        llm_client=mock_llm,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder,
    )
    return graphiti


def _make_mock_driver() -> MagicMock:
    """Build a mock GraphDriver that satisfies Graphiti's internal calls.

    Must use spec=GraphDriver so that isinstance(mock, GraphDriver) returns
    True — required by the GraphitiClients Pydantic model.
    """
    from graphiti_core.driver.driver import GraphDriver, GraphProvider

    mock_driver = MagicMock(spec=GraphDriver)
    mock_driver.provider = GraphProvider.NEO4J
    mock_driver._database = 'test_db'
    mock_driver.clone = MagicMock(return_value=mock_driver)

    # ensure_edge_type_index is awaitable
    mock_driver.ensure_edge_type_index = AsyncMock(return_value=None)

    return mock_driver


def _make_mock_llm() -> MagicMock:
    from graphiti_core.llm_client import LLMClient

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.config = MagicMock()
    mock_llm.model = 'test-model'
    mock_llm.small_model = 'test-small-model'
    mock_llm.temperature = 0.0
    mock_llm.max_tokens = 1000
    mock_llm.cache_enabled = False
    mock_llm.cache_dir = None
    mock_llm.set_tracer = MagicMock()
    mock_llm.generate_response = AsyncMock(
        return_value={
            'tool_calls': [
                {
                    'name': 'extract_entities',
                    'arguments': {'entities': []},
                }
            ]
        }
    )
    return mock_llm


def _make_mock_embedder() -> MagicMock:
    from graphiti_core.embedder.client import EmbedderClient

    mock_embedder = MagicMock(spec=EmbedderClient)
    mock_embedder.create = AsyncMock(return_value=[0.1] * 384)
    return mock_embedder


@pytest.mark.asyncio
async def test_no_episodes_saved_on_extraction_failure():
    """
    When LLM extraction raises an exception, add_nodes_and_edges_bulk must
    NOT have been called at all — no orphan episode nodes in the database.

    Before the fix: add_nodes_and_edges_bulk was called once (early save)
    before extraction, leaving orphaned episodes.
    After the fix: the early save is removed; the single final save is never
    reached because extraction raised first.
    """
    mock_driver = _make_mock_driver()
    mock_llm = _make_mock_llm()
    mock_embedder = _make_mock_embedder()

    graphiti = _make_graphiti(mock_driver, mock_llm, mock_embedder)

    bulk_episodes = [_make_raw_episode('episode-1'), _make_raw_episode('episode-2')]

    extraction_error = RuntimeError('LLM extraction failed')

    with (
        # Patch the save function at the import site used by graphiti.py
        patch(
            'graphiti_core.graphiti.add_nodes_and_edges_bulk',
            new_callable=AsyncMock,
        ) as mock_save,
        # Patch retrieve_previous_episodes_bulk so it doesn't need a real DB
        patch(
            'graphiti_core.graphiti.retrieve_previous_episodes_bulk',
            new_callable=AsyncMock,
            return_value=[],
        ),
        # Patch _extract_and_dedupe_nodes_bulk to simulate extraction failure
        patch.object(
            graphiti,
            '_extract_and_dedupe_nodes_bulk',
            new_callable=AsyncMock,
            side_effect=extraction_error,
        ),
    ):
        with pytest.raises(RuntimeError, match='LLM extraction failed'):
            await graphiti.add_episode_bulk(bulk_episodes, group_id='test_group')

        # KEY ASSERTION: no DB write must have occurred before extraction failed
        mock_save.assert_not_called()
