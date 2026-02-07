"""Tests for handle_multiple_group_ids decorator — single group_id cloning fix."""

from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.decorators import handle_multiple_group_ids
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_config import SearchResults


class FakeService:
    """Minimal stand-in for a Graphiti instance with a FalkorDB driver."""

    def __init__(self, driver_mock):
        self.clients = MagicMock()
        self.clients.driver = driver_mock

    @handle_multiple_group_ids
    async def search_method(self, query: str, *, group_ids: list[str] | None = None, driver=None):
        """Decorated method that mimics a search call."""
        return SearchResults(nodes=[], edges=[], communities=[])


def _make_falkordb_driver():
    """Return a MagicMock that looks like a FalkorDB GraphDriver."""
    driver = MagicMock()
    driver.provider = GraphProvider.FALKORDB
    driver.clone.return_value = MagicMock()
    return driver


async def _fake_semaphore_gather(*coroutines, max_coroutines=None):
    """Stand-in for semaphore_gather that awaits coroutines sequentially."""
    return [await coro for coro in coroutines]


@pytest.mark.asyncio
async def test_single_group_id_clones_driver():
    """A single-element group_ids list must still trigger driver.clone on FalkorDB."""
    driver = _make_falkordb_driver()
    svc = FakeService(driver)

    with patch('graphiti_core.decorators.semaphore_gather', _fake_semaphore_gather):
        await svc.search_method('test query', group_ids=['single_graph'])

    driver.clone.assert_called_once_with(database='single_graph')


@pytest.mark.asyncio
async def test_multiple_group_ids_clones_driver_for_each():
    """Multiple group_ids should clone the driver once per group_id."""
    driver = _make_falkordb_driver()
    svc = FakeService(driver)

    with patch('graphiti_core.decorators.semaphore_gather', _fake_semaphore_gather):
        await svc.search_method('test query', group_ids=['graph_a', 'graph_b'])

    assert driver.clone.call_count == 2
    driver.clone.assert_any_call(database='graph_a')
    driver.clone.assert_any_call(database='graph_b')


@pytest.mark.asyncio
async def test_neo4j_driver_skips_cloning():
    """Non-FalkorDB providers should NOT trigger cloning, even with group_ids."""
    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    svc = FakeService(driver)

    result = await svc.search_method('test query', group_ids=['some_graph'])

    driver.clone.assert_not_called()
    assert isinstance(result, SearchResults)
