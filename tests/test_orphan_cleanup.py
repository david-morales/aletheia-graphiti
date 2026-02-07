"""Tests for orphan node detection and removal."""

import pytest

from unittest.mock import AsyncMock

from graphiti_core.utils.maintenance.graph_data_operations import (
    detect_orphan_nodes,
    remove_orphan_nodes,
)


@pytest.fixture
def mock_driver():
    driver = AsyncMock()
    driver.execute_query = AsyncMock()
    return driver


@pytest.mark.asyncio
async def test_detect_no_orphans(mock_driver):
    """Empty result from driver means no orphans detected."""
    mock_driver.execute_query.return_value = ([], None, None)

    result = await detect_orphan_nodes(mock_driver)

    assert result == []
    mock_driver.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_detect_with_orphans(mock_driver):
    """Records returned from driver are mapped to a list of UUIDs."""
    mock_driver.execute_query.return_value = (
        [{'uuid': 'aaa'}, {'uuid': 'bbb'}, {'uuid': 'ccc'}],
        None,
        None,
    )

    result = await detect_orphan_nodes(mock_driver)

    assert result == ['aaa', 'bbb', 'ccc']


@pytest.mark.asyncio
async def test_detect_with_group_id(mock_driver):
    """When group_id is provided, it is forwarded as a query parameter."""
    mock_driver.execute_query.return_value = (
        [{'uuid': 'xxx'}],
        None,
        None,
    )

    result = await detect_orphan_nodes(mock_driver, group_id='g1')

    assert result == ['xxx']
    # The call should include the group_id keyword argument
    call_kwargs = mock_driver.execute_query.call_args
    assert call_kwargs.kwargs.get('group_id') == 'g1'


@pytest.mark.asyncio
async def test_remove_orphans(mock_driver):
    """remove_orphan_nodes detects then deletes orphan nodes."""
    # First call: detect (returns orphan UUIDs)
    # Second call: delete
    mock_driver.execute_query.side_effect = [
        ([{'uuid': 'u1'}, {'uuid': 'u2'}], None, None),
        ([], None, None),
    ]

    result = await remove_orphan_nodes(mock_driver)

    assert result == ['u1', 'u2']
    assert mock_driver.execute_query.call_count == 2

    # Second call should be the DELETE query with the orphan UUIDs
    delete_call = mock_driver.execute_query.call_args_list[1]
    assert delete_call.kwargs.get('uuids') == ['u1', 'u2']


@pytest.mark.asyncio
async def test_remove_no_orphans(mock_driver):
    """When no orphans are detected, no delete query is issued."""
    mock_driver.execute_query.return_value = ([], None, None)

    result = await remove_orphan_nodes(mock_driver)

    assert result == []
    # Only the detect query should have been called
    assert mock_driver.execute_query.call_count == 1
