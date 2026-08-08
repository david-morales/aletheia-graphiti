"""BUG-57 — the community wipe must carry a group filter on every code path.

Offline companion to `tests/driver/test_age_community_group_scope.py`, which proves the
behaviour against a live store. These assert the two things a live test on ONE backend
cannot: that the generated Cypher of every per-driver override carries the filter, and
that `Graphiti.build_communities` actually forwards the group_ids it was called with
instead of dropping them (the original defect — the scope existed at the call site and
was simply not passed down).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.falkordb.operations.graph_ops import FalkorGraphMaintenanceOperations
from graphiti_core.driver.kuzu.operations.graph_ops import KuzuGraphMaintenanceOperations
from graphiti_core.driver.neo4j.operations.graph_ops import Neo4jGraphMaintenanceOperations
from graphiti_core.driver.neptune.operations.graph_ops import NeptuneGraphMaintenanceOperations
from graphiti_core.utils.maintenance.community_operations import remove_communities


class RecordingExecutor:
    """Captures the query text and params every call was made with."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # `remove_communities` consults this before falling through to execute_query.
        self.graph_operations_interface = None

    async def execute_query(self, query: str, **kwargs: Any):
        self.calls.append((query, kwargs))
        return [], [], None

    @property
    def only_query(self) -> str:
        assert len(self.calls) == 1, self.calls
        return self.calls[0][0]

    @property
    def only_params(self) -> dict[str, Any]:
        assert len(self.calls) == 1, self.calls
        return self.calls[0][1]


ALL_OVERRIDES = [
    pytest.param(FalkorGraphMaintenanceOperations, id='falkordb'),
    pytest.param(Neo4jGraphMaintenanceOperations, id='neo4j'),
    pytest.param(KuzuGraphMaintenanceOperations, id='kuzu'),
    pytest.param(NeptuneGraphMaintenanceOperations, id='neptune'),
]


def _normalised(query: str) -> str:
    return ' '.join(query.split())


# ---------------------------------------------------------------------------
# the per-driver overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('ops_cls', ALL_OVERRIDES)
async def test_override_scopes_the_delete_when_group_ids_are_given(ops_cls):
    executor = RecordingExecutor()
    ops = ops_cls() if ops_cls is not NeptuneGraphMaintenanceOperations else ops_cls(driver=None)

    await ops.remove_communities(executor, group_ids=['alpha'])

    assert 'c.group_id IN $group_ids' in _normalised(executor.only_query)
    assert executor.only_params == {'group_ids': ['alpha']}


@pytest.mark.parametrize('ops_cls', ALL_OVERRIDES)
async def test_override_deletes_everything_when_no_group_ids_are_given(ops_cls):
    executor = RecordingExecutor()
    ops = ops_cls() if ops_cls is not NeptuneGraphMaintenanceOperations else ops_cls(driver=None)

    await ops.remove_communities(executor)

    assert _normalised(executor.only_query) == 'MATCH (c:Community) DETACH DELETE c'
    assert executor.only_params == {}


@pytest.mark.parametrize('ops_cls', ALL_OVERRIDES)
async def test_override_deletes_nothing_for_an_empty_group_list(ops_cls):
    """`[]` is an empty list of partitions, and deletes none of them.

    It was briefly read as "no scope given" — i.e. as `None` — on the theory that
    `IN []` would turn a full rebuild into a silent no-op. That theory was false in the
    dangerous direction: `get_community_clusters` iterates `for group_id in group_ids`
    and so builds nothing from an empty list, meaning the "protected" full rebuild
    deleted every community and restored none. The delete must stay filtered.
    """
    executor = RecordingExecutor()
    ops = ops_cls() if ops_cls is not NeptuneGraphMaintenanceOperations else ops_cls(driver=None)

    await ops.remove_communities(executor, group_ids=[])

    assert 'c.group_id IN $group_ids' in _normalised(executor.only_query)
    assert executor.only_params == {'group_ids': []}


# ---------------------------------------------------------------------------
# the generic fallback in community_operations
# ---------------------------------------------------------------------------


async def test_generic_fallback_scopes_the_delete():
    driver = RecordingExecutor()

    await remove_communities(driver, group_ids=['alpha', 'beta'])

    assert 'c.group_id IN $group_ids' in _normalised(driver.only_query)
    assert driver.only_params == {'group_ids': ['alpha', 'beta']}


async def test_generic_fallback_without_group_ids_is_unscoped():
    driver = RecordingExecutor()

    await remove_communities(driver)

    assert _normalised(driver.only_query) == 'MATCH (c:Community) DETACH DELETE c'


async def test_generic_fallback_deletes_nothing_for_an_empty_group_list():
    """`None` and `[]` are different arguments and must stay different deletes."""
    driver = RecordingExecutor()

    await remove_communities(driver, group_ids=[])

    assert 'c.group_id IN $group_ids' in _normalised(driver.only_query)
    assert driver.only_params == {'group_ids': []}


async def test_generic_fallback_forwards_group_ids_to_the_driver_interface():
    driver = RecordingExecutor()
    driver.graph_operations_interface = MagicMock()
    driver.graph_operations_interface.remove_communities = AsyncMock(return_value=None)

    await remove_communities(driver, group_ids=['alpha'])

    driver.graph_operations_interface.remove_communities.assert_awaited_once_with(
        driver, ['alpha']
    )
    assert driver.calls == []


async def test_generic_fallback_still_falls_through_on_not_implemented():
    driver = RecordingExecutor()
    driver.graph_operations_interface = MagicMock()
    driver.graph_operations_interface.remove_communities = AsyncMock(
        side_effect=NotImplementedError
    )

    await remove_communities(driver, group_ids=['alpha'])

    assert 'c.group_id IN $group_ids' in _normalised(driver.only_query)


# ---------------------------------------------------------------------------
# the call site — the actual defect
# ---------------------------------------------------------------------------


async def test_build_communities_forwards_its_group_ids_to_the_wipe(monkeypatch):
    """The bug in one assertion: `build_communities(group_ids=['a'])` used to clear
    with `remove_communities(driver)` — no scope — so every other partition went with
    it."""
    import graphiti_core.graphiti as graphiti_module

    seen: dict[str, Any] = {}

    async def _fake_remove(driver, group_ids=None):
        seen['group_ids'] = group_ids

    async def _fake_build(driver, llm_client, group_ids):
        return [], []

    monkeypatch.setattr(graphiti_module, 'remove_communities', _fake_remove)
    monkeypatch.setattr(graphiti_module, 'build_communities', _fake_build)

    graphiti = MagicMock()
    graphiti.clients.driver.provider = 'notfalkordb'
    graphiti.max_coroutines = 1

    await graphiti_module.Graphiti.build_communities(
        graphiti, group_ids=['alpha'], driver=MagicMock()
    )

    assert seen['group_ids'] == ['alpha']


async def test_build_communities_with_an_empty_group_list_touches_nothing():
    """The pair that makes the `[]` ruling checkable end to end, on a REAL driver
    double rather than a monkeypatched `remove_communities`.

    `build_communities([])` clusters nothing — `get_community_clusters` iterates the
    list — so the only thing it can do is delete. Reading `[]` as "no scope" therefore
    produced the purest form of the BUG-57 failure: wipe everything, restore nothing.
    Deleting nothing is the only outcome consistent with what the caller asked for.
    """
    import graphiti_core.graphiti as graphiti_module

    driver = RecordingExecutor()
    # `handle_multiple_group_ids` reads the provider off the driver to decide whether to
    # fan out per group_id (FalkorDB maps group_id -> database). Anything else takes the
    # single-call path, which is the one under test here.
    driver.provider = 'notfalkordb'

    graphiti = MagicMock()
    graphiti.clients.driver = driver
    graphiti.llm_client = MagicMock()
    graphiti.max_coroutines = 1

    nodes, edges = await graphiti_module.Graphiti.build_communities(
        graphiti, group_ids=[], driver=driver
    )

    assert nodes == []
    assert edges == []
    # One query, and it deletes nothing: the filter is present and matches no partition.
    destructive = [q for q, _ in driver.calls if 'DETACH DELETE' in q]
    assert len(destructive) == 1, driver.calls
    assert 'c.group_id IN $group_ids' in _normalised(destructive[0]), destructive[0]
    assert driver.calls[0][1] == {'group_ids': []}
