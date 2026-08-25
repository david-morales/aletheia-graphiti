"""BUG-57 — the community wipe must carry a group filter, and only where it runs.

The behaviour is proved against live stores by
`tests/driver/test_age_community_group_scope.py` and
`tests/driver/test_falkordb_community_group_scope.py` — one per flavour, because the
two reach the delete by different routes. This module asserts what a live test cannot:
that `Graphiti.build_communities` forwards the group_ids it was called with instead of
dropping them (the original defect — the scope existed at the call site and was simply
not passed down), and that `[]` and `None` stay different arguments.

WHICH PATH ACTUALLY RUNS. `remove_communities` consults exactly one thing:
`driver.graph_operations_interface`. Only `AGEDriver` sets it (`age_driver.py`), and
`AGEGraphOperations` does not override `remove_communities`, so the capability check
(`interface_dispatch.implements`, BUG-108) is False and AGE falls through too.
**Every flavour therefore reaches the generic query in `community_operations`** —
the per-driver
`*GraphMaintenanceOperations.remove_communities` methods are a DIFFERENT abstraction
(`driver.graph_ops`, `driver/operations/graph_ops.py`) that nothing calls for this
operation. Their tests live below under a heading that says so, because a green
parametrised suite over four dead implementations is exactly the shape of coverage
that is not coverage.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.falkordb.operations.graph_ops import FalkorGraphMaintenanceOperations
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
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
# the per-driver overrides — a DORMANT surface
#
# These four implement `GraphMaintenanceOperations` (driver.graph_ops), which nothing
# calls for remove_communities; the live path is the generic query below. They are kept
# rather than deleted because the method is an @abstractmethod on an upstream ABC, so
# removing it from the concrete classes would mean diverging the ABC too — merge cost
# for a maintained fork, in exchange for deleting code that is already correct.
#
# Kept, therefore tested for CONSISTENCY, not for protection: four dormant
# implementations that disagree with the live one are a trap for whoever wires them.
# The test names say `the_dormant_override`, and `test_the_live_path_does_not_go
# _through_the_dormant_overrides` pins the dormancy itself, so wiring them turns this
# module red and forces the claims here to be re-read.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('ops_cls', ALL_OVERRIDES)
async def test_the_dormant_override_scopes_the_delete_when_group_ids_are_given(ops_cls):
    executor = RecordingExecutor()
    ops = ops_cls() if ops_cls is not NeptuneGraphMaintenanceOperations else ops_cls(driver=None)

    await ops.remove_communities(executor, group_ids=['alpha'])

    assert 'c.group_id IN $group_ids' in _normalised(executor.only_query)
    assert executor.only_params == {'group_ids': ['alpha']}


@pytest.mark.parametrize('ops_cls', ALL_OVERRIDES)
async def test_the_dormant_override_deletes_everything_when_no_group_ids_are_given(ops_cls):
    executor = RecordingExecutor()
    ops = ops_cls() if ops_cls is not NeptuneGraphMaintenanceOperations else ops_cls(driver=None)

    await ops.remove_communities(executor)

    assert _normalised(executor.only_query) == 'MATCH (c:Community) DETACH DELETE c'
    assert executor.only_params == {}


@pytest.mark.parametrize('ops_cls', ALL_OVERRIDES)
async def test_the_dormant_override_deletes_nothing_for_an_empty_group_list(ops_cls):
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


async def test_the_live_path_does_not_go_through_the_dormant_overrides():
    """Pin the dormancy itself, so the three tests above cannot be mistaken for proof.

    `remove_communities` consults `graph_operations_interface` and nothing else. A
    driver that exposes a perfectly correct `graph_ops.remove_communities` still has it
    ignored — which is why the FalkorDB protection had to be proved against a real
    driver (`tests/driver/test_falkordb_community_group_scope.py`) and not against
    `FalkorGraphMaintenanceOperations`.

    If someone later wires `graph_ops` into this call, this test goes red. That is the
    intent: the wiring may well be right, but the coverage claims in this module would
    then be wrong and need rewriting.
    """
    driver = RecordingExecutor()
    driver.graph_ops = MagicMock()
    driver.graph_ops.remove_communities = AsyncMock(return_value=None)

    await remove_communities(driver, group_ids=['alpha'])

    driver.graph_ops.remove_communities.assert_not_awaited()
    assert 'c.group_id IN $group_ids' in _normalised(driver.only_query)


def test_the_age_operations_class_does_not_override_remove_communities():
    """The other half of the reachability premise, asserted on the class.

    AGE is the ONLY driver that sets `graph_operations_interface`, so if
    `AGEGraphOperations` ever defines `remove_communities` the AGE flavour stops using
    the generic query and the live AGE test starts covering a different path than the
    one this module describes.
    """
    from graphiti_core.driver.graph_operations.age_graph_operations import AGEGraphOperations
    from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface

    assert 'remove_communities' not in vars(AGEGraphOperations), (
        'AGEGraphOperations now implements remove_communities: the AGE flavour no '
        'longer falls through to the generic query, so re-read this module and '
        'tests/driver/test_age_community_group_scope.py before trusting either.'
    )
    # ...and the base it inherits still refuses, which is what makes the fall-through happen.
    assert 'remove_communities' in vars(GraphOperationsInterface)


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


async def test_generic_fallback_still_runs_when_the_interface_does_not_override():
    """The fall-through condition is a MISSING override, not a caught exception.

    This is the AGE path: `AGEGraphOperations` inherits `remove_communities` from
    the base and every flavour therefore reaches the generic query.
    """
    driver = RecordingExecutor()
    driver.graph_operations_interface = GraphOperationsInterface()

    await remove_communities(driver, group_ids=['alpha'])

    assert 'c.group_id IN $group_ids' in _normalised(driver.only_query)


async def test_a_failing_override_is_not_swallowed_into_the_generic_query():
    """BUG-108: an interface that HAS the method owns its errors.

    The old idiom read any `NotImplementedError` as "not implemented" and
    re-issued the generic delete against the same driver — after the interface's
    own delete had already run.
    """
    driver = RecordingExecutor()
    driver.graph_operations_interface = MagicMock()
    driver.graph_operations_interface.remove_communities = AsyncMock(
        side_effect=NotImplementedError('helper is missing')
    )

    with pytest.raises(NotImplementedError, match='helper is missing'):
        await remove_communities(driver, group_ids=['alpha'])

    assert driver.calls == []


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
