"""BUG-57 on the FalkorDB driver, over a real connection.

The companion to `test_age_community_group_scope.py`. Both flavours need their own
live proof because they reach the delete by different routes: AGE sets
`graph_operations_interface`, FalkorDB leaves it None, and only the second case falls
through to the generic query in `community_operations`. A fix proved on one arm is not
proved on the other.

It is also the answer to a specific trap. `FalkorGraphMaintenanceOperations.
remove_communities` LOOKS like the FalkorDB implementation of this operation and is
not — nothing calls it (see `test_remove_communities_group_scope.py`, the dormant
surface section). Asserting against that class would have produced a green suite over
an untouched code path. This test drives a real `FalkorDriver`.

Gated on an explicit `FALKORDB_TEST_URI`. There is NO localhost:6379 default on
purpose: on a developer machine that port is a real, populated store, and a test that
seeds and DETACH DELETEs Community nodes must never find one by accident.

    docker run -d --name ax_falkor -p 127.0.0.1:16379:6379 \
        -e FALKORDB_ARGS="TIMEOUT 0" falkordb/falkordb:v4.18.0
    FALKORDB_TEST_URI=redis://localhost:16379 python -m pytest \
        tests/driver/test_falkordb_community_group_scope.py
"""

import os
import socket
import uuid
from contextlib import suppress

import pytest
import pytest_asyncio

from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.utils.maintenance.community_operations import remove_communities

FALKORDB_TEST_URI = os.environ.get('FALKORDB_TEST_URI', '')

GROUP_A = 'bug57groupa'
GROUP_B = 'bug57groupb'


def _host_port(uri: str) -> tuple[str, int]:
    rest = uri.split('://', 1)[-1].split('@')[-1].split('/')[0]
    if ':' in rest:
        host, _, port = rest.rpartition(':')
        return host or 'localhost', int(port)
    return rest or 'localhost', 6379


@pytest_asyncio.fixture
async def falkor_driver():
    """A FalkorDriver on a throwaway graph, deleted afterwards."""
    if not FALKORDB_TEST_URI:
        pytest.skip('set FALKORDB_TEST_URI to run the live FalkorDB community tests')
    host, port = _host_port(FALKORDB_TEST_URI)
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError as exc:
        pytest.skip(f'FalkorDB not reachable at {FALKORDB_TEST_URI}: {exc}')

    database = 'bug57' + uuid.uuid4().hex[:10]
    driver = FalkorDriver(host=host, port=port, database=database)
    try:
        yield driver
    finally:
        with suppress(Exception):
            await driver.execute_query('MATCH (n) DETACH DELETE n')
        await driver.close()


async def _seed_two_groups(driver) -> None:
    """One Community per group, each DETACH-able (it owns a HAS_MEMBER edge)."""
    for group in (GROUP_A, GROUP_B):
        await driver.execute_query(
            'CREATE (c:Community {uuid: $cuuid, name: $name, group_id: $group})'
            '-[:HAS_MEMBER {group_id: $group}]->'
            '(e:Entity {uuid: $euuid, name: $ename, group_id: $group})',
            cuuid=f'c-{group}',
            euuid=f'e-{group}',
            name=f'community {group}',
            ename=f'entity {group}',
            group=group,
        )


async def _community_group_ids(driver) -> set[str]:
    records, _, _ = await driver.execute_query(
        'MATCH (c:Community) RETURN c.group_id AS group_id'
    )
    return {r['group_id'] for r in records}


async def _has_member_count(driver) -> int:
    records, _, _ = await driver.execute_query(
        'MATCH (:Community)-[r:HAS_MEMBER]->(:Entity) RETURN count(r) AS n'
    )
    return records[0]['n']


@pytest.mark.asyncio
async def test_the_falkordb_driver_takes_the_generic_scoped_path(falkor_driver):
    """The premise the other two tests rest on, asserted rather than assumed.

    If a future change wired `graph_operations_interface` on this driver, the delete
    would move to a different implementation and these tests would silently stop
    covering the path production uses.
    """
    assert falkor_driver.graph_operations_interface is None


@pytest.mark.asyncio
async def test_remove_communities_scoped_to_one_group_leaves_the_other_intact(falkor_driver):
    await _seed_two_groups(falkor_driver)
    assert await _community_group_ids(falkor_driver) == {GROUP_A, GROUP_B}
    assert await _has_member_count(falkor_driver) == 2

    await remove_communities(falkor_driver, group_ids=[GROUP_A])

    # The failure this pins: before BUG-57 both groups came back empty here.
    assert await _community_group_ids(falkor_driver) == {GROUP_B}
    assert await _has_member_count(falkor_driver) == 1


@pytest.mark.asyncio
async def test_remove_communities_without_group_ids_still_clears_the_whole_graph(falkor_driver):
    """`None` is still the whole graph — that is what a full rebuild needs."""
    await _seed_two_groups(falkor_driver)
    assert await _community_group_ids(falkor_driver) == {GROUP_A, GROUP_B}

    await remove_communities(falkor_driver)

    assert await _community_group_ids(falkor_driver) == set()
    assert await _has_member_count(falkor_driver) == 0


@pytest.mark.asyncio
async def test_remove_communities_with_an_empty_group_list_deletes_nothing(falkor_driver):
    """HIGH-1's ruling on the second flavour: `[]` is no partitions, not no scope."""
    await _seed_two_groups(falkor_driver)

    await remove_communities(falkor_driver, group_ids=[])

    assert await _community_group_ids(falkor_driver) == {GROUP_A, GROUP_B}
    assert await _has_member_count(falkor_driver) == 2
