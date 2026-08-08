"""BUG-57 regression: a community rebuild for one group_id must not wipe the others.

`Graphiti.build_communities(group_ids=['a'])` clears the existing communities before
rebuilding. Until this fix the clear was `MATCH (c:Community) DETACH DELETE c` with no
filter at all, so on any store that holds more than one partition — which is every AGE
deployment, since AGE keeps every group_id in ONE graph — rebuilding group A silently
destroyed group B's whole community layer and never rebuilt it.

These run against the live AGE bed (the `age_driver` fixture builds and drops its own
uniquely-named graph); they self-skip when the bed is unreachable.
"""

import pytest

from graphiti_core.utils.maintenance.community_operations import remove_communities

GROUP_A = 'bug57groupa'
GROUP_B = 'bug57groupb'


async def _seed_two_groups(driver) -> None:
    """One Community per group, each DETACH-able (it owns a HAS_MEMBER edge)."""
    for group in (GROUP_A, GROUP_B):
        await driver.execute_query(
            f"CREATE (c:Community {{uuid: 'c-{group}', name: 'community {group}', "
            f"group_id: '{group}'}})"
        )
        await driver.execute_query(
            f"CREATE (e:Entity {{uuid: 'e-{group}', name: 'entity {group}', "
            f"group_id: '{group}'}})"
        )
        await driver.execute_query(
            f"MATCH (c:Community {{uuid: 'c-{group}'}}), (e:Entity {{uuid: 'e-{group}'}}) "
            f"CREATE (c)-[:HAS_MEMBER {{group_id: '{group}'}}]->(e)"
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
async def test_remove_communities_scoped_to_one_group_leaves_the_other_intact(age_driver):
    await _seed_two_groups(age_driver)
    assert await _community_group_ids(age_driver) == {GROUP_A, GROUP_B}
    assert await _has_member_count(age_driver) == 2

    await remove_communities(age_driver, group_ids=[GROUP_A])

    # The failure this pins: before the fix BOTH groups came back empty here.
    assert await _community_group_ids(age_driver) == {GROUP_B}
    assert await _has_member_count(age_driver) == 1


@pytest.mark.asyncio
async def test_remove_communities_without_group_ids_still_clears_the_whole_graph(age_driver):
    """The unscoped call is still supported — it is what a full rebuild needs."""
    await _seed_two_groups(age_driver)
    assert await _community_group_ids(age_driver) == {GROUP_A, GROUP_B}

    await remove_communities(age_driver)

    assert await _community_group_ids(age_driver) == set()
    assert await _has_member_count(age_driver) == 0


@pytest.mark.asyncio
async def test_remove_communities_with_an_unknown_group_id_deletes_nothing(age_driver):
    """An empty *match* must never widen into an empty *filter*."""
    await _seed_two_groups(age_driver)

    await remove_communities(age_driver, group_ids=['bug57nosuchgroup'])

    assert await _community_group_ids(age_driver) == {GROUP_A, GROUP_B}
    assert await _has_member_count(age_driver) == 2
