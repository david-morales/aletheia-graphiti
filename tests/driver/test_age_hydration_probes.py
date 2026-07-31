"""Bounded diagnosis: does the AGE READ path ever drop node attributes?

Companion to `test_age_attribute_persistence.py`, which covers the WRITE side of
the 2026-08-01 attribute-loss bug. These probes ask the other half of the
question: attributes are stored correctly — do they survive hydration on the two
paths production actually reads through?

  (a) `node_get_by_uuids`      — the bulk get used by every consumer
  (b) `search_interface.node_fulltext_search` — the search path, which hydrates
      through `_hydrate_nodes_in_order` -> `node_get_by_uuids`

Both passing means hydration is intact and the wipe entered only via re-saves.
Live-gated through the shared `age_driver` fixture (tests/driver/conftest.py):
a per-test `g_<hex>` graph on AGE_TEST_DSN, dropped in teardown, skipped when
the store is unreachable.
"""

from datetime import datetime, timezone

import pytest

ATTRS = {'nacimiento_fecha': '1980-01-01', 'telefono': '600000000', 'sexo': 'F'}


def _entity(driver, uuid, name):
    from graphiti_core.nodes import EntityNode

    n = EntityNode(
        uuid=uuid, name=name, group_id='attrloss',
        labels=['Entity', 'Persona'], created_at=datetime.now(timezone.utc),
        summary='persona detenida', attributes=dict(ATTRS),
    )
    n.name_embedding = [0.1] * driver.embedding_dim
    return n


@pytest.mark.asyncio
async def test_get_by_uuids_carries_attributes(age_driver):
    """Probe (a): the bulk get hydrates the stored attribute map."""
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    await ops.node_save(_entity(age_driver, 'attrloss-hyd-1', 'PERSONA HIDRATADA'), age_driver)

    got = await ops.node_get_by_uuids(EntityNode, age_driver, ['attrloss-hyd-1'])
    assert len(got) == 1
    assert got[0].attributes == ATTRS


@pytest.mark.asyncio
async def test_fulltext_search_result_carries_attributes(age_driver):
    """Probe (b): the search path hydrates attributes too.

    `node_save` writes both the AGE vertex and its `_node_tbl` search row, so the
    fulltext row exists exactly as production creates it.
    """
    ops = age_driver.graph_operations_interface
    await ops.node_save(_entity(age_driver, 'attrloss-hyd-2', 'KHADIJA DAOUD'), age_driver)

    found = await age_driver.search_interface.node_fulltext_search(
        age_driver, 'KHADIJA', None, group_ids=['attrloss'], limit=10
    )
    assert [n.uuid for n in found] == ['attrloss-hyd-2']
    assert found[0].attributes == ATTRS
