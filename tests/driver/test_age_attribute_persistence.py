"""Regression: AGE node re-saves must not erase stored attributes.

2026-08-01 attribute-loss bug: a node saved with attributes, then re-saved
(as entity resolution does) with an empty in-memory attributes map, ended
with attributes = {} in the graph. AGE stores attributes as ONE nested map
and `SET n += {...}` replaces a nested map wholesale.

Live-gated: needs the Postgres+AGE store (AGE_TEST_DSN, default
postgresql://age:age@localhost:5434/age_test). Throwaway graph only.
"""

import os
from datetime import datetime, timezone

import pytest
import pytest_asyncio

AGE_DSN = os.environ.get('AGE_TEST_DSN', 'postgresql://age:age@localhost:5434/age_test')
GRAPH = 'attrloss_repro_test'


@pytest_asyncio.fixture
async def driver():
    """Connected AGEDriver on the throwaway `attrloss_` graph, dropped afterwards.

    Skips (never fails) when the store is unreachable.
    """
    from graphiti_core.driver.age_driver import AGEDriver

    d = AGEDriver(dsn=AGE_DSN, graph_name=GRAPH, embedding_dim=8)
    try:
        # Idempotent: creates the throwaway graph + its shadow tables.
        await d.build_indices_and_constraints()
        await d.execute_query('RETURN 1')
    except Exception as exc:  # store down => skip, never fail
        await d.close()
        pytest.skip(f'AGE store not reachable: {exc}')
    try:
        yield d
    finally:
        await d.drop_graph()
        await d.close()


def _entity(uuid, attributes, summary=''):
    from graphiti_core.nodes import EntityNode

    n = EntityNode(
        uuid=uuid, name='PERSONA UNO', group_id='attrloss',
        labels=['Entity', 'Persona'], created_at=datetime.now(timezone.utc),
        summary=summary, attributes=attributes,
    )
    n.name_embedding = None
    return n


@pytest.mark.asyncio
async def test_resave_with_empty_attributes_preserves_stored(driver):
    from graphiti_core.nodes import EntityNode

    ops = driver.graph_operations_interface
    await ops.node_save(
        _entity('attrloss-1', {'nacimiento_fecha': '1980-01-01', 'telefono': '600000000'}), driver
    )
    await ops.node_save(_entity('attrloss-1', {}, summary='seen again'), driver)
    got = await ops.node_get_by_uuid(EntityNode, driver, 'attrloss-1')
    assert got.attributes.get('nacimiento_fecha') == '1980-01-01'
    assert got.attributes.get('telefono') == '600000000'
    assert got.summary == 'seen again'  # non-attribute props must still update


@pytest.mark.asyncio
async def test_resave_merges_per_key_incoming_wins(driver):
    from graphiti_core.nodes import EntityNode

    ops = driver.graph_operations_interface
    await ops.node_save(_entity('attrloss-2', {'telefono': '600000000', 'sexo': 'F'}), driver)
    await ops.node_save(_entity('attrloss-2', {'telefono': '699999999'}), driver)
    got = await ops.node_get_by_uuid(EntityNode, driver, 'attrloss-2')
    assert got.attributes.get('telefono') == '699999999'  # incoming non-empty wins
    assert got.attributes.get('sexo') == 'F'              # stored key survives


def _bulk_fields(uuid, attributes, summary=''):
    """The flat dict shape the bulk path passes to `_write_entity_from_fields`:
    known node keys plus the node's attributes flattened at the top level
    (see `neo4j/operations/entity_node_ops.py::save_bulk`)."""
    d = {
        'uuid': uuid,
        'name': 'PERSONA UNO',
        'group_id': 'attrloss',
        'summary': summary,
        'created_at': datetime.now(timezone.utc),
        'name_embedding': None,
        'labels': ['Entity', 'Persona'],
    }
    d.update(attributes)
    return d


@pytest.mark.asyncio
async def test_bulk_resave_with_empty_attributes_preserves_stored(driver):
    """The narrative path writes through `node_save_bulk` -> `_write_entity_from_fields`,
    which has the same `SET n +=` shape and so must carry the same guard."""
    from graphiti_core.nodes import EntityNode

    ops = driver.graph_operations_interface
    await ops.node_save(_entity('attrloss-3', {'nacimiento_fecha': '1980-01-01', 'sexo': 'F'}), driver)
    await ops.node_save_bulk(
        EntityNode, driver, None, [_bulk_fields('attrloss-3', {}, summary='seen again')]
    )
    got = await ops.node_get_by_uuid(EntityNode, driver, 'attrloss-3')
    assert got.attributes.get('nacimiento_fecha') == '1980-01-01'
    assert got.attributes.get('sexo') == 'F'
    assert got.summary == 'seen again'


@pytest.mark.asyncio
async def test_bulk_resave_merges_per_key_incoming_wins(driver):
    from graphiti_core.nodes import EntityNode

    ops = driver.graph_operations_interface
    await ops.node_save(_entity('attrloss-4', {'telefono': '600000000', 'sexo': 'F'}), driver)
    await ops.node_save_bulk(
        EntityNode, driver, None, [_bulk_fields('attrloss-4', {'telefono': '699999999'})]
    )
    got = await ops.node_get_by_uuid(EntityNode, driver, 'attrloss-4')
    assert got.attributes.get('telefono') == '699999999'
    assert got.attributes.get('sexo') == 'F'
