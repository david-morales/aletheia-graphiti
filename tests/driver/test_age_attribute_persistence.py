"""Regression: AGE re-saves must not erase stored attributes.

2026-08-01 attribute-loss bug: a node saved with attributes, then re-saved
(as entity resolution does) with an empty in-memory attributes map, ended
with attributes = {} in the graph. AGE stores attributes as ONE nested map
and `SET n += {...}` replaces a nested map wholesale.

Edges have the identical shape (`SET r += {...}` over an `attributes` nested
map), so both writer families — node and edge, single and bulk — are covered
here.

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


# ================================================================= edge writers
# Edges carry an `attributes` nested map exactly as nodes do, written through the
# same `SET r += {...}` shape — so the same wipe applies and the same guard must.
def _edge(uuid, attributes, fact=''):
    from graphiti_core.edges import EntityEdge

    e = EntityEdge(
        uuid=uuid, source_node_uuid='attrloss-src', target_node_uuid='attrloss-tgt',
        name='ES_DETENIDO', fact=fact, group_id='attrloss',
        created_at=datetime.now(timezone.utc), attributes=attributes,
    )
    e.fact_embedding = None
    return e


def _edge_bulk_fields(uuid, attributes, fact=''):
    """The flat dict shape the bulk path passes to `_write_entity_edge_from_fields`:
    known edge keys plus the edge's attributes flattened at the top level
    (see `neo4j/operations/entity_edge_ops.py::save_bulk`)."""
    d = {
        'uuid': uuid,
        'source_node_uuid': 'attrloss-src',
        'target_node_uuid': 'attrloss-tgt',
        'name': 'ES_DETENIDO',
        'fact': fact,
        'fact_embedding': None,
        'group_id': 'attrloss',
        'episodes': [],
        'created_at': datetime.now(timezone.utc),
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
    }
    d.update(attributes)
    return d


async def _endpoints(driver):
    """Both edge writers do `MATCH (a), (b)`, so the endpoints must exist first."""
    ops = driver.graph_operations_interface
    await ops.node_save(_entity('attrloss-src', {}), driver)
    await ops.node_save(_entity('attrloss-tgt', {}), driver)
    return ops


@pytest.mark.asyncio
async def test_edge_resave_with_empty_attributes_preserves_stored(driver):
    from graphiti_core.edges import EntityEdge

    ops = await _endpoints(driver)
    await ops.edge_save(_edge('attrloss-e1', {'motivo': 'flagrante', 'lugar': 'via publica'}), driver)
    await ops.edge_save(_edge('attrloss-e1', {}, fact='visto de nuevo'), driver)
    got = await ops.edge_get_by_uuid(EntityEdge, driver, 'attrloss-e1')
    assert got.attributes.get('motivo') == 'flagrante'
    assert got.attributes.get('lugar') == 'via publica'
    assert got.fact == 'visto de nuevo'  # non-attribute props must still update


@pytest.mark.asyncio
async def test_edge_resave_merges_per_key_incoming_wins(driver):
    from graphiti_core.edges import EntityEdge

    ops = await _endpoints(driver)
    await ops.edge_save(_edge('attrloss-e2', {'motivo': 'flagrante', 'turno': 'noche'}), driver)
    await ops.edge_save(_edge('attrloss-e2', {'motivo': 'orden judicial'}), driver)
    got = await ops.edge_get_by_uuid(EntityEdge, driver, 'attrloss-e2')
    assert got.attributes.get('motivo') == 'orden judicial'  # incoming non-empty wins
    assert got.attributes.get('turno') == 'noche'            # stored key survives


@pytest.mark.asyncio
async def test_bulk_edge_resave_with_empty_attributes_preserves_stored(driver):
    """The narrative path writes edges through `edge_save_bulk` ->
    `_write_entity_edge_from_fields`, which has the same `SET r +=` shape."""
    from graphiti_core.edges import EntityEdge

    ops = await _endpoints(driver)
    await ops.edge_save(_edge('attrloss-e3', {'motivo': 'flagrante', 'turno': 'noche'}), driver)
    await ops.edge_save_bulk(
        EntityEdge, driver, None, [_edge_bulk_fields('attrloss-e3', {}, fact='visto de nuevo')]
    )
    got = await ops.edge_get_by_uuid(EntityEdge, driver, 'attrloss-e3')
    assert got.attributes.get('motivo') == 'flagrante'
    assert got.attributes.get('turno') == 'noche'
    assert got.fact == 'visto de nuevo'


@pytest.mark.asyncio
async def test_bulk_edge_resave_merges_per_key_incoming_wins(driver):
    from graphiti_core.edges import EntityEdge

    ops = await _endpoints(driver)
    await ops.edge_save(_edge('attrloss-e4', {'motivo': 'flagrante', 'turno': 'noche'}), driver)
    await ops.edge_save_bulk(
        EntityEdge, driver, None, [_edge_bulk_fields('attrloss-e4', {'motivo': 'orden judicial'})]
    )
    got = await ops.edge_get_by_uuid(EntityEdge, driver, 'attrloss-e4')
    assert got.attributes.get('motivo') == 'orden judicial'
    assert got.attributes.get('turno') == 'noche'
